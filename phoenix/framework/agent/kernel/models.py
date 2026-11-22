"""PHX v2 Hot-Path Classical ML Engines (FR-03, FR-04, FR-06, FR-07, FR-11).

Includes:
  - CalibratedRFRouter (Isotonic/Platt calibrated Random Forest + ONNX/Treelite hook + ECE + Feature Attributions)
  - RFContextPruner (Selective memory item relevance filter before waking LLM)
  - BoostingPlannerHeuristic (LightGBM / HistGBDT P90 latency & utility estimator + adaptive recovery classifier)
  - SafetyGatekeeper (Hard Rule Invariants ALWAYS above probabilistic RF risk gate)
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from phoenix.framework.agent.kernel.features import FEATURE_NAMES
from phoenix.framework.agent.kernel.schema import FeatureAttribution, RecoveryStrategy

try:
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import (
        HistGradientBoostingClassifier,
        HistGradientBoostingRegressor,
        RandomForestClassifier,
    )
    from sklearn.linear_model import LogisticRegression
    HAS_SKLEARN = True
except ImportError:  # pragma: no cover
    HAS_SKLEARN = False

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    lgb = None
    HAS_LIGHTGBM = False


def compute_ece(probs: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE) over predicted top confidences."""
    if len(probs) == 0:
        return 0.0
    probs_arr = np.asarray(probs, dtype=np.float64)
    correct = (np.asarray(y_pred) == np.asarray(y_true)).astype(np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = float(len(probs_arr))
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (probs_arr >= lo) & (probs_arr <= hi if i == n_bins - 1 else probs_arr < hi)
        cnt = int(np.sum(mask))
        if cnt == 0:
            continue
        acc = float(np.mean(correct[mask]))
        conf = float(np.mean(probs_arr[mask]))
        ece += (cnt / n) * abs(acc - conf)
    return round(float(ece), 5)


def _top_feature_attributions(
    importances: Optional[np.ndarray],
    x_vec: np.ndarray,
    feature_names: Sequence[str] = FEATURE_NAMES,
    top_k: int = 4,
) -> List[FeatureAttribution]:
    """Fast per-decision explainability combining global tree split importance with active feature magnitude."""
    if importances is None or len(importances) == 0:
        return []
    imp = np.asarray(importances, dtype=np.float64).ravel()
    x = np.asarray(x_vec, dtype=np.float64).ravel()
    n = min(len(imp), len(x), len(feature_names))
    if n == 0:
        return []
    scores = imp[:n] * (0.25 + np.abs(x[:n]))
    total = float(np.sum(scores))
    if total <= 1e-12:
        scores = imp[:n]
        total = float(np.sum(scores)) or 1.0
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [
        FeatureAttribution(feature=str(feature_names[int(i)]), weight=round(float(scores[int(i)] / total), 4))
        for i in top_idx
        if scores[int(i)] > 0
    ]


class CalibratedRFRouter:
    """FR-03: Calibrated Random Forest Action Router with ONNX export hook and native explainability."""

    def __init__(self, n_estimators: int = 40, max_depth: int = 12, random_state: int = 42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.fitted: bool = False
        self.classes_: List[str] = []
        self._base_rf: Any = None
        self._calibrated: Any = None
        self._onnx_session: Any = None
        self._feature_importances: Optional[np.ndarray] = None
        self.calibration_method: str = "none"

    def fit(self, X: np.ndarray, y: Sequence[str]) -> "CalibratedRFRouter":
        if not HAS_SKLEARN:
            return self
        X_arr = np.asarray(X, dtype=np.float32)
        y_list = [str(lbl) for lbl in y]
        unique_labels = sorted(set(y_list))
        if X_arr.ndim != 2 or len(y_list) < 6 or len(unique_labels) < 1:
            self.fitted = False
            return self

        self.classes_ = unique_labels
        if len(unique_labels) == 1:
            # Degenerate single-action domain: keep safe constant router
            self._base_rf = None
            self._calibrated = None
            self._feature_importances = np.ones(X_arr.shape[1], dtype=np.float64) / X_arr.shape[1]
            self.calibration_method = "single_class"
            self.fitted = True
            return self

        rf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=1,
            random_state=self.random_state,
            n_jobs=1,  # Single-thread keeps Hot Path predict < 1ms without threadpool wakeup overhead
        )
        rf.fit(X_arr, y_list)
        self._base_rf = rf
        self._feature_importances = getattr(rf, "feature_importances_", None)
        self.classes_ = [str(c) for c in rf.classes_]

        # Calibrate probabilities via Isotonic (preferred when >= 10/class) or Platt/Sigmoid
        counts = {c: y_list.count(c) for c in self.classes_}
        min_cnt = min(counts.values())
        if min_cnt >= 3 and len(y_list) >= 12:
            cv_folds = min(3, min_cnt)
            method = "isotonic" if min_cnt >= 8 else "sigmoid"
            try:
                cal = CalibratedClassifierCV(
                    estimator=RandomForestClassifier(
                        n_estimators=self.n_estimators,
                        max_depth=self.max_depth,
                        random_state=self.random_state,
                        n_jobs=1,
                    ),
                    method=method,
                    cv=cv_folds,
                )
                cal.fit(X_arr, y_list)
                self._calibrated = cal
                self.classes_ = [str(c) for c in cal.classes_]
                self.calibration_method = method
            except Exception:
                self._calibrated = rf
                self.calibration_method = "raw_rf"
        else:
            self._calibrated = rf
            self.calibration_method = "raw_rf"

        self.fitted = True
        return self

    def try_export_onnx(self, n_features: int) -> bool:
        """Optional NFR portability hook: compile RF model into ONNX Runtime if skl2onnx + onnxruntime exist."""
        if not self.fitted or self._calibrated is None:
            return False
        try:
            import onnxruntime as ort
            from skl2onnx import convert_sklearn
            from skl2onnx.common.data_types import FloatTensorType

            onx = convert_sklearn(
                self._calibrated,
                initial_types=[("X", FloatTensorType([None, int(n_features)]))],
                options={id(self._calibrated): {"zipmap": False}},
            )
            self._onnx_session = ort.InferenceSession(onx.SerializeToString(), providers=["CPUExecutionProvider"])
            return True
        except Exception:
            self._onnx_session = None
            return False

    def predict(self, x_vec: np.ndarray) -> Tuple[str, float, List[Tuple[str, float]], List[FeatureAttribution]]:
        """Predict `(top_action, calibrated_prob, ranked_actions, top_attributions)` in < 1ms CPU."""
        if not self.fitted or not self.classes_:
            return "", 0.0, [], []
        x = np.asarray(x_vec, dtype=np.float32).reshape(1, -1)
        if len(self.classes_) == 1:
            return self.classes_[0], 0.90, [(self.classes_[0], 0.90)], _top_feature_attributions(self._feature_importances, x)

        if self._onnx_session is not None:
            input_name = self._onnx_session.get_inputs()[0].name
            outputs = self._onnx_session.run(None, {input_name: x})
            probs = np.asarray(outputs[1][0], dtype=np.float64)
        else:
            probs = np.asarray(self._calibrated.predict_proba(x)[0], dtype=np.float64)

        order = np.argsort(probs)[::-1]
        ranked = [(self.classes_[int(i)], round(float(probs[int(i)]), 4)) for i in order]
        top_act, top_p = ranked[0]
        attrs = _top_feature_attributions(self._feature_importances, x)
        return top_act, top_p, ranked, attrs


class RFContextPruner:
    """FR-04: Filters candidate memory items (`Relevant` vs `Noise`) prior to waking the LLM Battery."""

    _PRUNE_FEATURE_NAMES = ("lexical_overlap", "recency_score", "regime_match", "is_error_context", "length_density")

    def __init__(self, threshold: float = 0.45):
        self.threshold = float(threshold)
        self.fitted: bool = False
        self._clf: Any = None
        self._importances: Optional[np.ndarray] = None

    @staticmethod
    def _extract_pair_features(
        query_text: str,
        item_text: str,
        *,
        recency_rank: int = 0,
        total_items: int = 1,
        same_regime: bool = True,
        is_error: bool = False,
    ) -> np.ndarray:
        q_toks = set(re.findall(r"[a-zA-Z0-9_]+|[\u0600-\u06FF]+", (query_text or "").lower()))
        i_toks = set(re.findall(r"[a-zA-Z0-9_]+|[\u0600-\u06FF]+", (item_text or "").lower()))
        overlap = len(q_toks & i_toks) / max(1, len(q_toks))
        recency = 1.0 - (min(recency_rank, max(1, total_items)) / float(max(1, total_items)))
        reg_f = 1.0 if same_regime else 0.0
        err_f = 1.0 if (is_error or "error" in (item_text or "").lower() or "fail" in (item_text or "").lower()) else 0.0
        len_density = min(1.0, len(i_toks) / 40.0)
        return np.array([overlap, recency, reg_f, err_f, len_density], dtype=np.float32)

    def fit(self, X_pairs: np.ndarray, y_relevant: Sequence[int]) -> "RFContextPruner":
        if not HAS_SKLEARN:
            return self
        arr = np.asarray(X_pairs, dtype=np.float32)
        y = np.asarray(y_relevant, dtype=int)
        if arr.ndim != 2 or len(y) < 10 or len(set(y.tolist())) < 2:
            return self
        clf = RandomForestClassifier(n_estimators=20, max_depth=6, random_state=42, n_jobs=1)
        clf.fit(arr, y)
        self._clf = clf
        self._importances = getattr(clf, "feature_importances_", None)
        self.fitted = True
        return self

    def prune(
        self,
        query_text: str,
        memory_items: Sequence[Dict[str, Any]],
        *,
        current_regime: int = 0,
        max_items: int = 6,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Filter candidate memory items down to Relevant items only (< 0.5ms CPU)."""
        if not memory_items:
            return [], {"kept": 0, "dropped": 0, "top_attributions": []}

        total = len(memory_items)
        scored: List[Tuple[float, int, Dict[str, Any], np.ndarray]] = []
        for idx, item in enumerate(memory_items):
            content = str(item.get("content") or item.get("text") or item)
            same_reg = int(item.get("regime_id", current_regime)) == int(current_regime)
            is_err = bool(item.get("is_error", False))
            fv = self._extract_pair_features(
                query_text,
                content,
                recency_rank=total - 1 - idx,
                total_items=total,
                same_regime=same_reg,
                is_error=is_err,
            )
            if self.fitted and self._clf is not None:
                prob = float(self._clf.predict_proba(fv.reshape(1, -1))[0][1])
            else:
                # Deterministic warm heuristic prior to Cold Path training
                prob = float(0.55 * fv[0] + 0.20 * fv[1] + 0.15 * fv[2] + 0.15 * fv[3])
            scored.append((prob, idx, item, fv))

        kept_tuples = [t for t in scored if t[0] >= self.threshold]
        kept_tuples.sort(key=lambda t: t[0], reverse=True)
        kept_tuples = kept_tuples[:max_items]
        # Preserve chronological order among kept items
        kept_tuples.sort(key=lambda t: t[1])
        kept = [t[2] for t in kept_tuples]
        sample_fv = kept_tuples[0][3] if kept_tuples else scored[0][3]
        attrs = _top_feature_attributions(self._importances, sample_fv, self._PRUNE_FEATURE_NAMES, top_k=3)
        return kept, {
            "kept": len(kept),
            "dropped": total - len(kept),
            "top_attributions": [a.model_dump() for a in attrs],
        }


class BoostingPlannerHeuristic:
    """FR-06: LightGBM / HistGBDT P90 Latency & Utility Estimator + Adaptive Recovery Classifier."""

    def __init__(self):
        self.latency_fitted: bool = False
        self.recovery_fitted: bool = False
        self._lat_reg: Any = None
        self._util_reg: Any = None
        self._rec_clf: Any = None
        self._rec_classes: List[str] = []
        self._lat_importances: Optional[np.ndarray] = None
        self.engine_name: str = "lightgbm" if HAS_LIGHTGBM else ("hist_gbdt" if HAS_SKLEARN else "heuristic")

    def fit_cost_models(self, X: np.ndarray, y_latency_ms: Sequence[float], y_utility: Sequence[float]) -> "BoostingPlannerHeuristic":
        X_arr = np.asarray(X, dtype=np.float32)
        y_lat = np.asarray(y_latency_ms, dtype=np.float64)
        y_util = np.asarray(y_utility, dtype=np.float64)
        if X_arr.ndim != 2 or len(y_lat) < 12:
            return self
        if HAS_LIGHTGBM:
            self._lat_reg = lgb.LGBMRegressor(
                objective="quantile",
                alpha=0.90,
                n_estimators=40,
                max_depth=6,
                learning_rate=0.08,
                verbose=-1,
                n_jobs=1,
            )
            self._util_reg = lgb.LGBMRegressor(
                objective="huber",
                n_estimators=40,
                max_depth=6,
                learning_rate=0.08,
                verbose=-1,
                n_jobs=1,
            )
            self._lat_reg.fit(X_arr, y_lat)
            self._util_reg.fit(X_arr, y_util)
            self._lat_importances = getattr(self._lat_reg, "feature_importances_", None)
            self.latency_fitted = True
        elif HAS_SKLEARN:
            self._lat_reg = HistGradientBoostingRegressor(
                loss="quantile",
                quantile=0.90,
                max_iter=40,
                max_depth=6,
                random_state=42,
            )
            self._util_reg = HistGradientBoostingRegressor(
                loss="squared_error",
                max_iter=40,
                max_depth=6,
                random_state=42,
            )
            self._lat_reg.fit(X_arr, y_lat)
            self._util_reg.fit(X_arr, y_util)
            self._lat_importances = np.ones(X_arr.shape[1], dtype=np.float64) / X_arr.shape[1]
            self.latency_fitted = True
        return self

    def fit_recovery_classifier(self, X_err: np.ndarray, y_strategy: Sequence[str]) -> "BoostingPlannerHeuristic":
        X_arr = np.asarray(X_err, dtype=np.float32)
        y_list = [str(s) for s in y_strategy]
        if X_arr.ndim != 2 or len(y_list) < 12 or len(set(y_list)) < 2:
            return self
        if HAS_LIGHTGBM:
            clf = lgb.LGBMClassifier(n_estimators=40, max_depth=6, verbose=-1, n_jobs=1)
            clf.fit(X_arr, y_list)
            self._rec_clf = clf
            self._rec_classes = [str(c) for c in clf.classes_]
            self.recovery_fitted = True
        elif HAS_SKLEARN:
            clf = HistGradientBoostingClassifier(max_iter=40, max_depth=6, random_state=42)
            clf.fit(X_arr, y_list)
            self._rec_clf = clf
            self._rec_classes = [str(c) for c in clf.classes_]
            self.recovery_fitted = True
        return self

    def estimate_plan_step(self, x_vec: np.ndarray, fallback_latency_ms: float = 50.0) -> Tuple[float, float, List[FeatureAttribution]]:
        """Predict `(expected_p90_latency_ms, success_utility, attributions)` in < 1ms CPU."""
        x = np.asarray(x_vec, dtype=np.float32).reshape(1, -1)
        if self.latency_fitted and self._lat_reg is not None and self._util_reg is not None:
            lat = max(0.5, float(self._lat_reg.predict(x)[0]))
            util = float(np.clip(self._util_reg.predict(x)[0], 0.0, 1.0))
            attrs = _top_feature_attributions(self._lat_importances, x)
            return round(lat, 2), round(util, 4), attrs
        return round(float(fallback_latency_ms), 2), 0.80, []

    def rank_candidates(self, candidates: Sequence[Tuple[str, np.ndarray]]) -> List[Dict[str, Any]]:
        """Rank candidate plans/blueprints maximizing `utility / log(latency)`."""
        ranked: List[Dict[str, Any]] = []
        for cid, x_vec in candidates:
            lat, util, attrs = self.estimate_plan_step(x_vec)
            score = util / max(0.5, math.log1p(lat))
            ranked.append({
                "candidate_id": cid,
                "expected_latency_ms": lat,
                "expected_utility": util,
                "score": round(score, 4),
                "attributions": attrs,
            })
        ranked.sort(key=lambda r: r["score"], reverse=True)
        return ranked

    def classify_recovery(
        self,
        error_msg: str,
        x_vec: Optional[np.ndarray] = None,
        attempt: int = 1,
    ) -> Tuple[RecoveryStrategy, float, float]:
        """FR-06.2: Classify failure into WAIT_AND_RETRY | SWITCH_FALLBACK | ESCALATE_TO_LLM with backoff seconds."""
        if self.recovery_fitted and self._rec_clf is not None and x_vec is not None:
            x = np.asarray(x_vec, dtype=np.float32).reshape(1, -1)
            probs = self._rec_clf.predict_proba(x)[0]
            best_i = int(np.argmax(probs))
            strat = RecoveryStrategy(self._rec_classes[best_i])
            conf = round(float(probs[best_i]), 4)
            backoff = min(4.0, 0.25 * (2 ** max(0, attempt - 1))) if strat == RecoveryStrategy.WAIT_AND_RETRY else 0.0
            return strat, conf, backoff

        low = (error_msg or "").lower()
        transient = ("timeout", "timed out", "429", "rate limit", "temporarily", "connection reset", "busy", "503", "502")
        persistent = ("permission denied", "not found", "security violation", "blocked", "unsupported", "outside workspace", "401", "403")
        if any(k in low for k in transient) and attempt <= 3:
            return RecoveryStrategy.WAIT_AND_RETRY, 0.90, min(4.0, 0.25 * (2 ** max(0, attempt - 1)))
        if any(k in low for k in persistent):
            return RecoveryStrategy.SWITCH_FALLBACK, 0.88, 0.0
        return RecoveryStrategy.ESCALATE_TO_LLM, 0.75, 0.0


class SafetyGatekeeper:
    """FR-07: Deterministic Hard Rule Engine ALWAYS authoritative above RF Failure Risk Classifier."""

    HIGH_BLAST_TOOL_PATTERNS = (
        "delete", "drop", "truncate", "destroy", "wipe", "shutdown",
        "transfer_funds", "wire_payment", "refund_all", "format_disk",
    )
    DANGEROUS_PAYLOAD_REGEXES = (
        re.compile(r"\brm\s+-rf\s+/", re.IGNORECASE),
        re.compile(r"\bdrop\s+(?:table|database|schema)\b", re.IGNORECASE),
        re.compile(r"\btruncate\s+table\b", re.IGNORECASE),
        re.compile(r"\bdelete\s+from\s+\w+\s*;?\s*$", re.IGNORECASE),
        re.compile(r"\b(?:mkfs|dd\s+if=/dev/zero|chmod\s+-R\s+777\s+/)\b", re.IGNORECASE),
        re.compile(r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:"),
        re.compile(r"__import__\s*\(\s*['\"](?:os|subprocess|shutil)['\"]\s*\)"),
    )

    def __init__(self, risk_ceiling: float = 0.70):
        self.risk_ceiling = float(risk_ceiling)
        self.fitted: bool = False
        self._rf: Any = None
        self._importances: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y_failed: Sequence[int]) -> "SafetyGatekeeper":
        if not HAS_SKLEARN:
            return self
        arr = np.asarray(X, dtype=np.float32)
        y = np.asarray(y_failed, dtype=int)
        if arr.ndim != 2 or len(y) < 10 or len(set(y.tolist())) < 2:
            return self
        rf = RandomForestClassifier(n_estimators=30, max_depth=8, random_state=42, n_jobs=1)
        rf.fit(arr, y)
        self._rf = rf
        self._importances = getattr(rf, "feature_importances_", None)
        self.fitted = True
        return self

    def check_hard_invariants(self, tool_id: str, payload: Optional[Dict[str, Any]] = None) -> Tuple[bool, Optional[str]]:
        """FR-07.3: Hard deterministic invariants that CANNOT be overridden by ML scores."""
        t_low = (tool_id or "").lower()
        payload_str = str(payload or "")
        for rx in self.DANGEROUS_PAYLOAD_REGEXES:
            if rx.search(payload_str):
                return False, f"Hard Rule Block: destructive/unsafe payload pattern detected for `{tool_id}`."
        if any(pat in t_low for pat in self.HIGH_BLAST_TOOL_PATTERNS):
            approved = bool((payload or {}).get("__human_approved__"))
            if not approved:
                return False, f"Hard Rule Gate: High-Blast-Radius tool `{tool_id}` requires explicit deterministic approval."
        return True, None

    def evaluate(
        self,
        tool_id: str,
        x_vec: np.ndarray,
        payload: Optional[Dict[str, Any]] = None,
        *,
        block_on_ml_risk: bool = True,
    ) -> Tuple[bool, float, Optional[str], List[FeatureAttribution]]:
        """Evaluate Hard Rules first, then RF risk probability (< 0.5ms CPU)."""
        hard_ok, hard_reason = self.check_hard_invariants(tool_id, payload)
        if not hard_ok:
            return False, 1.0, hard_reason, []

        fail_prob = 0.05
        attrs: List[FeatureAttribution] = []
        if self.fitted and self._rf is not None:
            x = np.asarray(x_vec, dtype=np.float32).reshape(1, -1)
            probs = self._rf.predict_proba(x)[0]
            classes = list(self._rf.classes_)
            fail_prob = float(probs[classes.index(1)]) if 1 in classes else 0.0
            attrs = _top_feature_attributions(self._importances, x)

        if block_on_ml_risk and fail_prob > self.risk_ceiling:
            return (
                False,
                round(fail_prob, 4),
                f"RF Safety Gate blocked `{tool_id}`: predicted_fail_prob={fail_prob:.2f} > ceiling={self.risk_ceiling:.2f}",
                attrs,
            )
        return True, round(fail_prob, 4), None, attrs
