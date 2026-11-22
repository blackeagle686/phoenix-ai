"""PHX v2 Cold Path: TraceStore, HDBSCAN Skill Compiler, Episodic Memory Compressor,
PSI Drift Monitor, Shadow Validation Gate & Atomic Hot-Swap (FR-08 to FR-12).
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np

from phoenix.framework.agent.kernel.features import FEATURE_DIM
from phoenix.framework.agent.kernel.models import (
    BoostingPlannerHeuristic,
    CalibratedRFRouter,
    RFContextPruner,
    SafetyGatekeeper,
    compute_ece,
)
from phoenix.framework.agent.kernel.regime_ood import RegimeOODGuard
from phoenix.framework.agent.kernel.schema import (
    CompiledSkill,
    ExecutionTrace,
    OutcomeStatus,
)

try:
    import hdbscan as _ext_hdbscan
except ImportError:
    _ext_hdbscan = None

try:
    from sklearn.cluster import HDBSCAN as _SklearnHDBSCAN
except ImportError:
    _SklearnHDBSCAN = None


class TraceStore:
    """FR-08: Append-only ExecutionTrace event store (in-memory ring buffer + optional JSONL persistence)."""

    def __init__(self, storage_path: Optional[str] = None, max_memory_traces: int = 20000):
        self.storage_path = storage_path
        self._lock = threading.Lock()
        self._traces: Deque[ExecutionTrace] = deque(maxlen=max(100, int(max_memory_traces)))
        if self.storage_path and os.path.isfile(self.storage_path):
            self._load_existing()

    def _load_existing(self) -> None:
        if not self.storage_path:
            return
        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                for line in f:
                    raw = line.strip()
                    if not raw:
                        continue
                    try:
                        tr = ExecutionTrace.model_validate_json(raw) if hasattr(ExecutionTrace, "model_validate_json") else ExecutionTrace(**json.loads(raw))
                        self._traces.append(tr)
                    except Exception:
                        continue
        except OSError:
            pass

    def append(self, trace: ExecutionTrace) -> None:
        with self._lock:
            self._traces.append(trace)
            if self.storage_path:
                try:
                    parent = os.path.dirname(self.storage_path)
                    if parent:
                        os.makedirs(parent, exist_ok=True)
                    payload = trace.model_dump_json() if hasattr(trace, "model_dump_json") else trace.json()
                    with open(self.storage_path, "a", encoding="utf-8") as f:
                        f.write(payload + "\n")
                except OSError:
                    pass

    def all_traces(self) -> List[ExecutionTrace]:
        with self._lock:
            return list(self._traces)

    def verified_traces(self) -> List[ExecutionTrace]:
        """FR-12.1 Verified-Only Distillation: ONLY verifier_passed == True AND status == SUCCESS."""
        with self._lock:
            return [
                t for t in self._traces
                if t.outcome.verifier_passed and t.outcome.status == OutcomeStatus.SUCCESS
            ]

    def count_verified(self) -> int:
        with self._lock:
            return sum(
                1 for t in self._traces
                if t.outcome.verifier_passed and t.outcome.status == OutcomeStatus.SUCCESS
            )


class HDBSCANSkillMiner:
    """FR-09: Autonomous Skill Discovery & Macro Compilation via HDBSCAN + Replay Validation."""

    def __init__(self, min_cluster_size: int = 20, min_success_rate: float = 0.95):
        self.min_cluster_size = max(2, int(min_cluster_size))
        self.min_success_rate = float(min_success_rate)

    def _cluster_vectors(self, X: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        if n < self.min_cluster_size:
            return np.full(n, -1, dtype=int)
        if _ext_hdbscan is not None:
            try:
                clusterer = _ext_hdbscan.HDBSCAN(min_cluster_size=self.min_cluster_size, min_samples=max(2, self.min_cluster_size // 2))
                return np.asarray(clusterer.fit_predict(X), dtype=int)
            except Exception:
                pass
        if _SklearnHDBSCAN is not None:
            try:
                clusterer = _SklearnHDBSCAN(
                    min_cluster_size=self.min_cluster_size,
                    min_samples=max(2, self.min_cluster_size // 2),
                    copy=True,
                )
                return np.asarray(clusterer.fit_predict(X), dtype=int)
            except Exception:
                pass
        # Pure-NumPy fallback density grouping by tight Euclidean radius
        labels = np.full(n, -1, dtype=int)
        cid = 0
        for i in range(n):
            if labels[i] != -1:
                continue
            dists = np.linalg.norm(X - X[i], axis=1)
            members = np.where(dists <= 0.75)[0]
            if len(members) >= self.min_cluster_size:
                labels[members] = cid
                cid += 1
        return labels

    def mine(self, all_traces: Sequence[ExecutionTrace]) -> List[CompiledSkill]:
        """Cluster verified episodes via HDBSCAN and compile deterministic parameterized macros."""
        # Group traces by episode to evaluate full episode action sequences
        episodes: Dict[str, List[ExecutionTrace]] = {}
        for tr in all_traces:
            if tr.state_vector:
                episodes.setdefault(tr.episode_id, []).append(tr)

        episode_records: List[Dict[str, Any]] = []
        for ep_id, steps in episodes.items():
            steps_sorted = sorted(steps, key=lambda t: t.timestamp_ms)
            seq: List[str] = []
            for st in steps_sorted:
                if st.action_sequence:
                    seq.extend(st.action_sequence)
                elif st.decision.action_id:
                    seq.append(st.decision.action_id)
            if not seq:
                continue
            first_vec = np.asarray(steps_sorted[0].state_vector, dtype=np.float64)
            last_vec = np.asarray(steps_sorted[-1].state_vector, dtype=np.float64)
            delta = last_vec - first_vec
            seq_sig = hashlib.blake2b("->".join(seq).encode("utf-8"), digest_size=4).digest()
            seq_feat = np.array([b / 255.0 for b in seq_sig], dtype=np.float64)
            behavior_vec = np.concatenate([first_vec, delta, seq_feat])
            all_ok = all(s.outcome.verifier_passed and s.outcome.status == OutcomeStatus.SUCCESS for s in steps_sorted)
            episode_records.append({
                "episode_id": ep_id,
                "seq": tuple(seq),
                "first_vec": first_vec,
                "behavior_vec": behavior_vec,
                "success": all_ok,
                "regime_id": steps_sorted[0].regime_id,
                "arg_templates": [dict(s.action_args) for s in steps_sorted if s.action_args],
                "prompt": steps_sorted[0].prompt_excerpt,
            })

        verified_eps = [r for r in episode_records if r["success"]]
        if len(verified_eps) < self.min_cluster_size:
            return []

        # Ensure uniform vector dimensionality
        target_len = len(verified_eps[0]["behavior_vec"])
        verified_eps = [r for r in verified_eps if len(r["behavior_vec"]) == target_len]
        if len(verified_eps) < self.min_cluster_size:
            return []

        X_beh = np.vstack([r["behavior_vec"] for r in verified_eps])
        labels = self._cluster_vectors(X_beh)

        # Fallback: if all points had identical behavior vectors (0 distance), HDBSCAN may assign -1 or 0
        # Also evaluate exact action-sequence groups with tight state-vector variance
        cluster_groups: Dict[int, List[Dict[str, Any]]] = {}
        for rec, lbl in zip(verified_eps, labels):
            if int(lbl) >= 0:
                cluster_groups.setdefault(int(lbl), []).append(rec)

        if not cluster_groups:
            by_seq: Dict[Tuple[str, ...], List[Dict[str, Any]]] = {}
            for rec in verified_eps:
                by_seq.setdefault(rec["seq"], []).append(rec)
            for idx, (seq_key, recs) in enumerate(by_seq.items()):
                if len(recs) >= self.min_cluster_size:
                    cluster_groups[idx] = recs

        compiled: List[CompiledSkill] = []
        for cid, members in cluster_groups.items():
            if len(members) < self.min_cluster_size:
                continue
            seq_counts = Counter(m["seq"] for m in members)
            dominant_seq, dom_count = seq_counts.most_common(1)[0]
            seq_consistency = dom_count / float(len(members))
            if seq_consistency < 0.85:
                continue

            # Replay Validation over ALL historical episodes (including failures!) inside cluster radius
            state_vecs = np.vstack([m["first_vec"] for m in members])
            centroid = state_vecs.mean(axis=0)
            member_dists = np.linalg.norm(state_vecs - centroid, axis=1)
            radius = max(float(np.percentile(member_dists, 95)) * 1.15, 0.35)

            matched_total = 0
            matched_success = 0
            for ep in episode_records:
                if len(ep["first_vec"]) != len(centroid):
                    continue
                if float(np.linalg.norm(ep["first_vec"] - centroid)) <= radius:
                    matched_total += 1
                    if ep["success"] and ep["seq"] == dominant_seq:
                        matched_success += 1

            replay_sr = (matched_success / float(matched_total)) if matched_total > 0 else 0.0
            if matched_success < self.min_cluster_size or replay_sr < self.min_success_rate:
                continue

            regimes = sorted({int(m["regime_id"]) for m in members})
            sample_args = members[0]["arg_templates"]
            skill_hash = hashlib.blake2b(f"{dominant_seq}:{cid}".encode("utf-8"), digest_size=4).hexdigest()
            compiled.append(
                CompiledSkill(
                    skill_id=f"skill_{skill_hash}",
                    name=f"macro_{'_'.join(dominant_seq[:3])}",
                    action_sequence=list(dominant_seq),
                    precondition_centroid=centroid.astype(float).tolist(),
                    precondition_radius=round(radius, 4),
                    regime_ids=regimes,
                    default_arg_templates=sample_args,
                    postcondition_checks=["verifier_passed==True", "status==SUCCESS"],
                    verified_support=matched_success,
                    initial_success_rate=round(replay_sr, 4),
                )
            )
        return compiled


class EpisodicMemoryCompressor:
    """FR-10: Compresses routine in-radius episodes into centroid statistics while keeping full outlier/failure traces."""

    def __init__(self, outlier_radius: float = 1.25, max_centroids: int = 16):
        self.outlier_radius = float(outlier_radius)
        self.max_centroids = max(2, int(max_centroids))
        self._centroids: List[Dict[str, Any]] = []  # {mean: np.ndarray, m2: np.ndarray, count: int}
        self.preserved_outliers: List[ExecutionTrace] = []
        self.compressed_count: int = 0

    def ingest(self, trace: ExecutionTrace) -> str:
        """Ingest trace: preserve full trace for failures/outliers, aggregate routine events into centroids."""
        is_failure = (not trace.outcome.verifier_passed) or (trace.outcome.status != OutcomeStatus.SUCCESS)
        vec = np.asarray(trace.state_vector, dtype=np.float64)
        if vec.size == 0 or is_failure:
            self.preserved_outliers.append(trace)
            return "preserved_failure" if is_failure else "preserved_empty"

        if not self._centroids:
            self._centroids.append({"mean": vec.copy(), "m2": np.zeros_like(vec), "count": 1})
            self.compressed_count += 1
            return "created_centroid"

        best_idx = -1
        best_dist = float("inf")
        for idx, c in enumerate(self._centroids):
            if c["mean"].size != vec.size:
                continue
            d = float(np.linalg.norm(vec - c["mean"]))
            if d < best_dist:
                best_dist = d
                best_idx = idx

        if best_idx >= 0 and best_dist <= self.outlier_radius:
            c = self._centroids[best_idx]
            c["count"] += 1
            delta = vec - c["mean"]
            c["mean"] += delta / c["count"]
            c["m2"] += delta * (vec - c["mean"])
            self.compressed_count += 1
            return "compressed"

        # Outlier: preserve full trace AND spawn a new centroid if capacity allows
        self.preserved_outliers.append(trace)
        if len(self._centroids) < self.max_centroids:
            self._centroids.append({"mean": vec.copy(), "m2": np.zeros_like(vec), "count": 1})
        return "preserved_outlier"

    def summary(self) -> Dict[str, Any]:
        return {
            "centroids": len(self._centroids),
            "compressed_routine_events": self.compressed_count,
            "preserved_outliers_and_failures": len(self.preserved_outliers),
            "compression_ratio": round(
                self.compressed_count / float(max(1, self.compressed_count + len(self.preserved_outliers))),
                4,
            ),
        }


class DriftAndCalibrationMonitor:
    """NFR & FR-06.3: Population Stability Index (PSI) drift detection & ECE calibration auditing."""

    def __init__(self, psi_threshold: float = 0.20, n_bins: int = 10):
        self.psi_threshold = float(psi_threshold)
        self.n_bins = max(4, int(n_bins))
        self.baseline_X: Optional[np.ndarray] = None
        self.last_psi: float = 0.0
        self.last_ece: float = 0.0
        self.drift_detected: bool = False

    def set_baseline(self, X: np.ndarray) -> None:
        arr = np.asarray(X, dtype=np.float64)
        if arr.ndim == 2 and arr.shape[0] >= 10:
            self.baseline_X = arr.copy()

    def compute_psi(self, recent_X: np.ndarray) -> Tuple[float, bool]:
        """Compute mean & max Population Stability Index across X_t features; flag drift if PSI > 0.20."""
        if self.baseline_X is None:
            return 0.0, False
        cur = np.asarray(recent_X, dtype=np.float64)
        if cur.ndim != 2 or cur.shape[0] < 10 or cur.shape[1] != self.baseline_X.shape[1]:
            return 0.0, False

        eps = 1e-6
        psi_vals: List[float] = []
        for col in range(cur.shape[1]):
            b_col = self.baseline_X[:, col]
            c_col = cur[:, col]
            if float(np.std(b_col)) < 1e-6 and float(np.std(c_col)) < 1e-6:
                psi_vals.append(0.0)
                continue
            lo = min(float(np.min(b_col)), float(np.min(c_col)))
            hi = max(float(np.max(b_col)), float(np.max(c_col)))
            if hi - lo < 1e-6:
                psi_vals.append(0.0)
                continue
            edges = np.linspace(lo, hi, self.n_bins + 1)
            b_hist, _ = np.histogram(b_col, bins=edges)
            c_hist, _ = np.histogram(c_col, bins=edges)
            b_pct = np.clip(b_hist / float(max(1, len(b_col))), eps, 1.0)
            c_pct = np.clip(c_hist / float(max(1, len(c_col))), eps, 1.0)
            col_psi = float(np.sum((c_pct - b_pct) * np.log(c_pct / b_pct)))
            psi_vals.append(max(0.0, col_psi))

        # Use top-quartile PSI to avoid dilution across constant zero-variance columns
        sorted_psi = sorted(psi_vals, reverse=True)
        top_k = max(1, len(sorted_psi) // 4)
        agg_psi = round(float(np.mean(sorted_psi[:top_k])), 4)
        self.last_psi = agg_psi
        self.drift_detected = agg_psi > self.psi_threshold
        return agg_psi, self.drift_detected


@dataclass
class ModelBundle:
    """Immutable snapshot of all Hot Path models & compiled skills swapped atomically (FR-12.3)."""

    version: int = 0
    created_at_ms: int = field(default_factory=lambda: int(time.time() * 1000))
    verified_trace_count: int = 0
    regime_ood: RegimeOODGuard = field(default_factory=RegimeOODGuard)
    rf_router: CalibratedRFRouter = field(default_factory=CalibratedRFRouter)
    context_pruner: RFContextPruner = field(default_factory=RFContextPruner)
    planner_heuristic: BoostingPlannerHeuristic = field(default_factory=BoostingPlannerHeuristic)
    safety_gatekeeper: SafetyGatekeeper = field(default_factory=SafetyGatekeeper)
    compiled_skills: Dict[str, CompiledSkill] = field(default_factory=dict)
    shadow_agreement: float = 0.0
    calibration_ece: float = 0.0


class ColdPathTrainer:
    """FR-12: Verified-only training, Shadow Validation Gate, and Atomic Hot-Swap."""

    def __init__(
        self,
        min_shadow_agreement: float = 0.90,
        max_calibration_ece: float = 0.15,
        skill_min_cluster_size: int = 20,
    ):
        self.min_shadow_agreement = float(min_shadow_agreement)
        self.max_calibration_ece = float(max_calibration_ece)
        self.skill_miner = HDBSCANSkillMiner(min_cluster_size=skill_min_cluster_size)

    def train_and_validate(
        self,
        trace_store: TraceStore,
        current_bundle: ModelBundle,
        drift_monitor: Optional[DriftAndCalibrationMonitor] = None,
        *,
        force_promote: bool = False,
    ) -> Tuple[Optional[ModelBundle], Dict[str, Any]]:
        """Train candidate bundle strictly on verified traces and validate before atomic promotion."""
        all_traces = trace_store.all_traces()
        verified = [
            t for t in all_traces
            if t.outcome.verifier_passed and t.outcome.status == OutcomeStatus.SUCCESS and len(t.state_vector) == FEATURE_DIM
        ]
        if len(verified) < 6:
            return None, {
                "promoted": False,
                "reason": f"Need at least 6 verified traces to train ML bundle (have {len(verified)}).",
                "verified_count": len(verified),
            }

        X_ver = np.vstack([np.asarray(t.state_vector, dtype=np.float32) for t in verified])
        y_act = [t.decision.action_id for t in verified]
        y_lat = [t.outcome.actual_latency_ms for t in verified]
        y_util = [max(0.0, min(1.0, t.outcome.reward_signal)) for t in verified]

        # Stratified deterministic train/val split so bursty action logs stay balanced across holdout folds
        if len(verified) >= 20:
            by_cls: Dict[str, List[int]] = {}
            for idx, lbl in enumerate(y_act):
                by_cls.setdefault(lbl, []).append(idx)
            train_idx: List[int] = []
            val_idx: List[int] = []
            for cls_indices in by_cls.values():
                if len(cls_indices) >= 5:
                    cut = max(1, int(len(cls_indices) * 0.8))
                    train_idx.extend(cls_indices[:cut])
                    val_idx.extend(cls_indices[cut:])
                else:
                    train_idx.extend(cls_indices)
            if not val_idx:
                val_idx = train_idx
            X_train, X_val = X_ver[train_idx], X_ver[val_idx]
            y_train = [y_act[i] for i in train_idx]
            y_val = [y_act[i] for i in val_idx]
        else:
            X_train, X_val = X_ver, X_ver
            y_train, y_val = y_act, y_act

        regime_ood = RegimeOODGuard(n_clusters=min(6, max(2, len(X_train) // 10)))
        regime_ood.fit(X_train)

        rf_router = CalibratedRFRouter()
        rf_router.fit(X_train, y_train)
        rf_router.try_export_onnx(FEATURE_DIM)

        # Evaluate Shadow Agreement & ECE on holdout
        val_probs: List[float] = []
        val_preds: List[str] = []
        hi_conf_correct = 0
        hi_conf_total = 0
        all_correct = 0
        for xv, yt in zip(X_val, y_val):
            pred, prob, _, _ = rf_router.predict(xv)
            val_probs.append(prob)
            val_preds.append(pred)
            if pred == yt:
                all_correct += 1
            if prob >= 0.85:
                hi_conf_total += 1
                if pred == yt:
                    hi_conf_correct += 1

        shadow_agreement = (
            (hi_conf_correct / float(hi_conf_total))
            if hi_conf_total > 0
            else (all_correct / float(max(1, len(y_val))))
        )
        ece = compute_ece(np.asarray(val_probs), np.asarray(y_val), np.asarray(val_preds))

        planner_heuristic = BoostingPlannerHeuristic()
        planner_heuristic.fit_cost_models(X_ver, y_lat, y_util)

        # Train Safety Gatekeeper across ALL traces (both verified successes AND recorded failures!)
        valid_all = [t for t in all_traces if len(t.state_vector) == FEATURE_DIM]
        safety_gate = SafetyGatekeeper(risk_ceiling=current_bundle.safety_gatekeeper.risk_ceiling)
        if len(valid_all) >= 10:
            X_all = np.vstack([np.asarray(t.state_vector, dtype=np.float32) for t in valid_all])
            y_fail = [0 if (t.outcome.verifier_passed and t.outcome.status == OutcomeStatus.SUCCESS) else 1 for t in valid_all]
            safety_gate.fit(X_all, y_fail)

        # Mine HDBSCAN Compiled Skills
        mined_skills = self.skill_miner.mine(all_traces)
        skills_map = dict(current_bundle.compiled_skills)
        for sk in mined_skills:
            if sk.skill_id not in skills_map or not skills_map[sk.skill_id].active:
                skills_map[sk.skill_id] = sk

        # Shadow Validation Gate check before promotion
        passed_shadow_gate = force_promote or (
            shadow_agreement >= self.min_shadow_agreement and ece <= self.max_calibration_ece
        )
        if not passed_shadow_gate:
            return None, {
                "promoted": False,
                "reason": (
                    f"Shadow validation gate rejected candidate bundle "
                    f"(agreement={shadow_agreement:.3f} < {self.min_shadow_agreement}, ece={ece:.4f} > {self.max_calibration_ece})."
                ),
                "shadow_agreement": round(shadow_agreement, 4),
                "ece": ece,
                "verified_count": len(verified),
            }

        if drift_monitor is not None:
            drift_monitor.set_baseline(X_ver)
            drift_monitor.last_ece = ece

        new_bundle = ModelBundle(
            version=current_bundle.version + 1,
            verified_trace_count=len(verified),
            regime_ood=regime_ood,
            rf_router=rf_router,
            context_pruner=current_bundle.context_pruner,
            planner_heuristic=planner_heuristic,
            safety_gatekeeper=safety_gate,
            compiled_skills=skills_map,
            shadow_agreement=round(shadow_agreement, 4),
            calibration_ece=ece,
        )
        return new_bundle, {
            "promoted": True,
            "version": new_bundle.version,
            "verified_count": len(verified),
            "shadow_agreement": new_bundle.shadow_agreement,
            "ece": ece,
            "calibration_method": rf_router.calibration_method,
            "boosting_engine": planner_heuristic.engine_name,
            "compiled_skills_count": len(skills_map),
        }
