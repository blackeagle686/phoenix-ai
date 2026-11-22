"""PHX v2 Kernel: Sub-5ms CPU Hot Path Coordinator & 3-Stage Cold-Start Graduation.

Unifies FeatureExtractor, Regime & OOD Guard, Calibrated RF Action Router,
RF Context Pruner, LightGBM/HistGBDT Planner Heuristic, Safety Gatekeeper,
Compiled Skills, and Cold Path Async Distillation with Atomic Hot-Swapping.
"""

from __future__ import annotations

import random
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np

from phoenix.framework.agent.kernel.cold_path import (
    ColdPathTrainer,
    DriftAndCalibrationMonitor,
    EpisodicMemoryCompressor,
    ModelBundle,
    TraceStore,
)
from phoenix.framework.agent.kernel.features import (
    FEATURE_SCHEMA_VERSION,
    FeatureExtractor,
    RuntimeTelemetryTracker,
    WorldStateSnapshot,
)
from phoenix.framework.agent.kernel.schema import (
    CompiledSkill,
    DecisionRecord,
    DecisionSource,
    ExecutionTrace,
    FeatureAttribution,
    GraduationStage,
    OutcomeRecord,
    OutcomeStatus,
    RecoveryStrategy,
)


@dataclass
class KernelDecision:
    """Result of a single < 5ms Hot Path routing & gating cycle."""

    source: DecisionSource
    action_id: Optional[str]
    should_wake_llm: bool
    confidence: float
    regime_id: int
    ood_distance: float
    is_ood: bool
    state_vector: List[float]
    predicted_latency_ms: float
    predicted_fail_prob: float
    safety_allowed: bool
    safety_reason: Optional[str] = None
    matched_skill: Optional[CompiledSkill] = None
    top_attributions: List[FeatureAttribution] = field(default_factory=list)
    shadow_rf_action: Optional[str] = None
    shadow_rf_prob: Optional[float] = None
    exploration_audited: bool = False
    hot_path_latency_ms: float = 0.0
    graduation_stage: GraduationStage = GraduationStage.SHADOW


class PHXKernel:
    """
    PHX v2 Kernel-First Hybrid ML Autonomous Agent Engine.

    Hot Path executes in < 5ms on CPU:
      1. Canonical Feature Extraction X_t (shared train/serve)
      2. Compiled Skill Precondition Lookup (with Circuit Breaker + Exploration Budget)
      3. MiniBatch K-Means Regime Classification + OOD Guard (d > theta_ood -> LLM)
      4. 3-Stage Graduation Gating + Calibrated RF Action Router (P >= tau -> Deterministic)
      5. RF Context Pruning when waking Stateless LLM Battery
      6. LightGBM/HistGBDT Cost/Utility Planner + Safety Gatekeeper (Hard Rules > RF)
      7. ExecutionTrace emission to Cold Path + Atomic ModelBundle Hot-Swap
    """

    def __init__(
        self,
        *,
        trace_storage_path: Optional[str] = None,
        autonomous_tau: float = 0.85,
        conservative_tau: float = 0.92,
        stage2_min_traces: int = 1000,
        stage3_min_traces: int = 10000,
        exploration_rate: float = 0.03,
        risk_ceiling: float = 0.70,
        skill_min_cluster_size: int = 20,
        read_only_tools: Optional[Sequence[str]] = None,
        random_seed: Optional[int] = None,
    ):
        self.autonomous_tau = float(autonomous_tau)
        self.conservative_tau = float(conservative_tau)
        self.stage2_min_traces = max(1, int(stage2_min_traces))
        self.stage3_min_traces = max(self.stage2_min_traces + 1, int(stage3_min_traces))
        self.exploration_rate = max(0.0, min(0.25, float(exploration_rate)))
        self.read_only_tools = set(read_only_tools or (
            "file_read", "file_search", "python_analyzer", "compile_code",
            "web_search", "wikipedia_search", "arxiv_search", "web_scraper",
            "sql_query", "python_diagnose",
        ))
        self._rng = random.Random(random_seed)

        self.feature_extractor = FeatureExtractor()
        self.telemetry = RuntimeTelemetryTracker()
        self.trace_store = TraceStore(storage_path=trace_storage_path)
        self.memory_compressor = EpisodicMemoryCompressor()
        self.drift_monitor = DriftAndCalibrationMonitor(psi_threshold=0.20)
        self.trainer = ColdPathTrainer(skill_min_cluster_size=skill_min_cluster_size)

        # Atomic active bundle pointer (swapped cleanly without locks on read path)
        initial_bundle = ModelBundle()
        initial_bundle.safety_gatekeeper.risk_ceiling = float(risk_ceiling)
        self._active_bundle: ModelBundle = initial_bundle
        self._swap_lock = threading.Lock()

        # Operational metrics for NFR observability
        self._hot_path_latencies_ms: Deque[float] = deque(maxlen=500)
        self._recent_vectors: Deque[np.ndarray] = deque(maxlen=250)
        self._total_decisions: int = 0
        self._llm_wakeups: int = 0
        self._skill_hits: int = 0
        self._rf_hits: int = 0
        self._ood_escalations: int = 0
        self._safety_blocks: int = 0
        self._tau_drift_bump: float = 0.0

    @property
    def active_bundle(self) -> ModelBundle:
        return self._active_bundle

    def get_graduation_stage(self) -> GraduationStage:
        """Section 6: Metric-gated 3-stage Cold-Start Graduation Protocol."""
        verified_cnt = max(self.trace_store.count_verified(), self._active_bundle.verified_trace_count)
        if verified_cnt >= self.stage3_min_traces:
            return GraduationStage.AUTONOMOUS
        if verified_cnt >= self.stage2_min_traces:
            return GraduationStage.CONSERVATIVE
        return GraduationStage.SHADOW

    def _effective_tau(self, stage: GraduationStage) -> float:
        base = self.conservative_tau if stage == GraduationStage.CONSERVATIVE else self.autonomous_tau
        return min(0.98, base + self._tau_drift_bump)

    def _match_compiled_skill(
        self,
        bundle: ModelBundle,
        x_vec: np.ndarray,
        regime_id: int,
        prompt_text: str,
    ) -> Optional[CompiledSkill]:
        low = (prompt_text or "").lower()
        best_skill: Optional[CompiledSkill] = None
        best_dist = float("inf")
        for sk in bundle.compiled_skills.values():
            if not sk.active:
                continue
            if sk.regime_ids and regime_id not in sk.regime_ids:
                continue
            if sk.required_keywords and not all(kw.lower() in low for kw in sk.required_keywords):
                continue
            cent = np.asarray(sk.precondition_centroid, dtype=np.float64)
            if cent.size != x_vec.size:
                continue
            dist = float(np.linalg.norm(x_vec.astype(np.float64) - cent))
            if dist <= sk.precondition_radius and dist < best_dist:
                best_dist = dist
                best_skill = sk
        return best_skill

    def decide(
        self,
        prompt: str,
        world_state: Optional[WorldStateSnapshot] = None,
        candidate_payload: Optional[Dict[str, Any]] = None,
        *,
        raw_embedding: Optional[List[float]] = None,
        allow_exploration: bool = True,
    ) -> KernelDecision:
        """Execute full Hot Path decision pipeline in < 5ms on CPU."""
        t0 = time.perf_counter()
        bundle = self._active_bundle  # Lock-free atomic snapshot read
        ws = world_state or WorldStateSnapshot()
        stage = self.get_graduation_stage()

        # (1) Canonical feature vector extraction
        x_vec = self.feature_extractor.extract(
            prompt,
            world_state=ws,
            telemetry=self.telemetry,
            candidate=candidate_payload,
            raw_embedding=raw_embedding,
        )
        self._recent_vectors.append(x_vec)

        # (2) Regime classification & OOD Guard (FR-02)
        regime_id, ood_dist, is_ood = bundle.regime_ood.evaluate(x_vec)
        ws.regime_cluster_id = regime_id
        # Update regime slot in canonical state vector
        x_vec = x_vec.copy()
        x_vec[35] = min(1.0, max(0.0, float(regime_id)) / 10.0)

        self._total_decisions += 1
        exploration_hit = allow_exploration and self.exploration_rate > 0 and (self._rng.random() < self.exploration_rate)

        # (2b) Compiled Skill check (Stage 3 Autonomous or explicit active skills)
        if stage == GraduationStage.AUTONOMOUS and not is_ood and not exploration_hit:
            matched_sk = self._match_compiled_skill(bundle, x_vec, regime_id, prompt)
            if matched_sk is not None:
                first_act = matched_sk.action_sequence[0] if matched_sk.action_sequence else "compiled_skill"
                pred_lat, _, _ = bundle.planner_heuristic.estimate_plan_step(x_vec, fallback_latency_ms=10.0)
                safe_ok, fail_p, safe_reason, safe_attrs = bundle.safety_gatekeeper.evaluate(
                    first_act, x_vec, candidate_payload, block_on_ml_risk=True
                )
                elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 4)
                self._hot_path_latencies_ms.append(elapsed_ms)
                if not safe_ok:
                    self._safety_blocks += 1
                    return KernelDecision(
                        source=DecisionSource.HARD_RULE,
                        action_id=first_act,
                        should_wake_llm=True,
                        confidence=matched_sk.rolling_success_rate,
                        regime_id=regime_id,
                        ood_distance=ood_dist,
                        is_ood=False,
                        state_vector=x_vec.astype(float).tolist(),
                        predicted_latency_ms=pred_lat,
                        predicted_fail_prob=fail_p,
                        safety_allowed=False,
                        safety_reason=safe_reason,
                        matched_skill=matched_sk,
                        top_attributions=safe_attrs,
                        hot_path_latency_ms=elapsed_ms,
                        graduation_stage=stage,
                    )
                self._skill_hits += 1
                return KernelDecision(
                    source=DecisionSource.COMPILED_SKILL,
                    action_id=first_act,
                    should_wake_llm=False,
                    confidence=matched_sk.rolling_success_rate,
                    regime_id=regime_id,
                    ood_distance=ood_dist,
                    is_ood=False,
                    state_vector=x_vec.astype(float).tolist(),
                    predicted_latency_ms=pred_lat,
                    predicted_fail_prob=fail_p,
                    safety_allowed=True,
                    matched_skill=matched_sk,
                    top_attributions=safe_attrs,
                    hot_path_latency_ms=elapsed_ms,
                    graduation_stage=stage,
                )

        # (3) OOD Guard -> Bypass RF immediately to prevent tree overconfidence
        if is_ood:
            self._ood_escalations += 1
            self._llm_wakeups += 1
            pred_lat, _, _ = bundle.planner_heuristic.estimate_plan_step(x_vec, fallback_latency_ms=300.0)
            elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 4)
            self._hot_path_latencies_ms.append(elapsed_ms)
            return KernelDecision(
                source=DecisionSource.LLM_BATTERY,
                action_id=None,
                should_wake_llm=True,
                confidence=0.0,
                regime_id=regime_id,
                ood_distance=ood_dist,
                is_ood=True,
                state_vector=x_vec.astype(float).tolist(),
                predicted_latency_ms=pred_lat,
                predicted_fail_prob=0.0,
                safety_allowed=True,
                hot_path_latency_ms=elapsed_ms,
                graduation_stage=stage,
            )

        # (4) Calibrated RF Action Router
        rf_act, rf_prob, _, rf_attrs = bundle.rf_router.predict(x_vec)
        tau = self._effective_tau(stage)

        # Determine whether RF has control authority in current graduation stage
        rf_has_authority = False
        if bundle.rf_router.fitted and rf_act and not exploration_hit:
            if stage == GraduationStage.AUTONOMOUS and rf_prob >= tau:
                rf_has_authority = True
            elif stage == GraduationStage.CONSERVATIVE and rf_prob >= tau and rf_act in self.read_only_tools:
                rf_has_authority = True

        if rf_has_authority:
            pred_lat, _, _ = bundle.planner_heuristic.estimate_plan_step(x_vec)
            block_ml = stage == GraduationStage.AUTONOMOUS
            safe_ok, fail_p, safe_reason, safe_attrs = bundle.safety_gatekeeper.evaluate(
                rf_act, x_vec, candidate_payload, block_on_ml_risk=block_ml
            )
            elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 4)
            self._hot_path_latencies_ms.append(elapsed_ms)
            if not safe_ok:
                self._safety_blocks += 1
                self._llm_wakeups += 1
                return KernelDecision(
                    source=DecisionSource.HARD_RULE,
                    action_id=rf_act,
                    should_wake_llm=True,
                    confidence=rf_prob,
                    regime_id=regime_id,
                    ood_distance=ood_dist,
                    is_ood=False,
                    state_vector=x_vec.astype(float).tolist(),
                    predicted_latency_ms=pred_lat,
                    predicted_fail_prob=fail_p,
                    safety_allowed=False,
                    safety_reason=safe_reason,
                    top_attributions=safe_attrs or rf_attrs,
                    shadow_rf_action=rf_act,
                    shadow_rf_prob=rf_prob,
                    hot_path_latency_ms=elapsed_ms,
                    graduation_stage=stage,
                )
            self._rf_hits += 1
            return KernelDecision(
                source=DecisionSource.RF_ROUTER,
                action_id=rf_act,
                should_wake_llm=False,
                confidence=rf_prob,
                regime_id=regime_id,
                ood_distance=ood_dist,
                is_ood=False,
                state_vector=x_vec.astype(float).tolist(),
                predicted_latency_ms=pred_lat,
                predicted_fail_prob=fail_p,
                safety_allowed=True,
                top_attributions=rf_attrs,
                shadow_rf_action=rf_act,
                shadow_rf_prob=rf_prob,
                hot_path_latency_ms=elapsed_ms,
                graduation_stage=stage,
            )

        # (5) Escalate to Stateless LLM Battery (log RF shadow prediction for benchmarking)
        self._llm_wakeups += 1
        pred_lat, _, _ = bundle.planner_heuristic.estimate_plan_step(x_vec, fallback_latency_ms=300.0)
        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 4)
        self._hot_path_latencies_ms.append(elapsed_ms)
        return KernelDecision(
            source=DecisionSource.LLM_BATTERY,
            action_id=None,
            should_wake_llm=True,
            confidence=rf_prob,
            regime_id=regime_id,
            ood_distance=ood_dist,
            is_ood=False,
            state_vector=x_vec.astype(float).tolist(),
            predicted_latency_ms=pred_lat,
            predicted_fail_prob=0.0,
            safety_allowed=True,
            top_attributions=rf_attrs,
            shadow_rf_action=rf_act or None,
            shadow_rf_prob=rf_prob if rf_act else None,
            exploration_audited=exploration_hit,
            hot_path_latency_ms=elapsed_ms,
            graduation_stage=stage,
        )

    def prune_context(
        self,
        query_text: str,
        memory_items: Sequence[Dict[str, Any]],
        *,
        regime_id: int = 0,
        max_items: int = 6,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """FR-04: Selective memory filtering before waking the Stateless LLM Battery."""
        return self._active_bundle.context_pruner.prune(
            query_text,
            memory_items,
            current_regime=regime_id,
            max_items=max_items,
        )

    def gate_tool_execution(
        self,
        tool_id: str,
        payload: Optional[Dict[str, Any]] = None,
        prompt: str = "",
        world_state: Optional[WorldStateSnapshot] = None,
    ) -> Tuple[bool, float, Optional[str], List[FeatureAttribution]]:
        """FR-07: Pre-execution Safety Gatekeeper (Hard Rules + RF risk check) for ANY tool source."""
        bundle = self._active_bundle
        cand = {
            "tool_id": tool_id,
            "is_mutating": tool_id not in self.read_only_tools,
            "is_high_blast_radius": any(p in tool_id.lower() for p in bundle.safety_gatekeeper.HIGH_BLAST_TOOL_PATTERNS),
            "payload_size_bytes": len(str(payload or "").encode("utf-8")),
        }
        x_vec = self.feature_extractor.extract(prompt or tool_id, world_state=world_state, telemetry=self.telemetry, candidate=cand)
        block_ml = self.get_graduation_stage() == GraduationStage.AUTONOMOUS
        allowed, fail_prob, reason, attrs = bundle.safety_gatekeeper.evaluate(
            tool_id, x_vec, payload, block_on_ml_risk=block_ml
        )
        if not allowed:
            self._safety_blocks += 1
        return allowed, fail_prob, reason, attrs

    def classify_failure_recovery(
        self,
        error_msg: str,
        prompt: str = "",
        attempt: int = 1,
        world_state: Optional[WorldStateSnapshot] = None,
    ) -> Tuple[RecoveryStrategy, float, float]:
        """FR-06.2: Adaptive retry vs fallback vs LLM escalation classification."""
        x_vec = self.feature_extractor.extract(prompt, world_state=world_state, telemetry=self.telemetry)
        return self._active_bundle.planner_heuristic.classify_recovery(error_msg, x_vec=x_vec, attempt=attempt)

    def rank_candidate_plans(
        self,
        prompt: str,
        candidate_plans: Sequence[Dict[str, Any]],
        world_state: Optional[WorldStateSnapshot] = None,
    ) -> List[Dict[str, Any]]:
        """FR-06.1: Rank multiple valid plans/blueprints using LightGBM/HistGBDT P90 latency & utility."""
        pairs: List[Tuple[str, np.ndarray]] = []
        for idx, plan in enumerate(candidate_plans):
            cid = str(plan.get("id") or plan.get("tool_id") or f"plan_{idx}")
            xv = self.feature_extractor.extract(prompt, world_state=world_state, telemetry=self.telemetry, candidate=plan)
            pairs.append((cid, xv))
        return self._active_bundle.planner_heuristic.rank_candidates(pairs)

    def record_execution(
        self,
        *,
        episode_id: str,
        prompt: str,
        decision: Optional[KernelDecision],
        action_id: str,
        action_args: Optional[Dict[str, Any]] = None,
        action_sequence: Optional[List[str]] = None,
        status: OutcomeStatus,
        verifier_passed: bool,
        actual_latency_ms: float,
        reward_signal: Optional[float] = None,
        error_message: Optional[str] = None,
        tokens_used: int = 0,
        capsule_ids: Optional[List[str]] = None,
        world_state: Optional[WorldStateSnapshot] = None,
    ) -> ExecutionTrace:
        """FR-08: Emit canonical ExecutionTrace, update telemetry, circuit breakers, and episodic compressor."""
        if decision is not None:
            vec = list(decision.state_vector)
            regime_id = decision.regime_id
            ood_dist = decision.ood_distance
            src = decision.source
            rf_prob = decision.confidence if src in (DecisionSource.RF_ROUTER, DecisionSource.COMPILED_SKILL) else (decision.shadow_rf_prob or 0.0)
            llm_ovr = src == DecisionSource.LLM_BATTERY
            pred_lat = decision.predicted_latency_ms
            pred_fail = decision.predicted_fail_prob
            attrs = decision.top_attributions
            shadow_act = decision.shadow_rf_action
            shadow_p = decision.shadow_rf_prob
            exp_aud = decision.exploration_audited
            if decision.matched_skill is not None:
                decision.matched_skill.record_outcome(bool(verifier_passed and status == OutcomeStatus.SUCCESS))
        else:
            xv = self.feature_extractor.extract(
                prompt,
                world_state=world_state,
                telemetry=self.telemetry,
                candidate={"tool_id": action_id, "is_mutating": action_id not in self.read_only_tools},
            )
            regime_id, ood_dist, _ = self._active_bundle.regime_ood.evaluate(xv)
            vec = xv.astype(float).tolist()
            src = DecisionSource.LLM_BATTERY
            rf_prob = 0.0
            llm_ovr = True
            pred_lat = actual_latency_ms
            pred_fail = 0.0
            attrs = []
            shadow_act = None
            shadow_p = None
            exp_aud = False

        success = bool(verifier_passed and status == OutcomeStatus.SUCCESS)
        reward = (1.0 if success else -0.5) if reward_signal is None else float(reward_signal)
        self.telemetry.record_execution(
            tool_id=action_id,
            latency_ms=actual_latency_ms,
            success=success,
            regime_id=regime_id,
            tokens_used=tokens_used,
        )

        trace = ExecutionTrace(
            episode_id=episode_id,
            feature_schema_version=FEATURE_SCHEMA_VERSION,
            state_vector=vec,
            regime_id=regime_id,
            ood_distance=ood_dist,
            decision=DecisionRecord(
                source=src,
                action_id=action_id,
                rf_top_prob=rf_prob,
                llm_override=llm_ovr,
                predicted_latency_ms=pred_lat,
                predicted_fail_prob=pred_fail,
                top_feature_attributions=attrs,
                shadow_rf_action=shadow_act,
                shadow_rf_prob=shadow_p,
                exploration_audited=exp_aud,
            ),
            outcome=OutcomeRecord(
                status=status,
                actual_latency_ms=round(float(actual_latency_ms), 2),
                verifier_passed=bool(verifier_passed),
                reward_signal=reward,
                error_message=error_message,
            ),
            action_sequence=list(action_sequence or [action_id]),
            action_args=dict(action_args or {}),
            prompt_excerpt=(prompt or "")[:200],
            is_read_only=action_id in self.read_only_tools,
            capsule_ids=list(capsule_ids or []),
        )
        self.trace_store.append(trace)
        self.memory_compressor.ingest(trace)
        return trace

    def check_drift(self) -> Dict[str, Any]:
        """NFR: Compute PSI drift over recent state vectors and temporarily raise tau if PSI > 0.20."""
        if len(self._recent_vectors) < 10:
            return {"psi": 0.0, "drift_detected": False, "tau_bump": self._tau_drift_bump}
        recent_mat = np.vstack(list(self._recent_vectors))
        psi, drifted = self.drift_monitor.compute_psi(recent_mat)
        self._tau_drift_bump = 0.07 if drifted else 0.0
        return {"psi": psi, "drift_detected": drifted, "tau_bump": self._tau_drift_bump}

    def run_cold_path_cycle(self, *, force_promote: bool = False) -> Dict[str, Any]:
        """FR-12: Execute Cold Path distillation, Shadow Validation Gate, and Atomic Hot-Swap."""
        candidate_bundle, report = self.trainer.train_and_validate(
            self.trace_store,
            self._active_bundle,
            drift_monitor=self.drift_monitor,
            force_promote=force_promote,
        )
        if candidate_bundle is not None:
            with self._swap_lock:
                self._active_bundle = candidate_bundle
                self._tau_drift_bump = 0.0
        report["graduation_stage"] = self.get_graduation_stage().name
        report["episodic_compression"] = self.memory_compressor.summary()
        return report

    def register_compiled_skill(self, skill: CompiledSkill) -> None:
        """Register or replace a validated CompiledSkill atomically in the active bundle."""
        with self._swap_lock:
            self._active_bundle.compiled_skills[skill.skill_id] = skill

    def get_stats(self) -> Dict[str, Any]:
        lats = sorted(self._hot_path_latencies_ms)
        p50 = lats[len(lats) // 2] if lats else 0.0
        p99 = lats[min(len(lats) - 1, int(len(lats) * 0.99))] if lats else 0.0
        offload_rate = (
            (self._rf_hits + self._skill_hits) / float(self._total_decisions)
            if self._total_decisions > 0
            else 0.0
        )
        active_skills = [s.skill_id for s in self._active_bundle.compiled_skills.values() if s.active]
        return {
            "bundle_version": self._active_bundle.version,
            "graduation_stage": self.get_graduation_stage().name,
            "verified_traces": self.trace_store.count_verified(),
            "total_traces": len(self.trace_store.all_traces()),
            "total_decisions": self._total_decisions,
            "llm_wakeups": self._llm_wakeups,
            "rf_hits": self._rf_hits,
            "skill_hits": self._skill_hits,
            "ood_escalations": self._ood_escalations,
            "safety_blocks": self._safety_blocks,
            "llm_offload_rate": round(offload_rate, 4),
            "hot_path_p50_ms": round(p50, 4),
            "hot_path_p99_ms": round(p99, 4),
            "shadow_agreement": self._active_bundle.shadow_agreement,
            "calibration_ece": self._active_bundle.calibration_ece,
            "psi_drift": self.drift_monitor.last_psi,
            "drift_detected": self.drift_monitor.drift_detected,
            "active_skills": active_skills,
            "episodic_compression": self.memory_compressor.summary(),
        }
