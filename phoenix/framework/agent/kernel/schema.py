"""PHX v2 Kernel Canonical Data Schemas (ExecutionTrace, Decisions, Outcomes, Compiled Skills).

Implements the data contracts specified in PHX-SPEC-V2.0 (FR-01, FR-08, FR-09, Section 5.2).
"""

from __future__ import annotations

import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any, Deque, Dict, List, Optional

from pydantic import BaseModel, Field


class DecisionSource(str, Enum):
    COMPILED_SKILL = "COMPILED_SKILL"
    RF_ROUTER = "RF_ROUTER"
    LLM_BATTERY = "LLM_BATTERY"
    HARD_RULE = "HARD_RULE"
    FALLBACK = "FALLBACK"


class OutcomeStatus(str, Enum):
    SUCCESS = "SUCCESS"
    RETRYABLE_ERR = "RETRYABLE_ERR"
    FATAL_ERR = "FATAL_ERR"
    BLOCKED = "BLOCKED"


class RecoveryStrategy(str, Enum):
    WAIT_AND_RETRY = "WAIT_AND_RETRY"
    SWITCH_FALLBACK = "SWITCH_FALLBACK"
    ESCALATE_TO_LLM = "ESCALATE_TO_LLM"


class GraduationStage(IntEnum):
    SHADOW = 1         # 0 .. 1,000 verified traces (zero ML control authority)
    CONSERVATIVE = 2   # 1,000 .. 10,000 verified traces (tau=0.92, read-only/idempotent only)
    AUTONOMOUS = 3     # 10,000+ verified traces (tau~0.85 + compiled skills)


class FeatureAttribution(BaseModel):
    feature: str
    weight: float


class DecisionRecord(BaseModel):
    source: DecisionSource
    action_id: str
    rf_top_prob: float = 0.0
    llm_override: bool = False
    predicted_latency_ms: float = 0.0
    predicted_fail_prob: float = 0.0
    top_feature_attributions: List[FeatureAttribution] = Field(default_factory=list)
    shadow_rf_action: Optional[str] = None
    shadow_rf_prob: Optional[float] = None
    exploration_audited: bool = False


class OutcomeRecord(BaseModel):
    status: OutcomeStatus
    actual_latency_ms: float = 0.0
    verifier_passed: bool = False
    reward_signal: float = 0.0
    error_message: Optional[str] = None


class ExecutionTrace(BaseModel):
    """Canonical ExecutionTrace schema emitted by Hot Path and consumed by Cold Path."""

    trace_id: str = Field(default_factory=lambda: f"tr_{uuid.uuid4().hex[:10]}")
    episode_id: str = Field(default_factory=lambda: f"ep_{uuid.uuid4().hex[:8]}")
    timestamp_ms: int = Field(default_factory=lambda: int(time.time() * 1000))
    feature_schema_version: str = "v2.0"
    state_vector: List[float] = Field(default_factory=list)
    regime_id: int = 0
    ood_distance: float = 0.0
    decision: DecisionRecord
    outcome: OutcomeRecord
    # Optional compact metadata for skill mining / replay validation
    action_sequence: List[str] = Field(default_factory=list)
    action_args: Dict[str, Any] = Field(default_factory=dict)
    prompt_excerpt: str = ""
    is_read_only: bool = True
    capsule_ids: List[str] = Field(default_factory=list)


@dataclass
class CompiledSkill:
    """Deterministic Parameterized Macro extracted by HDBSCAN Skill Miner (FR-09)."""

    skill_id: str
    name: str
    action_sequence: List[str]
    precondition_centroid: List[float]
    precondition_radius: float
    regime_ids: List[int] = field(default_factory=list)
    required_keywords: List[str] = field(default_factory=list)
    default_arg_templates: List[Dict[str, Any]] = field(default_factory=list)
    postcondition_checks: List[str] = field(default_factory=list)
    verified_support: int = 0
    initial_success_rate: float = 1.0
    window_size: int = 20
    min_success_rate: float = 0.90
    active: bool = True
    demoted_reason: Optional[str] = None
    recent_outcomes: Deque[bool] = field(default_factory=lambda: deque(maxlen=20))

    def __post_init__(self) -> None:
        if self.recent_outcomes.maxlen != self.window_size:
            self.recent_outcomes = deque(self.recent_outcomes, maxlen=max(5, int(self.window_size)))

    def record_outcome(self, passed: bool) -> bool:
        """Record an invocation outcome and enforce the FR-09.5 sliding-window circuit breaker."""
        self.recent_outcomes.append(bool(passed))
        if len(self.recent_outcomes) >= min(self.window_size, 5):
            rate = sum(1 for x in self.recent_outcomes if x) / len(self.recent_outcomes)
            if rate < self.min_success_rate:
                self.active = False
                self.demoted_reason = (
                    f"Circuit breaker tripped: rolling success rate {rate:.2%} < {self.min_success_rate:.0%} "
                    f"over last {len(self.recent_outcomes)} runs."
                )
        return self.active

    @property
    def rolling_success_rate(self) -> float:
        if not self.recent_outcomes:
            return self.initial_success_rate
        return sum(1 for x in self.recent_outcomes if x) / len(self.recent_outcomes)
