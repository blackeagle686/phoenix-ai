"""Phoenix Micro CVAgent — Ultra-Low-RAM Edge/Phone Computer Vision Agent (<= 1B MoE only).

Lives side-by-side with `phoenix.framework.agent` with zero heavy ML imports on startup.
"""

from phoenix.framework.cv_agent.agent import CVAgent, CVAgentConfig, CVAgentResult, MicroCVAgent
from phoenix.framework.cv_agent.context import ContextSnapshot, SlidingCVContext
from phoenix.framework.cv_agent.decision import (
    ContextDecisionRule,
    DecisionGroup,
    MicroDecisionEngine,
    MicroToolCall,
    TinyDecision,
)
from phoenix.framework.cv_agent.events import CVEvent, CVSeverity, Detection
from phoenix.framework.cv_agent.moe_model import (
    HARD_MAX_MOE_PARAMS_B,
    MICRO_MOE_PRESETS,
    MicroMoELLM,
    MicroMoEPolicyError,
    MicroMoESpec,
    validate_micro_moe_model,
)
from phoenix.framework.cv_agent.tools import (
    AlertDispatcher,
    LocalFileStore,
    LocalSQLiteStore,
    MicroTool,
    MicroToolRegistry,
    MicroToolResult,
)

__all__ = [
    "CVAgent",
    "MicroCVAgent",
    "CVAgentConfig",
    "CVAgentResult",
    "CVEvent",
    "CVSeverity",
    "Detection",
    "SlidingCVContext",
    "ContextSnapshot",
    "TinyDecision",
    "MicroToolCall",
    "DecisionGroup",
    "ContextDecisionRule",
    "MicroDecisionEngine",
    "HARD_MAX_MOE_PARAMS_B",
    "MICRO_MOE_PRESETS",
    "MicroMoESpec",
    "MicroMoEPolicyError",
    "MicroMoELLM",
    "validate_micro_moe_model",
    "MicroTool",
    "MicroToolResult",
    "MicroToolRegistry",
    "AlertDispatcher",
    "LocalFileStore",
    "LocalSQLiteStore",
]
