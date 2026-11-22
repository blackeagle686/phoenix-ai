"""PHX v2 Unified Feature Extractor, World State Store & Real-Time Telemetry (FR-01).

Shared identically across Hot Path serving (< 0.5ms CPU) and Cold Path training to
guarantee zero Training-Serving Skew.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

FEATURE_SCHEMA_VERSION = "v2.0"

_SEMANTIC_DIMS = 24
_STRUCTURAL_NAMES = [
    "token_length_norm",
    "has_code_block",
    "entity_count_norm",
    "intent_hash_norm",
]
_WORLD_STATE_NAMES = [
    "step_index_norm",
    "consecutive_failures_norm",
    "last_tool_id_norm",
    "workspace_file_count_norm",
    "has_traceback",
    "requires_security",
    "is_code_task",
    "regime_cluster_id_norm",
]
_TELEMETRY_NAMES = [
    "ewma_tool_latency_norm",
    "recent_error_rate",
    "cpu_util",
    "mem_util",
    "budget_spent_ratio",
    "active_tool_count_norm",
]
_CANDIDATE_NAMES = [
    "tool_id_norm",
    "is_mutating",
    "is_high_blast_radius",
    "payload_size_norm",
    "tool_regime_success_rate",
    "candidate_step_count_norm",
]

FEATURE_NAMES: List[str] = (
    [f"sem_{i:02d}" for i in range(_SEMANTIC_DIMS)]
    + _STRUCTURAL_NAMES
    + _WORLD_STATE_NAMES
    + _TELEMETRY_NAMES
    + _CANDIDATE_NAMES
)
FEATURE_DIM: int = len(FEATURE_NAMES)  # 48 fixed canonical floats

_TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+|[\u0600-\u06FF]+")
_FILE_OR_ENTITY_RE = re.compile(r"\b[A-Za-z0-9_./-]+\.(?:py|js|ts|json|sql|yaml|yml|md|txt)\b|`[^`]+`")
_SECURITY_HINTS = ("secure", "security", "safe", "eval", "sanitize", "auth", "jwt", "otp", "rbac", "أمان", "آمن")
_CODE_HINTS = ("code", "python", "function", "class", "bug", "fix", "edit", "refactor", "api", "fastapi", "endpoint", "capsule", "كود")
_TRACEBACK_HINTS = ("traceback", "error:", "exception:", "line ", "valueerror", "typeerror", "attributeerror", "keyerror")


def stable_id_hash(label: str, buckets: int = 1000) -> float:
    """Map an arbitrary string identifier deterministically into [0.0, 1.0]."""
    if not label:
        return 0.0
    digest = hashlib.blake2b(str(label).encode("utf-8"), digest_size=4).digest()
    val = int.from_bytes(digest, "little") % max(1, buckets)
    return float(val) / float(max(1, buckets))


@dataclass
class WorldStateSnapshot:
    """Deterministic World State snapshot captured prior to a Hot Path decision."""

    episode_id: str = ""
    step_index: int = 0
    consecutive_failures: int = 0
    last_tool_id: str = ""
    workspace_file_count: int = 0
    regime_cluster_id: int = 0
    available_tools: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class RuntimeTelemetryTracker:
    """Maintains low-overhead EWMA latencies, rolling error rates, and per-(tool, regime) stats."""

    def __init__(self, ewma_alpha: float = 0.25, window_size: int = 100):
        self.ewma_alpha = ewma_alpha
        self.ewma_latency_ms: float = 20.0
        self._tool_ewma: Dict[str, float] = {}
        self._recent_outcomes: Deque[bool] = deque(maxlen=max(10, window_size))
        self._tool_regime_stats: Dict[Tuple[str, int], List[int]] = defaultdict(lambda: [0, 0])  # [succ, total]
        self._budget_spent_tokens: int = 0
        self._budget_cap_tokens: int = 100_000
        self._last_sys_sample_ts: float = 0.0
        self._cached_sys_util: Tuple[float, float] = (0.2, 0.3)

    def reset_episode_budget(self, token_cap: int = 100_000) -> None:
        self._budget_spent_tokens = 0
        self._budget_cap_tokens = max(1000, int(token_cap))

    def record_execution(
        self,
        tool_id: str,
        latency_ms: float,
        success: bool,
        regime_id: int = 0,
        tokens_used: int = 0,
    ) -> None:
        lat = max(0.0, float(latency_ms))
        self.ewma_latency_ms = (1.0 - self.ewma_alpha) * self.ewma_latency_ms + self.ewma_alpha * lat
        prev = self._tool_ewma.get(tool_id, lat)
        self._tool_ewma[tool_id] = (1.0 - self.ewma_alpha) * prev + self.ewma_alpha * lat
        self._recent_outcomes.append(bool(success))
        rec = self._tool_regime_stats[(tool_id, int(regime_id))]
        rec[1] += 1
        if success:
            rec[0] += 1
        self._budget_spent_tokens += max(0, int(tokens_used))

    def get_tool_ewma_latency(self, tool_id: str) -> float:
        return self._tool_ewma.get(tool_id, self.ewma_latency_ms)

    def get_recent_error_rate(self) -> float:
        if not self._recent_outcomes:
            return 0.0
        errs = sum(1 for ok in self._recent_outcomes if not ok)
        return errs / len(self._recent_outcomes)

    def get_tool_regime_success_rate(self, tool_id: str, regime_id: int) -> float:
        succ, total = self._tool_regime_stats.get((tool_id, int(regime_id)), [0, 0])
        # Laplace smoothing with optimistic prior (0.85) so cold tools aren't unfairly penalized
        return (succ + 1.7) / (total + 2.0)

    def get_budget_spent_ratio(self) -> float:
        return min(1.0, self._budget_spent_tokens / float(max(1, self._budget_cap_tokens)))

    def get_system_utilization(self) -> Tuple[float, float]:
        """Cached non-blocking CPU/memory utilization ratio in [0.0, 1.0] sampled at most once per 2s."""
        now = time.monotonic()
        if now - self._last_sys_sample_ts < 2.0:
            return self._cached_sys_util
        self._last_sys_sample_ts = now
        cpu_u, mem_u = 0.2, 0.3
        try:
            import psutil

            cpu_u = min(1.0, max(0.0, psutil.cpu_percent(interval=None) / 100.0))
            mem_u = min(1.0, max(0.0, psutil.virtual_memory().percent / 100.0))
        except Exception:
            try:
                load1 = os.getloadavg()[0]
                cpu_u = min(1.0, load1 / max(1, os.cpu_count() or 1))
            except Exception:
                pass
        self._cached_sys_util = (cpu_u, mem_u)
        return self._cached_sys_util


class FeatureExtractor:
    """Shared Train/Serve canonical feature extractor producing X_t in R^48 in < 0.5ms CPU."""

    schema_version: str = FEATURE_SCHEMA_VERSION
    feature_names: List[str] = FEATURE_NAMES
    dim: int = FEATURE_DIM

    def __init__(self, svd_components: Optional[np.ndarray] = None):
        # Optional learned projection matrix (from TruncatedSVD / PCA) for external dense embeddings
        self._projection = svd_components

    def _hash_text_to_semantic(self, text: str) -> np.ndarray:
        """Signed feature hashing (char 3-grams + word tokens) into 24-D unit-normalized float32 vector."""
        vec = np.zeros(_SEMANTIC_DIMS, dtype=np.float32)
        if not text:
            return vec
        low = text.lower()
        tokens = _TOKEN_RE.findall(low)
        if not tokens:
            return vec
        # Word unigrams + bigrams for clean intent separation (read vs edit vs search vs sql vs compose)
        features: List[str] = list(tokens[:160])
        for i in range(min(len(tokens) - 1, 80)):
            features.append(f"{tokens[i]}_{tokens[i + 1]}")
        for tok in features:
            h = hashlib.blake2b(tok.encode("utf-8"), digest_size=4).digest()
            idx = h[0] % _SEMANTIC_DIMS
            sign = 1.0 if (h[1] & 1) == 0 else -1.0
            vec[idx] += sign
        norm = float(np.linalg.norm(vec))
        if norm > 1e-8:
            vec /= norm
        return vec

    def compress_embedding(self, raw_embedding: Optional[List[float]]) -> Optional[np.ndarray]:
        """FR-01.3: Compress high-D embeddings (e.g. 1536-D) into 24-D via learned projection or folded average."""
        if not raw_embedding:
            return None
        arr = np.asarray(raw_embedding, dtype=np.float32).ravel()
        if arr.size == 0:
            return None
        if self._projection is not None and self._projection.shape[0] == arr.size:
            proj = arr @ self._projection
            out = proj[:_SEMANTIC_DIMS].astype(np.float32)
        elif arr.size == _SEMANTIC_DIMS:
            out = arr
        else:
            # Deterministic folded projection into 24 dims (zero-latency fallback before PCA/SVD fit)
            out = np.zeros(_SEMANTIC_DIMS, dtype=np.float32)
            for i, val in enumerate(arr):
                out[i % _SEMANTIC_DIMS] += val
        norm = float(np.linalg.norm(out))
        if norm > 1e-8:
            out /= norm
        return out

    def extract(
        self,
        text: str,
        world_state: Optional[WorldStateSnapshot] = None,
        telemetry: Optional[RuntimeTelemetryTracker] = None,
        candidate: Optional[Dict[str, Any]] = None,
        raw_embedding: Optional[List[float]] = None,
    ) -> np.ndarray:
        """Build canonical fixed-length vector X_t in R^48 (< 0.5ms CPU)."""
        ws = world_state or WorldStateSnapshot()
        tel = telemetry or RuntimeTelemetryTracker()
        cand = candidate or {}

        low = (text or "").lower()
        tokens = _TOKEN_RE.findall(low)

        sem = self.compress_embedding(raw_embedding)
        if sem is None:
            sem = self._hash_text_to_semantic(text)

        # Segment 1b: Structural scalars (4)
        token_len_norm = min(1.0, math.log1p(len(tokens)) / 7.0)
        has_code_block = 1.0 if ("```" in text or "def " in text or "class " in text) else 0.0
        entity_count_norm = min(1.0, len(_FILE_OR_ENTITY_RE.findall(text or "")) / 10.0)
        intent_hash = stable_id_hash(" ".join(tokens[:6]))

        # Segment 2: World State features (8)
        step_norm = min(1.0, ws.step_index / 20.0)
        consec_fail_norm = min(1.0, ws.consecutive_failures / 5.0)
        last_tool_norm = stable_id_hash(ws.last_tool_id)
        ws_files_norm = min(1.0, math.log1p(max(0, ws.workspace_file_count)) / 6.0)
        has_tb = 1.0 if any(h in low for h in _TRACEBACK_HINTS) else 0.0
        req_sec = 1.0 if any(h in low for h in _SECURITY_HINTS) else 0.0
        is_code = 1.0 if (has_code_block > 0 or any(h in low for h in _CODE_HINTS)) else 0.0
        regime_norm = min(1.0, max(0.0, float(ws.regime_cluster_id)) / 10.0)

        # Segment 3: Telemetry features (6)
        tool_id = str(cand.get("tool_id") or ws.last_tool_id or "")
        ewma_lat_norm = min(1.0, math.log1p(max(0.0, tel.get_tool_ewma_latency(tool_id))) / 9.0)
        err_rate = float(tel.get_recent_error_rate())
        cpu_u, mem_u = tel.get_system_utilization()
        budget_ratio = float(tel.get_budget_spent_ratio())
        avail_tools_norm = min(1.0, len(ws.available_tools) / 25.0)

        # Segment 4: Action candidate features (6)
        tool_norm = stable_id_hash(tool_id)
        is_mutating = 1.0 if cand.get("is_mutating") else 0.0
        is_high_blast = 1.0 if cand.get("is_high_blast_radius") else 0.0
        payload_bytes = float(cand.get("payload_size_bytes") or 0.0)
        payload_norm = min(1.0, math.log1p(max(0.0, payload_bytes)) / 12.0)
        tool_reg_sr = float(tel.get_tool_regime_success_rate(tool_id, ws.regime_cluster_id)) if tool_id else 0.85
        cand_steps_norm = min(1.0, float(cand.get("step_count") or 1.0) / 10.0)

        scalars = np.array(
            [
                token_len_norm, has_code_block, entity_count_norm, intent_hash,
                step_norm, consec_fail_norm, last_tool_norm, ws_files_norm,
                has_tb, req_sec, is_code, regime_norm,
                ewma_lat_norm, err_rate, cpu_u, mem_u, budget_ratio, avail_tools_norm,
                tool_norm, is_mutating, is_high_blast, payload_norm, tool_reg_sr, cand_steps_norm,
            ],
            dtype=np.float32,
        )
        return np.concatenate([sem, scalars])
