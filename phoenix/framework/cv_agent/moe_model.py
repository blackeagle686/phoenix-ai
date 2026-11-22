"""Strict <= 1B MoE Model Policy & Ultra-Low-RAM Phone/Edge CPU Inference Engine.

Enforces two non-negotiable hardware invariants for CVAgent:
  1. Model MUST be a Mixture-of-Experts (MoE) architecture.
  2. Model parameter budget MUST be <= 1.0B (strict hard ceiling; rejects dense models
     and rejects >1B MoE models before allocating any RAM).

Zero heavy imports (`torch`, `transformers`, `llama_cpp`) are executed at module import
time so edge devices and mobile phones pay near-zero baseline memory overhead.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import re
import sys
import threading
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

HARD_MAX_MOE_PARAMS_B: float = 1.0
_DC_FROZEN_SLOTS: Dict[str, Any] = {"frozen": True, "slots": True} if sys.version_info >= (3, 10) else {"frozen": True}


class MicroMoEPolicyError(ValueError):
    """Raised when a model violates the CVAgent `MoE <= 1B` hardware policy."""


@dataclass(**_DC_FROZEN_SLOTS)
class MicroMoESpec:
    """Validated metadata for an allowed <= 1B MoE model."""

    model_id: str
    params_b: float
    active_params_b: float
    num_experts: int = 8
    experts_per_tok: int = 2
    gguf_repo: Optional[str] = None
    gguf_filename: Optional[str] = None
    ram_q4_mb: int = 650
    is_moe: bool = True


# Curated <= 1B MoE presets suitable for phone CPUs & low-RAM edge devices
MICRO_MOE_PRESETS: Dict[str, Dict[str, Any]] = {
    "ibm-granite/granite-3.1-1b-a400m-instruct": {
        "aliases": (
            "granite-1b-moe",
            "granite-moe-1b",
            "granite-400m",
            "granite-3.1-1b-a400m",
            "micro-granite",
        ),
        "params_b": 1.0,
        "active_params_b": 0.4,
        "num_experts": 32,
        "experts_per_tok": 8,
        "gguf_repo": "bartowski/granite-3.1-1b-a400m-instruct-GGUF",
        "gguf_filename": "*Q4_K_M.gguf",
        "ram_q4_mb": 780,
        "is_moe": True,
    },
    "ibm-granite/granite-3.0-1b-a400m-instruct": {
        "aliases": ("granite-3.0-1b-moe", "granite-3.0-400m"),
        "params_b": 1.0,
        "active_params_b": 0.4,
        "num_experts": 32,
        "experts_per_tok": 8,
        "gguf_repo": "bartowski/granite-3.0-1b-a400m-instruct-GGUF",
        "gguf_filename": "*Q4_K_M.gguf",
        "ram_q4_mb": 780,
        "is_moe": True,
    },
    "micro-moe-500m-a150m": {
        "aliases": ("micro-moe-500m", "tiny-moe-0.5b", "edge-moe-500m"),
        "params_b": 0.5,
        "active_params_b": 0.15,
        "num_experts": 8,
        "experts_per_tok": 2,
        "gguf_repo": None,
        "gguf_filename": None,
        "ram_q4_mb": 340,
        "is_moe": True,
    },
    "micro-moe-300m-a100m": {
        "aliases": ("micro-moe-300m", "nano-moe-300m", "phone-moe-300m"),
        "params_b": 0.3,
        "active_params_b": 0.1,
        "num_experts": 8,
        "experts_per_tok": 2,
        "gguf_repo": None,
        "gguf_filename": None,
        "ram_q4_mb": 220,
        "is_moe": True,
    },
}

_MOE_NAME_HINTS = (
    "moe",
    "mixtral",
    "olmoe",
    "switch",
    "experts",
    "a400m",
    "a300m",
    "a250m",
    "a200m",
    "a150m",
    "a100m",
    "a0.5b",
    "a0.4b",
    "a0.3b",
    "a0.2b",
    "a0.1b",
    "a800m",
    "a2.7b",
)

# Matches patterns like `1b`, `0.5b`, `400m`, `8x7b`, `a400m`, `a2.7b`, `1.5b`
_MULTI_EXPERT_SIZE_RE = re.compile(r"(?<![a-z0-9])(\d+)\s*[xX]\s*(\d+(?:\.\d+)?)\s*([bmBM])(?![a-z0-9])")
_ACTIVE_SIZE_RE = re.compile(r"(?<![a-z0-9])a(\d+(?:\.\d+)?)\s*([bmBM])(?![a-z0-9])")
_PLAIN_SIZE_RE = re.compile(r"(?<![a-z0-9])(\d+(?:\.\d+)?)\s*([bmBM])(?![a-z0-9])")


def _to_billions(val: float, unit: str) -> float:
    return round(val / 1000.0, 4) if unit.lower() == "m" else round(val, 4)


def _infer_params_from_name(model_name: str) -> Tuple[Optional[float], Optional[float], bool]:
    """Infer `(total_params_b, active_params_b, looks_like_moe)` from model name/filename."""
    low = (model_name or "").lower()
    looks_moe = any(h in low for h in _MOE_NAME_HINTS)

    # Check NxMB/NxGB expert multipliers first (e.g. `8x7b` -> 56B, `4x150m` -> 0.6B)
    mx = _MULTI_EXPERT_SIZE_RE.search(low)
    if mx:
        looks_moe = True
        n_exp = int(mx.group(1))
        per_exp = _to_billions(float(mx.group(2)), mx.group(3))
        total_b = round(n_exp * per_exp, 4)
        active_b = round(min(total_b, per_exp * 2.0), 4)
        return total_b, active_b, True

    active_b: Optional[float] = None
    act_m = _ACTIVE_SIZE_RE.search(low)
    if act_m:
        looks_moe = True
        active_b = _to_billions(float(act_m.group(1)), act_m.group(2))

    # Collect all plain sizes (excluding the matched `a<num><unit>` span)
    cleaned = _ACTIVE_SIZE_RE.sub(" ", low)
    # Strip quant markers like `q4_k_m` so they never confuse size parsing
    cleaned = re.sub(r"q\d(?:_[a-z0-9]+)*", " ", cleaned)
    plain_sizes = [_to_billions(float(m.group(1)), m.group(2)) for m in _PLAIN_SIZE_RE.finditer(cleaned)]
    total_b = max(plain_sizes) if plain_sizes else active_b

    # Handle `olmoe-1b-7b` style where the larger size is total and smaller is active
    if len(plain_sizes) >= 2 and active_b is None:
        total_b = max(plain_sizes)
        active_b = min(plain_sizes)

    if active_b is None and total_b is not None:
        active_b = total_b

    return total_b, active_b, looks_moe


def _inspect_local_config_json(model_path: str) -> Tuple[Optional[float], Optional[bool], Optional[int], Optional[int]]:
    """Inspect local HF `config.json` if `model_path` is a local directory (zero HF/torch import)."""
    cfg_path = os.path.join(model_path, "config.json") if os.path.isdir(model_path) else None
    if not cfg_path or not os.path.isfile(cfg_path):
        return None, None, None, None
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception:
        return None, None, None, None

    num_experts = cfg.get("num_local_experts") or cfg.get("num_experts") or cfg.get("n_routed_experts")
    top_k = cfg.get("num_experts_per_tok") or cfg.get("top_k")
    is_moe = bool(num_experts and int(num_experts) > 1)
    if not is_moe:
        archs = " ".join(str(a).lower() for a in (cfg.get("architectures") or []))
        mtype = str(cfg.get("model_type") or "").lower()
        if "moe" in archs or "moe" in mtype or "mixtral" in mtype:
            is_moe = True
    params_b = cfg.get("num_parameters_b") or cfg.get("params_b")
    return (
        float(params_b) if params_b is not None else None,
        is_moe,
        int(num_experts) if num_experts else None,
        int(top_k) if top_k else None,
    )


def validate_micro_moe_model(
    model: str,
    *,
    params_b: Optional[float] = None,
    active_params_b: Optional[float] = None,
    is_moe: Optional[bool] = None,
    max_params_b: float = HARD_MAX_MOE_PARAMS_B,
) -> MicroMoESpec:
    """Strictly validate that `model` is a Mixture-of-Experts model with `<= 1.0B` parameters."""
    if not model or not str(model).strip():
        raise MicroMoEPolicyError("Model name/path cannot be empty.")

    effective_cap = min(float(max_params_b), HARD_MAX_MOE_PARAMS_B)
    if effective_cap <= 0:
        raise MicroMoEPolicyError("max_params_b must be positive and <= 1.0B.")

    raw = str(model).strip()
    low = raw.lower()

    # 1. Exact or alias preset lookup
    for canonical, meta in MICRO_MOE_PRESETS.items():
        if canonical.lower() == low or low in meta.get("aliases", ()):
            p_b = float(params_b if params_b is not None else meta["params_b"])
            a_b = float(active_params_b if active_params_b is not None else meta["active_params_b"])
            if p_b > effective_cap or a_b > effective_cap:
                raise MicroMoEPolicyError(
                    f"Model '{canonical}' ({p_b}B total / {a_b}B active) exceeds CVAgent ceiling of {effective_cap}B."
                )
            return MicroMoESpec(
                model_id=canonical,
                params_b=p_b,
                active_params_b=a_b,
                num_experts=int(meta.get("num_experts", 8)),
                experts_per_tok=int(meta.get("experts_per_tok", 2)),
                gguf_repo=meta.get("gguf_repo"),
                gguf_filename=meta.get("gguf_filename"),
                ram_q4_mb=int(meta.get("ram_q4_mb", 650)),
                is_moe=True,
            )

    # 2. Check local config.json if directory exists, plus filename/repo heuristics
    cfg_params, cfg_is_moe, cfg_experts, cfg_topk = _inspect_local_config_json(raw)
    name_total, name_active, name_is_moe = _infer_params_from_name(raw)

    resolved_is_moe = (
        bool(is_moe)
        if is_moe is not None
        else (cfg_is_moe if cfg_is_moe is not None else name_is_moe)
    )
    if not resolved_is_moe:
        raise MicroMoEPolicyError(
            f"CVAgent ONLY supports Mixture-of-Experts (MoE) models <= {effective_cap}B. "
            f"Rejected non-MoE model '{raw}'. Use an MoE preset like 'granite-1b-moe' or pass explicit MoE metadata."
        )

    resolved_total = (
        float(params_b)
        if params_b is not None
        else (cfg_params if cfg_params is not None else name_total)
    )
    resolved_active = (
        float(active_params_b)
        if active_params_b is not None
        else (name_active if name_active is not None else resolved_total)
    )

    if resolved_total is None:
        raise MicroMoEPolicyError(
            f"Cannot verify parameter size for MoE model '{raw}'. "
            f"Include parameter size in name (e.g. 'micro-moe-500m.gguf') or pass `params_b <= {effective_cap}`."
        )

    if resolved_total > effective_cap:
        raise MicroMoEPolicyError(
            f"Model '{raw}' has {resolved_total}B parameters, exceeding CVAgent's hard <= {effective_cap}B limit "
            "(designed for phone CPUs & very low RAM)."
        )

    if resolved_active is not None and resolved_active > effective_cap:
        raise MicroMoEPolicyError(
            f"Model '{raw}' has {resolved_active}B active parameters, exceeding CVAgent's hard <= {effective_cap}B limit."
        )

    est_q4_mb = max(80, int(resolved_total * 720))
    return MicroMoESpec(
        model_id=raw,
        params_b=round(resolved_total, 4),
        active_params_b=round(resolved_active if resolved_active is not None else resolved_total, 4),
        num_experts=cfg_experts or 8,
        experts_per_tok=cfg_topk or 2,
        gguf_filename=raw if low.endswith(".gguf") else None,
        ram_q4_mb=est_q4_mb,
        is_moe=True,
    )


class MicroMoELLM:
    """
    Ultra-lightweight <= 1B MoE inference adapter engineered for phone CPUs and low-RAM devices.

    Backends (lazy-loaded only when invoked):
      - `callable`: user-provided lightweight sync/async generator function (0 extra RAM; great for testing/custom runtimes)
      - `llama_cpp`: quantized GGUF (Q4_K_M / Q3_K_M / Q2_K) with `use_mmap=True, use_mlock=False` for phone RAM safety
      - `transformers`: CPU bfloat16/int8 fallback if `llama_cpp` is unavailable
    """

    def __init__(
        self,
        model: str = "ibm-granite/granite-3.1-1b-a400m-instruct",
        *,
        params_b: Optional[float] = None,
        active_params_b: Optional[float] = None,
        is_moe: Optional[bool] = None,
        max_params_b: float = HARD_MAX_MOE_PARAMS_B,
        backend: str = "auto",  # auto | llama_cpp | transformers | callable
        gguf_path: Optional[str] = None,
        n_ctx: int = 1024,
        max_tokens: int = 160,
        n_threads: Optional[int] = None,
        temperature: float = 0.05,
        generate_fn: Optional[Callable[[str, int], Any]] = None,
    ):
        self.spec: MicroMoESpec = validate_micro_moe_model(
            model,
            params_b=params_b,
            active_params_b=active_params_b,
            is_moe=is_moe,
            max_params_b=max_params_b,
        )
        self.model = self.spec.model_id
        self.gguf_path = gguf_path or self.spec.gguf_filename
        self.n_ctx = max(256, min(int(n_ctx), 2048))  # Hard cap KV cache for low-RAM phone safety
        self.max_tokens = max(32, min(int(max_tokens), 384))
        self.n_threads = max(1, min(int(n_threads or (os.cpu_count() or 2)), 4))
        self.temperature = float(temperature)
        self._generate_fn = generate_fn
        self.backend = "callable" if generate_fn is not None else (backend or "auto").lower()

        self._llama: Any = None
        self._hf_model: Any = None
        self._hf_tokenizer: Any = None
        self._lock = threading.Lock()
        self._initialized = False

    def _resolve_backend(self) -> str:
        if self._generate_fn is not None or self.backend == "callable":
            return "callable"
        if self.backend in ("llama_cpp", "transformers"):
            return self.backend
        if self.gguf_path and str(self.gguf_path).lower().endswith(".gguf"):
            if importlib.util.find_spec("llama_cpp") is not None:
                return "llama_cpp"
        if importlib.util.find_spec("llama_cpp") is not None and (self.spec.gguf_repo or os.path.isfile(self.model)):
            return "llama_cpp"
        return "transformers"

    def _init_sync(self) -> None:
        if self._initialized:
            return
        with self._lock:
            if self._initialized:
                return
            chosen = self._resolve_backend()
            self.backend = chosen
            if chosen == "callable":
                self._initialized = True
                return

            if chosen == "llama_cpp":
                from llama_cpp import Llama

                kwargs: Dict[str, Any] = {
                    "n_ctx": self.n_ctx,
                    "n_threads": self.n_threads,
                    "n_gpu_layers": 0,  # Pure CPU by default for phone/edge portability
                    "use_mmap": True,
                    "use_mlock": False,
                    "verbose": False,
                }
                if self.gguf_path and os.path.isfile(self.gguf_path):
                    self._llama = Llama(model_path=self.gguf_path, **kwargs)
                elif os.path.isfile(self.model):
                    self._llama = Llama(model_path=self.model, **kwargs)
                elif self.spec.gguf_repo and self.spec.gguf_filename:
                    self._llama = Llama.from_pretrained(
                        repo_id=self.spec.gguf_repo,
                        filename=self.spec.gguf_filename,
                        **kwargs,
                    )
                else:
                    raise RuntimeError(
                        f"MicroMoELLM llama_cpp requires a local `.gguf` path or preset GGUF repo for '{self.model}'."
                    )
                self._initialized = True
                return

            # Lightweight CPU transformers path (loaded strictly in bfloat16/low_cpu_mem_usage)
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            torch.set_num_threads(self.n_threads)
            self._hf_tokenizer = AutoTokenizer.from_pretrained(self.model)
            if self._hf_tokenizer.pad_token_id is None and self._hf_tokenizer.eos_token_id is not None:
                self._hf_tokenizer.pad_token_id = self._hf_tokenizer.eos_token_id
            load_kw: Dict[str, Any] = {"low_cpu_mem_usage": True, "torch_dtype": torch.bfloat16}
            try:
                self._hf_model = AutoModelForCausalLM.from_pretrained(self.model, **load_kw)
            except TypeError:
                load_kw.pop("torch_dtype", None)
                load_kw["dtype"] = torch.bfloat16
                self._hf_model = AutoModelForCausalLM.from_pretrained(self.model, **load_kw)
            self._hf_model.eval()
            self._initialized = True

    def generate_sync(self, prompt: str, *, max_tokens: Optional[int] = None) -> str:
        """Synchronous micro-generation suitable for embedded OpenCV/edge loops."""
        self._init_sync()
        out_cap = min(int(max_tokens or self.max_tokens), 384)
        if self.backend == "callable":
            assert self._generate_fn is not None
            res = self._generate_fn(prompt, out_cap)
            if asyncio.iscoroutine(res):
                raise RuntimeError("Async generate_fn passed to generate_sync(); call `await generate(...)` instead.")
            return str(res or "").strip()

        with self._lock:
            if self.backend == "llama_cpp":
                resp = self._llama.create_chat_completion(
                    messages=[
                        {"role": "system", "content": "You are a micro CV edge decision engine. Reply concisely in compact JSON."},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=out_cap,
                    temperature=self.temperature,
                )
                return str(((resp.get("choices") or [{}])[0].get("message") or {}).get("content") or "").strip()

            import torch

            inputs = self._hf_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.n_ctx - out_cap)
            with torch.inference_mode():
                out_ids = self._hf_model.generate(
                    **inputs,
                    max_new_tokens=out_cap,
                    do_sample=self.temperature > 0.05,
                    temperature=max(self.temperature, 1e-4) if self.temperature > 0.05 else None,
                    use_cache=True,
                    pad_token_id=self._hf_tokenizer.pad_token_id or self._hf_tokenizer.eos_token_id,
                )
            gen_ids = out_ids[0][inputs["input_ids"].shape[1] :]
            return self._hf_tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    async def generate(self, prompt: str, *, max_tokens: Optional[int] = None) -> str:
        """Async micro-generation (offloads blocking CPU inference off the asyncio loop)."""
        if self._generate_fn is not None:
            out_cap = min(int(max_tokens or self.max_tokens), 384)
            res = self._generate_fn(prompt, out_cap)
            if asyncio.iscoroutine(res):
                return str(await res or "").strip()
            return str(res or "").strip()
        return await asyncio.to_thread(self.generate_sync, prompt, max_tokens=max_tokens)

    def info(self) -> Dict[str, Any]:
        return {
            "model": self.spec.model_id,
            "params_b": self.spec.params_b,
            "active_params_b": self.spec.active_params_b,
            "num_experts": self.spec.num_experts,
            "experts_per_tok": self.spec.experts_per_tok,
            "estimated_q4_ram_mb": self.spec.ram_q4_mb,
            "backend": self.backend,
            "n_ctx": self.n_ctx,
            "n_threads": self.n_threads,
            "initialized": self._initialized,
        }
