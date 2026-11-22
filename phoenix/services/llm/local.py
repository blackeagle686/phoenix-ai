import asyncio
import gc
import inspect
import json
import os
import threading
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

import httpx

from phoenix.core.config import config
from phoenix.core.container import container
from phoenix.core.hardware_check import HardwareChecker
from phoenix.services.llm.base import BaseLLM, _extract_json_dict
from phoenix.services.observability.logger import get_logger
from phoenix.services.observability.tracing import tracer

logger = get_logger("Phoenix AI.LLM.Local")


def _summarize_param_schema(meta: Any, depth: int = 0) -> str:
    if not isinstance(meta, dict):
        return "any"
    if "enum" in meta and isinstance(meta["enum"], list):
        return "enum(" + "|".join(str(x) for x in meta["enum"][:6]) + ")"
    ptype = meta.get("type", "any")
    if ptype == "array" and depth < 2:
        items = meta.get("items") or {}
        if isinstance(items, dict) and items.get("type") == "object" and isinstance(items.get("properties"), dict):
            fields = ",".join(f"{k}:{_summarize_param_schema(v, depth + 1)}" for k, v in list(items["properties"].items())[:4])
            return f"list[{{{fields}}}]"
        return f"list[{_summarize_param_schema(items, depth + 1)}]"
    return str(ptype)


def _compact_tools_for_prompt(tools: List[Dict[str, Any]]) -> str:
    compact = []
    for item in tools or []:
        fn = item.get("function", item) if isinstance(item, dict) else {}
        if isinstance(fn, dict) and fn.get("name"):
            params = fn.get("parameters") or {}
            props = params.get("properties", {}) if isinstance(params, dict) else {}
            req = set(params.get("required", []) if isinstance(params, dict) else ())
            args_map = {
                f"{p_name}{'*' if p_name in req else ''}": _summarize_param_schema(p_meta)
                for p_name, p_meta in props.items()
            }
            desc = " ".join(str(fn.get("description", "")).split())
            compact.append({
                "name": fn.get("name"),
                "desc": desc[:130] + ("..." if len(desc) > 130 else ""),
                "args": args_map,
            })
    return json.dumps(compact, separators=(",", ":"), ensure_ascii=False)


class LocalLLM(BaseLLM):
    """
    High-performance, secure, CPU & GPU optimized Local LLM service.

    Supports downloading and running local models from HuggingFace or local paths via:
      1. `llama_cpp` (GGUF Q4_K_M / Q5_K_M / Q8_0 — fastest on CPU & hybrid CPU+GPU, grammar-constrained JSON)
      2. `transformers` (HuggingFace repos — CUDA NF4 4-bit + SDPA / MPS fp16 / CPU dynamic int8 + tuned threads)
      3. `ollama` (`/api/chat` with connection pooling, `keep_alive`, constrained JSON schema & native tool calling)
    """

    _model_cache: Dict[str, Dict[str, Any]] = {}
    _cache_lock = threading.Lock()
    _max_cached_models: Optional[int] = None

    @classmethod
    def configure_cache(cls, max_models: Optional[int] = None) -> None:
        """Limit cached local models (useful on small-VRAM GPUs when swapping MoE experts)."""
        cls._max_cached_models = max(1, int(max_models)) if max_models is not None else None
        if cls._max_cached_models is not None:
            while len(cls._model_cache) > cls._max_cached_models:
                cls._evict_oldest_cached_model()

    @classmethod
    def _reclaim_memory(cls) -> None:
        """Aggressively reclaim Python RAM and all CUDA GPU cached/fragmented memory."""
        gc.collect()
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                for dev_idx in range(max(1, int(torch.cuda.device_count()))):
                    try:
                        with torch.cuda.device(dev_idx):
                            torch.cuda.empty_cache()
                            if hasattr(torch.cuda, "ipc_collect"):
                                torch.cuda.ipc_collect()
                    except Exception:
                        pass
        except Exception:
            pass

    @classmethod
    def _release_entry(cls, entry: Optional[Dict[str, Any]]) -> None:
        if isinstance(entry, dict):
            entry.pop("hf_model", None)
            entry.pop("llama_model", None)
            entry.pop("tokenizer", None)

    @classmethod
    def _evict_oldest_cached_model(cls) -> Optional[str]:
        evicted_key = None
        evicted_entry = None
        with cls._cache_lock:
            if cls._model_cache:
                evicted_key = next(iter(cls._model_cache))
                evicted_entry = cls._model_cache.pop(evicted_key, None)
        if evicted_key is not None:
            cls._release_entry(evicted_entry)
            cls._reclaim_memory()
        return evicted_key

    @classmethod
    def clear_cache(cls) -> None:
        """Unload all cached local models and release GPU/CPU memory."""
        with cls._cache_lock:
            entries = list(cls._model_cache.values())
            cls._model_cache.clear()
        for entry in entries:
            cls._release_entry(entry)
        cls._reclaim_memory()

    def __init__(
        self,
        model: Optional[str] = None,
        backend: str = "auto",              # "auto" | "llama_cpp" | "transformers" | "ollama"
        quantization: str = "auto",         # "auto" | "4bit" | "8bit" | "fp16" | "bf16" | "none"
        device: str = "auto",               # "auto" | "cuda" | "mps" | "cpu"
        gguf_filename: Optional[str] = None,
        n_ctx: int = 4096,
        n_gpu_layers: Optional[int] = None,
        n_threads: Optional[int] = None,
        temperature: float = 0.1,
        top_p: float = 0.9,
        base_url: str = "http://localhost:11434",
        keep_alive: str = "30m",
        trust_remote_code: bool = False,
        allow_auto_load: Optional[bool] = None,
    ):
        explicit_model = model is not None or gguf_filename is not None
        self.model = model or config.LOCAL_LLM_TEXT_MODEL or "Qwen/Qwen2.5-3B-Instruct"
        self.backend = (backend or "auto").lower()
        self.quantization = (quantization or "auto").lower()
        self.device = (device or "auto").lower()
        self.gguf_filename = gguf_filename
        self.n_ctx = max(512, int(n_ctx))
        self.n_gpu_layers = n_gpu_layers
        self.n_threads = n_threads
        self.temperature = max(0.0, float(temperature))
        self.top_p = float(top_p)
        self.base_url = base_url.rstrip("/")
        self.keep_alive = keep_alive
        self.trust_remote_code = bool(trust_remote_code)
        # Auto-allow loading when user explicitly instantiates LocalLLM(model=...)
        self._allow_auto_load = explicit_model if allow_auto_load is None else bool(allow_auto_load)

        self.active_backend: Optional[str] = None
        self.active_device: str = "cpu"
        self.hf_model = None
        self.tokenizer = None
        self.llama_model = None
        self.is_ollama = False
        self.client = None  # Signals initialized state cleanly to Agent.run()
        self._inference_lock: Optional[threading.Lock] = None
        self._http_client: Optional[httpx.AsyncClient] = None
        self._http_loop: Optional[asyncio.AbstractEventLoop] = None
        self._ollama_tools_unsupported: set = set()

    def _cache_key(self) -> str:
        return f"{self.backend}:{self.model}:{self.gguf_filename or ''}:{self.quantization}:{self.device}:{self.n_ctx}"

    def _customize_transformers_load_kwargs(
        self,
        load_kwargs: Dict[str, Any],
        hw: Dict[str, Any],
        *,
        device: str,
        has_accelerate: bool,
    ) -> None:
        """Subclass hook to customize model loading options (used by LocalMoELLM)."""
        return None

    def _detect_runtime_profile(self, use_cache: bool = True) -> Dict[str, Any]:
        gpu_info = HardwareChecker.check_gpu(use_cache=use_cache)
        cpu_info = HardwareChecker.check_cpu()
        detected_device = gpu_info.get("device", "cpu") if self.device == "auto" else self.device
        threads = self.n_threads or cpu_info.get("recommended_threads", max(1, (os.cpu_count() or 2) // 2))
        gpu_layers = self.n_gpu_layers
        if gpu_layers is None:
            gpu_layers = -1 if detected_device in ("cuda", "mps") else 0
        return {
            "device": detected_device,
            "has_gpu": detected_device in ("cuda", "mps"),
            "vram_gb": gpu_info.get("vram_gb", 0.0),
            "free_vram_gb": gpu_info.get("free_vram_gb", gpu_info.get("vram_gb", 0.0)),
            "total_vram_gb": gpu_info.get("total_vram_gb", gpu_info.get("vram_gb", 0.0)),
            "total_free_vram_gb": gpu_info.get("total_free_vram_gb", gpu_info.get("vram_gb", 0.0)),
            "gpu_count": gpu_info.get("gpu_count", 1 if detected_device in ("cuda", "mps") else 0),
            "gpus": gpu_info.get("gpus") or [],
            "bf16_supported": gpu_info.get("bf16_supported", False),
            "n_threads": threads,
            "n_gpu_layers": gpu_layers,
        }

    @staticmethod
    def _build_safe_max_memory(
        hw: Dict[str, Any],
        *,
        safety_ratio: float = 0.78,
        min_reserve_gb: float = 2.4,
        cpu_ratio: float = 0.65,
    ) -> Dict[Any, str]:
        """Compute per-GPU & CPU memory caps leaving headroom for weight conversion & KV cache."""
        max_mem: Dict[Any, str] = {}
        gpus = list(hw.get("gpus") or [])
        if not gpus and float(hw.get("vram_gb") or 0.0) > 0:
            gpus = [{
                "index": 0,
                "total_gb": float(hw.get("vram_gb") or 0.0),
                "free_gb": float(hw.get("free_vram_gb") or hw.get("vram_gb") or 0.0),
            }]
        for gpu in gpus:
            idx = int(gpu.get("index", 0))
            tot = float(gpu.get("total_gb") or 0.0)
            free = float(gpu.get("free_gb") or tot)
            if tot <= 0:
                continue
            reserve = min_reserve_gb if tot >= 8.0 else max(0.8, tot * 0.25)
            cap = max(1, int(min(max(1.0, free - reserve), tot * safety_ratio)))
            max_mem[idx] = f"{cap}GiB"

        avail_ram_gb = float(HardwareChecker.check_ram().get("available_gb", 8.0) or 8.0)
        cpu_cap = max(2, int(avail_ram_gb * cpu_ratio))
        max_mem["cpu"] = f"{cpu_cap}GiB"
        return max_mem

    def _resolve_backend(self) -> str:
        aliases = {
            "transformer": "transformers",
            "transformers": "transformers",
            "hf": "transformers",
            "huggingface": "transformers",
            "llama_cpp": "llama_cpp",
            "llamacpp": "llama_cpp",
            "llama-cpp": "llama_cpp",
            "gguf": "llama_cpp",
            "ollama": "ollama",
        }
        if self.backend in aliases:
            return aliases[self.backend]
        if self.gguf_filename or str(self.model).lower().endswith(".gguf"):
            return "llama_cpp"
        if ":" in self.model and "/" not in self.model and not os.path.exists(self.model):
            return "ollama"
        return "transformers"

    def _get_http_client(self) -> httpx.AsyncClient:
        loop = asyncio.get_running_loop()
        if self._http_client is None or self._http_client.is_closed or self._http_loop is not loop:
            self._http_client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(180.0, connect=10.0),
                limits=httpx.Limits(max_keepalive_connections=4, max_connections=10),
            )
            self._http_loop = loop
        return self._http_client

    def is_available(self) -> bool:
        return bool(self.model) and (self._allow_auto_load or config.LOAD_LOCAL_LLM)

    def _load_llama_cpp_sync(self, hw: Dict[str, Any]) -> Dict[str, Any]:
        import llama_cpp
        from llama_cpp import Llama

        actual_device = hw["device"]
        gpu_offload_fn = getattr(llama_cpp, "llama_supports_gpu_offload", None)
        supports_gpu = bool(gpu_offload_fn()) if callable(gpu_offload_fn) else True
        if hw["n_gpu_layers"] != 0 and not supports_gpu:
            actual_device = "cpu"
            logger.warning(
                "llama-cpp-python was compiled for CPU-only (no CUDA offload). Prompt evaluation will run on CPU. "
                "On Colab GPU, either use a HuggingFace Transformers model (`Qwen/Qwen2.5-3B-Instruct`, backend='transformers') "
                "with `pip install bitsandbytes accelerate`, or install a CUDA-enabled llama-cpp-python wheel."
            )

        common_kwargs = {
            "n_ctx": self.n_ctx,
            "n_batch": 512,
            "n_threads": hw["n_threads"],
            "n_gpu_layers": hw["n_gpu_layers"] if supports_gpu else 0,
            "verbose": False,
        }
        if os.path.isfile(self.model):
            llama = Llama(model_path=self.model, **common_kwargs)
        elif self.gguf_filename:
            llama = Llama.from_pretrained(
                repo_id=self.model,
                filename=self.gguf_filename,
                **common_kwargs,
            )
        else:
            raise ValueError(
                "For `llama_cpp` backend, pass either a local `.gguf` file path in `model` "
                "or a HuggingFace repo ID in `model` along with `gguf_filename` (e.g. '*Q4_K_M.gguf')."
            )
        return {
            "type": "llama_cpp",
            "llama_model": llama,
            "device": actual_device,
            "lock": threading.Lock(),
        }

    @staticmethod
    def _set_dtype_kwarg(load_kwargs: Dict[str, Any], from_pretrained_fn: Any, dtype_val: Any) -> None:
        """Pass `dtype` on transformers >=4.56 (avoiding deprecation warning) and `torch_dtype` on older releases."""
        use_new_dtype = False
        try:
            import transformers as _tf
            ver_parts = tuple(int(p) for p in str(getattr(_tf, "__version__", "4.0")).split(".")[:2] if p.isdigit())
            use_new_dtype = ver_parts >= (4, 56)
        except Exception:
            use_new_dtype = False
        if not use_new_dtype:
            try:
                sig = inspect.signature(from_pretrained_fn)
                if "dtype" in sig.parameters and "torch_dtype" not in sig.parameters:
                    use_new_dtype = True
            except (TypeError, ValueError):
                pass
        key = "dtype" if use_new_dtype else "torch_dtype"
        other_key = "torch_dtype" if key == "dtype" else "dtype"
        load_kwargs.pop(other_key, None)
        load_kwargs[key] = dtype_val

    @staticmethod
    def _is_oom_error(exc: Exception) -> bool:
        msg = str(exc or "").lower()
        return (
            "out of memory" in msg
            or "outofmemory" in type(exc).__name__.lower()
            or ("cuda" in msg and ("alloc" in msg or "memory" in msg))
            or "some modules are dispatched on the cpu or the disk" in msg
        )

    def _load_transformers_sync(self, hw: Dict[str, Any]) -> Dict[str, Any]:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        self._reclaim_memory()
        hw = self._detect_runtime_profile(use_cache=False) if hw.get("device") == "cuda" else hw

        device = hw["device"]
        if device == "cuda":
            try:
                if not torch.cuda.is_available():
                    device = "cpu"
                else:
                    _ = torch.zeros(1, device="cuda") + 1
            except Exception as cuda_err:
                logger.warning(f"CUDA unavailable or incompatible ({cuda_err}); safely falling back to CPU.")
                device = "cpu"

        if device == "cpu":
            try:
                torch.set_num_threads(hw["n_threads"])
            except Exception:
                pass

        try:
            import accelerate  # noqa: F401
            has_accelerate = True
        except ImportError:
            has_accelerate = False

        load_kwargs: Dict[str, Any] = {
            "trust_remote_code": self.trust_remote_code,
            "low_cpu_mem_usage": True,
        }

        quant_applied = "none"
        default_offload_dir = getattr(
            self,
            "offload_folder",
            os.path.join(os.path.expanduser("~"), ".cache", "phoenix", "hf_offload"),
        )
        if device == "cuda":
            compute_dtype = torch.bfloat16 if hw["bf16_supported"] else torch.float16
            self._set_dtype_kwarg(load_kwargs, AutoModelForCausalLM.from_pretrained, compute_dtype)
            if self.quantization in ("auto", "4bit", "8bit"):
                try:
                    import bitsandbytes  # noqa: F401
                    from transformers import BitsAndBytesConfig
                    if self.quantization == "8bit":
                        load_kwargs["quantization_config"] = BitsAndBytesConfig(
                            load_in_8bit=True,
                            llm_int8_enable_fp32_cpu_offload=True,
                        )
                        quant_applied = "bnb_8bit"
                    else:
                        load_kwargs["quantization_config"] = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_quant_type="nf4",
                            bnb_4bit_use_double_quant=True,
                            bnb_4bit_compute_dtype=compute_dtype,
                            llm_int8_enable_fp32_cpu_offload=True,
                        )
                        quant_applied = "bnb_nf4"
                except Exception as q_err:
                    logger.warning(f"bitsandbytes unavailable ({q_err}); loading in {compute_dtype} (tip: `pip install bitsandbytes accelerate` for 4-bit).")
                    quant_applied = str(compute_dtype)
            elif self.quantization == "bf16":
                self._set_dtype_kwarg(load_kwargs, AutoModelForCausalLM.from_pretrained, torch.bfloat16)
                quant_applied = "bf16"
            elif self.quantization == "fp16":
                self._set_dtype_kwarg(load_kwargs, AutoModelForCausalLM.from_pretrained, torch.float16)
                quant_applied = "fp16"
            else:
                quant_applied = str(compute_dtype)

            if has_accelerate:
                load_kwargs["device_map"] = "auto"
                load_kwargs["offload_state_dict"] = True
                # Reserve headroom on multi-GPU setups so tensor concatenation/conversion never OOMs
                if int(hw.get("gpu_count") or 1) > 1:
                    os.makedirs(default_offload_dir, exist_ok=True)
                    load_kwargs["offload_folder"] = default_offload_dir
                    load_kwargs["max_memory"] = self._build_safe_max_memory(
                        hw, safety_ratio=0.76, min_reserve_gb=2.6, cpu_ratio=0.65
                    )
            load_kwargs["attn_implementation"] = "sdpa"
        elif device == "mps":
            self._set_dtype_kwarg(load_kwargs, AutoModelForCausalLM.from_pretrained, torch.float16)
            if has_accelerate:
                load_kwargs["device_map"] = "mps"
            quant_applied = "fp16_mps"
        else:
            # Memory-safe CPU path: NEVER load in FP32 when bfloat16 is available (cuts RAM usage by 50%!)
            cpu_dtype = torch.bfloat16 if self.quantization in ("auto", "bf16", "4bit", "8bit") else torch.float32
            self._set_dtype_kwarg(load_kwargs, AutoModelForCausalLM.from_pretrained, cpu_dtype)
            quant_applied = "bf16_cpu" if cpu_dtype == torch.bfloat16 else "fp32_cpu"

        self._customize_transformers_load_kwargs(load_kwargs, hw, device=device, has_accelerate=has_accelerate)

        tokenizer = AutoTokenizer.from_pretrained(self.model, trust_remote_code=self.trust_remote_code)
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        def _invoke_from_pretrained(kwargs_dict: Dict[str, Any]):
            kw = dict(kwargs_dict)
            while True:
                try:
                    return AutoModelForCausalLM.from_pretrained(self.model, **kw)
                except TypeError as type_err:
                    err_s = str(type_err)
                    stripped = False
                    for opt_k in ("offload_buffers", "offload_state_dict", "low_cpu_mem_usage", "attn_implementation"):
                        if opt_k in kw and (opt_k in err_s or "unexpected keyword argument" in err_s):
                            kw.pop(opt_k, None)
                            stripped = True
                            break
                    if not stripped and "dtype" in kw and ("dtype" in err_s or "unexpected keyword argument" in err_s):
                        kw["torch_dtype"] = kw.pop("dtype")
                        stripped = True
                    if not stripped:
                        raise

        hf_model = None
        last_err: Optional[Exception] = None
        try:
            hf_model = _invoke_from_pretrained(load_kwargs)
        except Exception as load_err:
            last_err = load_err
            self._reclaim_memory()
            if not self._is_oom_error(load_err) and "attn_implementation" in load_kwargs:
                retry_kw = dict(load_kwargs)
                retry_kw.pop("attn_implementation", None)
                try:
                    hf_model = _invoke_from_pretrained(retry_kw)
                    load_kwargs = retry_kw
                    last_err = None
                except Exception as retry_err:
                    last_err = retry_err
                    self._reclaim_memory()

        if hf_model is None and last_err is not None and self._is_oom_error(last_err) and device == "cuda":
            # Stage 1 & Stage 2 progressive OOM recovery: evict other cached models, enforce strict GPU reserve,
            # spill overflow tensors to CPU RAM + NVMe offload_folder.
            LocalLLM.clear_cache()
            self._reclaim_memory()
            hw_live = self._detect_runtime_profile(use_cache=False)
            recovery_tiers = [
                (0.60, 3.4, 0.60),
                (0.42, 4.8, 0.50),
            ]
            for tier_idx, (s_ratio, rsv_gb, c_ratio) in enumerate(recovery_tiers, start=1):
                if not has_accelerate:
                    break
                logger.warning(
                    f"CUDA/RAM pressure loading '{self.model}' ({last_err}); retrying recovery tier {tier_idx} "
                    f"(strict max_memory headroom + CPU/disk offload)..."
                )
                os.makedirs(default_offload_dir, exist_ok=True)
                tier_kw = dict(load_kwargs)
                tier_kw.pop("attn_implementation", None)
                tier_kw["device_map"] = "auto"
                tier_kw["low_cpu_mem_usage"] = True
                tier_kw["offload_state_dict"] = True
                tier_kw["offload_buffers"] = True
                tier_kw["offload_folder"] = default_offload_dir
                tier_kw["max_memory"] = self._build_safe_max_memory(
                    hw_live, safety_ratio=s_ratio, min_reserve_gb=rsv_gb, cpu_ratio=c_ratio
                )
                qcfg = tier_kw.get("quantization_config")
                if qcfg is not None and hasattr(qcfg, "llm_int8_enable_fp32_cpu_offload"):
                    try:
                        qcfg.llm_int8_enable_fp32_cpu_offload = True
                    except Exception:
                        pass
                try:
                    hf_model = _invoke_from_pretrained(tier_kw)
                    load_kwargs = tier_kw
                    last_err = None
                    if hasattr(self, "_hybrid_offload_used"):
                        self._hybrid_offload_used = True
                    break
                except Exception as tier_err:
                    last_err = tier_err
                    self._reclaim_memory()

        if hf_model is None and last_err is not None:
            preset = getattr(self, "_moe_preset", {}) or {}
            gguf_repo = preset.get("gguf_repo")
            if gguf_repo and self._is_oom_error(last_err):
                try:
                    import importlib.util
                    if importlib.util.find_spec("llama_cpp") is not None:
                        logger.warning(
                            f"HuggingFace load exhausted GPU/RAM for '{self.model}'; switching automatically to compact Q4_K_M GGUF '{gguf_repo}'."
                        )
                        orig_model, orig_gguf = self.model, self.gguf_filename
                        self.model = gguf_repo
                        self.gguf_filename = preset.get("gguf_filename", "*Q4_K_M.gguf")
                        try:
                            return self._load_llama_cpp_sync(hw)
                        except Exception:
                            self.model, self.gguf_filename = orig_model, orig_gguf
                except Exception:
                    pass
            raise last_err

        if device in ("cuda", "mps") and not has_accelerate and "quantization_config" not in load_kwargs:
            hf_model = hf_model.to(device)

        hf_model.eval()
        self._reclaim_memory()

        # Only apply dynamic INT8 on CPU if user explicitly requested `8bit` AND there is plenty of free RAM
        # (because quantize_dynamic temporarily duplicates linear weights in RAM during conversion).
        ram_avail_gb = HardwareChecker.check_ram().get("available_gb", 0.0)
        if device == "cpu" and self.quantization == "8bit" and ram_avail_gb >= 10.0:
            try:
                hf_model = hf_model.to(torch.float32)
                hf_model = torch.ao.quantization.quantize_dynamic(
                    hf_model, {torch.nn.Linear}, dtype=torch.qint8
                )
                gc.collect()
                quant_applied = "cpu_dynamic_int8"
            except Exception as cpu_q_err:
                logger.warning(f"CPU dynamic int8 skipped ({cpu_q_err}); keeping {quant_applied}.")
        elif device == "cpu" and self.quantization in ("auto", "4bit"):
            logger.info(
                "Running Transformers on CPU in bfloat16. For 4-bit ~2GB RAM on CPU, use a GGUF model with `backend='llama_cpp'` or `backend='ollama'`."
            )

        dev_summary = ""
        dmap = getattr(hf_model, "hf_device_map", None)
        if isinstance(dmap, dict) and dmap:
            counts: Dict[str, int] = {}
            for loc in dmap.values():
                k = str(loc)
                counts[k] = counts.get(k, 0) + 1
            dev_summary = ", placements=" + json.dumps(counts, separators=(",", ":"))

        logger.info(
            f"Loaded LocalLLM ({self.model}) on {device} via transformers [quant={quant_applied}, threads={hw['n_threads']}{dev_summary}]"
        )
        return {
            "type": "transformers",
            "hf_model": hf_model,
            "tokenizer": tokenizer,
            "device": device,
            "quantization": quant_applied,
            "lock": threading.Lock(),
        }

    async def init(self):
        if not self.model:
            return
        if not self._allow_auto_load and not config.LOAD_LOCAL_LLM:
            logger.info(f"LocalLLM.init skipped for {self.model} because LOAD_LOCAL_LLM is False.")
            return

        cache_key = self._cache_key()
        hw = self._detect_runtime_profile()
        target_backend = self._resolve_backend()

        cached = LocalLLM._model_cache.get(cache_key)
        if cached is None:
            logger.info(f"Initializing LocalLLM '{self.model}' [backend={target_backend}, device={hw['device']}]...")
            if target_backend == "llama_cpp":
                try:
                    cached = await asyncio.to_thread(self._load_llama_cpp_sync, hw)
                except Exception as e:
                    if self.backend == "llama_cpp":
                        raise RuntimeError(f"Failed to load GGUF model via llama_cpp: {e}")
                    logger.warning(f"llama_cpp unavailable ({e}); falling back to Ollama.")
                    cached = {"type": "ollama", "device": hw["device"], "lock": threading.Lock()}
            elif target_backend == "transformers":
                final_err: Optional[Exception] = None
                try:
                    cached = await asyncio.to_thread(self._load_transformers_sync, hw)
                except Exception as e:
                    final_err = e
                    if self._is_oom_error(e) and LocalLLM._model_cache:
                        logger.warning("CUDA/memory pressure while loading model; clearing cached LocalLLMs and retrying...")
                        LocalLLM.clear_cache()
                        hw = self._detect_runtime_profile(use_cache=False)
                        try:
                            cached = await asyncio.to_thread(self._load_transformers_sync, hw)
                            final_err = None
                        except Exception as e2:
                            final_err = e2
                if cached is None:
                    err_str = str(final_err or "")
                    if (
                        self.backend == "transformers"
                        or "trust_remote_code" in err_str.lower()
                        or "/" in str(self.model)
                    ):
                        raise RuntimeError(f"Failed to load HuggingFace model '{self.model}': {final_err}")
                    logger.warning(f"Transformers local load failed ({final_err}); falling back to Ollama.")
                    cached = {"type": "ollama", "device": hw["device"], "lock": threading.Lock()}
            else:
                cached = {"type": "ollama", "device": hw["device"], "lock": threading.Lock()}

            with LocalLLM._cache_lock:
                if LocalLLM._max_cached_models is not None and len(LocalLLM._model_cache) >= LocalLLM._max_cached_models:
                    oldest = next(iter(LocalLLM._model_cache), None)
                    if oldest and oldest != cache_key:
                        LocalLLM._model_cache.pop(oldest, None)
                LocalLLM._model_cache[cache_key] = cached

        self.active_backend = cached["type"]
        self.active_device = cached.get("device", hw["device"])
        self._inference_lock = cached.get("lock") or threading.Lock()
        self.hf_model = cached.get("hf_model")
        self.tokenizer = cached.get("tokenizer")
        self.llama_model = cached.get("llama_model")
        self.is_ollama = self.active_backend == "ollama"
        self.client = True

        if self.is_ollama:
            try:
                client = self._get_http_client()
                resp = await client.get("/api/tags")
                if resp.status_code != 200:
                    logger.warning(f"Ollama responded with HTTP {resp.status_code} at {self.base_url}")
            except Exception:
                logger.warning(f"Could not connect to local Ollama at {self.base_url}. Ensure Ollama is running.")

    def _build_messages(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        history: Optional[List[Dict[str, Any]]] = None,
        context_prefix: str = "",
    ) -> List[Dict[str, str]]:
        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        for item in history or []:
            if isinstance(item, dict):
                if "role" in item and "content" in item:
                    messages.append({"role": str(item["role"]), "content": str(item["content"])})
                elif isinstance(item.get("content"), dict) and "role" in item["content"]:
                    messages.append(item["content"])
        messages.append({"role": "user", "content": f"{context_prefix}{prompt}"})
        return messages

    def _run_transformers_sync(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int,
        temperature: float,
    ) -> Tuple[str, Dict[str, int]]:
        import torch

        assert self._inference_lock is not None
        with self._inference_lock:
            if hasattr(self.tokenizer, "apply_chat_template"):
                try:
                    formatted = self.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                except Exception:
                    formatted = "\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nassistant:"
            else:
                formatted = "\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nassistant:"

            reserved_out = min(max_new_tokens, max(256, self.n_ctx // 3))
            inputs = self.tokenizer(
                formatted,
                return_tensors="pt",
                truncation=True,
                max_length=max(256, self.n_ctx - reserved_out),
            )
            model_device = getattr(self.hf_model, "device", torch.device("cpu"))
            if str(model_device) == "meta":
                model_device = next((p.device for p in self.hf_model.parameters() if p.device.type != "meta"), torch.device("cpu"))
            inputs = {k: v.to(model_device) for k, v in inputs.items()}

            gen_kwargs: Dict[str, Any] = {
                "max_new_tokens": max_new_tokens,
                "use_cache": True,
                "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            }
            if temperature <= 0.01:
                gen_kwargs["do_sample"] = False
            else:
                gen_kwargs["do_sample"] = True
                gen_kwargs["temperature"] = temperature
                gen_kwargs["top_p"] = self.top_p

            with torch.inference_mode():
                generated_ids = self.hf_model.generate(**inputs, **gen_kwargs)

            prompt_len = int(inputs["input_ids"].shape[1])
            new_tokens = generated_ids[0][prompt_len:]
            text = self.tokenizer.decode(
                new_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ).strip()
            comp_len = int(len(new_tokens))
            usage = {
                "prompt_tokens": prompt_len,
                "completion_tokens": comp_len,
                "total_tokens": prompt_len + comp_len,
            }
            del inputs, generated_ids, new_tokens
            return text, usage

    def _run_llama_cpp_sync(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
        json_schema: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Dict[str, int]]:
        assert self._inference_lock is not None
        with self._inference_lock:
            kwargs: Dict[str, Any] = {
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": self.top_p,
            }
            if json_schema is not None:
                # Use lightweight json_object mode in llama_cpp to avoid slow Pydantic $defs grammar compilation
                kwargs["response_format"] = {"type": "json_object"}

            resp = self.llama_model.create_chat_completion(**kwargs)
            content = (resp["choices"][0]["message"].get("content") or "").strip()
            raw_usage = resp.get("usage") or {}
            usage = {
                "prompt_tokens": int(raw_usage.get("prompt_tokens", 0) or 0),
                "completion_tokens": int(raw_usage.get("completion_tokens", 0) or 0),
                "total_tokens": int(raw_usage.get("total_tokens", 0) or 0),
            }
            return content, usage

    async def _run_ollama_chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
        json_schema: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, int]]:
        hw = self._detect_runtime_profile()
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {
                "num_ctx": self.n_ctx,
                "num_predict": max_tokens,
                "temperature": temperature,
                "top_p": self.top_p,
                "num_thread": hw["n_threads"],
            },
        }
        if json_schema is not None:
            payload["format"] = json_schema
        if tools:
            payload["tools"] = tools

        client = self._get_http_client()
        resp = await client.post("/api/chat", json=payload)
        resp.raise_for_status()
        data = resp.json()
        prompt_t = int(data.get("prompt_eval_count") or 0)
        comp_t = int(data.get("eval_count") or 0)
        usage = {
            "prompt_tokens": prompt_t,
            "completion_tokens": comp_t,
            "total_tokens": prompt_t + comp_t,
        }
        return data.get("message") or {}, usage

    async def _ensure_initialized(self) -> None:
        if self.active_backend is None:
            if not self._allow_auto_load and not config.LOAD_LOCAL_LLM:
                raise RuntimeError(
                    f"Attempted to use LocalLLM ({self.model}) but LOAD_LOCAL_LLM is False. "
                    "Pass `model=...` explicitly to LocalLLM(...) or initialize Phoenix AI with local=True."
                )
            await self.init()

    async def generate(
        self,
        prompt: str,
        session_id: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        await self._ensure_initialized()

        memory = None
        try:
            memory = container.get("memory")
        except KeyError:
            pass

        context_prefix = ""
        history = []
        if session_id and memory and hasattr(memory, "history") and hasattr(memory, "search_memory"):
            try:
                semantic_context = await memory.search_memory(session_id, prompt)
                if semantic_context:
                    context_prefix = f"Context from previous conversations:\n{semantic_context}\n\n"
                history = await memory.history.get(session_id)
            except Exception:
                pass

        messages = self._build_messages(
            prompt=prompt,
            system_prompt=system_prompt,
            history=history,
            context_prefix=context_prefix,
        )
        out_tokens = max_tokens or config.SECURITY_MAX_OUTPUT_LENGTH
        temp = self.temperature if temperature is None else float(temperature)

        span_id = tracer.start_span(
            "LocalLLM.generate",
            {"model": self.model, "backend": self.active_backend, "device": self.active_device},
        )
        try:
            if self.active_backend == "llama_cpp":
                text, usage = await asyncio.to_thread(self._run_llama_cpp_sync, messages, out_tokens, temp, None)
            elif self.active_backend == "transformers":
                text, usage = await asyncio.to_thread(self._run_transformers_sync, messages, out_tokens, temp)
            else:
                msg, usage = await self._run_ollama_chat(messages, out_tokens, temp)
                text = (msg.get("content") or "").strip()

            tracer.end_span(span_id, status="success", usage=usage)
        except Exception as e:
            tracer.end_span(span_id, status="error", error=str(e))
            raise RuntimeError(f"LocalLLM ({self.active_backend}) generate failed: {e}")

        if session_id and memory and hasattr(memory, "add_interaction"):
            try:
                await memory.add_interaction(session_id, prompt, text)
            except Exception:
                pass
        return text

    async def generate_structured(
        self,
        prompt: str,
        schema: Any,
        session_id: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> Any:
        """
        High-accuracy structured JSON generation for local models:
          - Uses native grammar-constrained JSON schema decoding on Ollama and llama.cpp.
          - Uses greedy decoding (temperature=0.0) + automatic JSON validation/repair on Transformers.
        """
        await self._ensure_initialized()

        schema_dict = (
            schema.model_json_schema()
            if hasattr(schema, "model_json_schema")
            else (schema.schema() if hasattr(schema, "schema") else {})
        )

        defs_map = (schema_dict.get("$defs") or schema_dict.get("definitions") or {}) if isinstance(schema_dict, dict) else {}

        def _minify_schema(node: Any, seen: Optional[set] = None) -> Any:
            seen = seen or set()
            if isinstance(node, dict):
                ref = node.get("$ref")
                if isinstance(ref, str) and ref.startswith("#/"):
                    ref_name = ref.split("/")[-1]
                    if ref_name in defs_map and ref_name not in seen:
                        resolved = _minify_schema(defs_map[ref_name], seen | {ref_name})
                        merged = dict(resolved) if isinstance(resolved, dict) else {}
                        for k, v in node.items():
                            if k not in ("$ref", "title", "description", "additionalProperties"):
                                merged[k] = _minify_schema(v, seen)
                        return merged
                return {
                    k: _minify_schema(v, seen)
                    for k, v in node.items()
                    if k not in ("title", "description", "additionalProperties", "$defs", "definitions")
                }
            if isinstance(node, list):
                return [_minify_schema(i, seen) for i in node]
            return node

        prompt_schema = _minify_schema(schema_dict) if (self.active_backend == "transformers" and isinstance(schema_dict, dict)) else schema_dict
        schema_str = json.dumps(prompt_schema, separators=(",", ":")) if isinstance(prompt_schema, dict) else str(schema)
        system_prompt = (
            "You are a strict JSON generator. Return ONLY a single compact single-line raw JSON object matching this schema "
            "(keep any 'thought_process' <= 15 words, no markdown, no indentation whitespace, no explanation):\n"
            f"{schema_str}"
        )
        messages = self._build_messages(prompt=prompt, system_prompt=system_prompt)
        out_tokens = max_tokens or config.SECURITY_MAX_OUTPUT_LENGTH
        if self.active_backend == "transformers":
            out_tokens = min(out_tokens, max(512, self.n_ctx // 3))

        async def _call_backend(msgs: List[Dict[str, str]], tok_limit: int) -> Tuple[str, Dict[str, int]]:
            if self.active_backend == "llama_cpp":
                return await asyncio.to_thread(
                    self._run_llama_cpp_sync, msgs, tok_limit, 0.0, schema_dict or None
                )
            if self.active_backend == "ollama":
                msg_out, u = await self._run_ollama_chat(
                    msgs, tok_limit, 0.0, json_schema=schema_dict or "json"
                )
                return (msg_out.get("content") or "").strip(), u
            return await asyncio.to_thread(self._run_transformers_sync, msgs, tok_limit, 0.0)

        span_id = tracer.start_span(
            "LocalLLM.generate_structured",
            {"model": self.model, "backend": self.active_backend, "device": self.active_device},
        )
        try:
            raw, usage = await _call_backend(messages, out_tokens)
            parsed = _extract_json_dict(raw)
            if not parsed:
                repair_messages = self._build_messages(
                    prompt=f"Fix this broken output into valid raw JSON matching the schema.\nBroken output:\n{raw[:1400]}",
                    system_prompt=system_prompt,
                )
                raw, _ = await _call_backend(repair_messages, min(out_tokens, 1024))
                parsed = _extract_json_dict(raw)

            if not parsed:
                raw_text = (raw or "").strip()
                looks_like_failed_json = any(
                    tok in raw_text
                    for tok in ('"tools_to_call"', '"tool_name"', "{", "file_write", "file_edit", "file_read")
                )
                if not looks_like_failed_json and raw_text and getattr(schema, "__name__", "") == "TaskExecutionSchema":
                    tracer.end_span(span_id, status="success", usage=usage)
                    return schema(
                        thought_process="",
                        tools_to_call=[],
                        direct_response=raw_text,
                        is_complete=True,
                    )
                raise ValueError(f"Invalid or empty JSON returned by {self.model}: {raw_text[:200]}")

            try:
                validated = schema.model_validate(parsed) if hasattr(schema, "model_validate") else schema(**parsed)
            except Exception as val_err:
                repair_messages = self._build_messages(
                    prompt=(
                        f"Your JSON failed schema validation: {str(val_err)[:400]}\n"
                        f"Fix ONLY the invalid fields and return a valid JSON object matching the schema:\n{json.dumps(parsed)[:1400]}"
                    ),
                    system_prompt=system_prompt,
                )
                raw, _ = await _call_backend(repair_messages, min(out_tokens, 1024))
                repaired = _extract_json_dict(raw)
                if not repaired:
                    raise ValueError(f"Schema validation repair failed for {self.model}: {val_err}") from val_err
                validated = schema.model_validate(repaired) if hasattr(schema, "model_validate") else schema(**repaired)

            tracer.end_span(span_id, status="success", usage=usage)
            return validated
        except ValueError as ve:
            tracer.end_span(span_id, status="error", error=str(ve))
            raise
        except Exception as e:
            tracer.end_span(span_id, status="error", error=str(e))
            raise RuntimeError(f"LocalLLM structured generation failed: {e}")

    async def generate_with_tools(
        self,
        prompt: str,
        tools: List[Dict[str, Any]],
        session_id: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> Any:
        """
        Tool calling optimized for local LLMs:
          - Ollama: uses native `/api/chat` `tools` parameter with automatic fallback.
          - llama_cpp & transformers: uses grammar-constrained / deterministic `TaskExecutionSchema` JSON
            with compact tool schema injection so every HuggingFace/GGUF model works reliably.
        """
        from phoenix.framework.agent.cognition.schema import TaskExecutionSchema, ToolCall

        await self._ensure_initialized()

        if self.active_backend == "ollama" and tools and self.model not in self._ollama_tools_unsupported:
            span_id = tracer.start_span(
                "LocalLLM.generate_with_tools",
                {"model": self.model, "backend": "ollama", "device": self.active_device},
            )
            try:
                messages = self._build_messages(prompt=prompt)
                msg, usage = await self._run_ollama_chat(
                    messages,
                    max_tokens=max_tokens or config.SECURITY_MAX_OUTPUT_LENGTH,
                    temperature=0.0,
                    tools=tools,
                )
                tracer.end_span(span_id, status="success", usage=usage)

                parsed_calls: List[ToolCall] = []
                for tc in msg.get("tool_calls") or []:
                    fn = tc.get("function", {}) if isinstance(tc, dict) else {}
                    t_name = fn.get("name", "")
                    t_args = fn.get("arguments", {})
                    if isinstance(t_args, str):
                        t_args = _extract_json_dict(t_args)
                    if t_name:
                        parsed_calls.append(ToolCall(tool_name=t_name, arguments=t_args if isinstance(t_args, dict) else {}))

                text = (msg.get("content") or "").strip()
                if parsed_calls:
                    return TaskExecutionSchema(
                        thought_process=text or "Executing local tool calls.",
                        tools_to_call=parsed_calls,
                        is_complete=False,
                    )
                return TaskExecutionSchema(
                    thought_process=text,
                    tools_to_call=[],
                    direct_response=text,
                    is_complete=True,
                )
            except Exception as e:
                tracer.end_span(span_id, status="error", error=str(e))
                self._ollama_tools_unsupported.add(self.model)

        # Reliable structured JSON path for HuggingFace Transformers, llama.cpp GGUF, or Ollama fallback
        tools_str = _compact_tools_for_prompt(tools)
        enriched_prompt = (
            f"{prompt}\n\n"
            f"Available Tools (use exact names & parameter types):\n{tools_str}\n"
            f"Populate `tools_to_call` to invoke tools, or set `direct_response` and `is_complete: true` if answered."
        )
        return await self.generate_structured(
            enriched_prompt,
            TaskExecutionSchema,
            session_id=session_id,
            max_tokens=max_tokens,
        )

    async def generate_stream(
        self,
        prompt: str,
        session_id: Optional[str] = None,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """True token-by-token async streaming across Ollama, llama_cpp, and Transformers."""
        await self._ensure_initialized()
        messages = self._build_messages(prompt=prompt, system_prompt=system_prompt)
        out_tokens = max_tokens or config.SECURITY_MAX_OUTPUT_LENGTH

        if self.active_backend == "ollama":
            hw = self._detect_runtime_profile()
            payload = {
                "model": self.model,
                "messages": messages,
                "stream": True,
                "keep_alive": self.keep_alive,
                "options": {
                    "num_ctx": self.n_ctx,
                    "num_predict": out_tokens,
                    "temperature": self.temperature,
                    "num_thread": hw["n_threads"],
                },
            }
            client = self._get_http_client()
            async with client.stream("POST", "/api/chat", json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                        delta = (chunk.get("message") or {}).get("content", "")
                        if delta:
                            yield delta
                    except Exception:
                        continue
            return

        if self.active_backend == "llama_cpp":
            loop = asyncio.get_running_loop()
            queue: asyncio.Queue[Optional[str]] = asyncio.Queue()

            def _worker():
                assert self._inference_lock is not None
                with self._inference_lock:
                    try:
                        stream = self.llama_model.create_chat_completion(
                            messages=messages,
                            max_tokens=out_tokens,
                            temperature=self.temperature,
                            stream=True,
                        )
                        for part in stream:
                            delta = (part["choices"][0].get("delta") or {}).get("content", "")
                            if delta:
                                loop.call_soon_threadsafe(queue.put_nowait, delta)
                    finally:
                        loop.call_soon_threadsafe(queue.put_nowait, None)

            thread_task = asyncio.create_task(asyncio.to_thread(_worker))
            try:
                while True:
                    token = await queue.get()
                    if token is None:
                        break
                    yield token
            finally:
                await thread_task
            return

        # Transformers TextIteratorStreamer
        try:
            import torch
            from transformers import TextIteratorStreamer

            loop = asyncio.get_running_loop()
            queue = asyncio.Queue()
            streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

            def _hf_worker():
                assert self._inference_lock is not None
                with self._inference_lock:
                    try:
                        formatted = (
                            self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                            if hasattr(self.tokenizer, "apply_chat_template")
                            else "\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nassistant:"
                        )
                        inputs = self.tokenizer(formatted, return_tensors="pt", truncation=True, max_length=self.n_ctx)
                        model_device = getattr(self.hf_model, "device", torch.device("cpu"))
                        inputs = {k: v.to(model_device) for k, v in inputs.items()}

                        gen_kwargs: Dict[str, Any] = {
                            "max_new_tokens": out_tokens,
                            "use_cache": True,
                            "streamer": streamer,
                            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                            "do_sample": self.temperature > 0.01,
                        }
                        if self.temperature > 0.01:
                            gen_kwargs["temperature"] = self.temperature
                            gen_kwargs["top_p"] = self.top_p

                        def _run_gen():
                            with torch.inference_mode():
                                self.hf_model.generate(**inputs, **gen_kwargs)

                        t = threading.Thread(target=_run_gen, daemon=True)
                        t.start()
                        for piece in streamer:
                            if piece:
                                loop.call_soon_threadsafe(queue.put_nowait, piece)
                        t.join()
                    finally:
                        loop.call_soon_threadsafe(queue.put_nowait, None)

            worker_task = asyncio.create_task(asyncio.to_thread(_hf_worker))
            try:
                while True:
                    piece = await queue.get()
                    if piece is None:
                        break
                    yield piece
            finally:
                await worker_task
        except Exception:
            text = await self.generate(prompt, session_id=session_id, max_tokens=out_tokens, system_prompt=system_prompt)
            yield text

    async def close(self) -> None:
        """Close pooled HTTP connections cleanly."""
        if self._http_client is not None and not self._http_client.is_closed:
            await self._http_client.aclose()
            self._http_client = None
