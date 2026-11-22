"""Native Mixture-of-Experts (MoE) Local LLM wrapper for CPU & GPU execution.

Enables drop-in loading and execution of real pre-trained MoE models (e.g.
Qwen1.5-MoE-A2.7B-Chat, OLMoE-1B-7B, Mixtral-8x7B, DeepSeek-MoE, or custom GGUF MoEs)
with automatic CPU/GPU quantization, hybrid RAM+VRAM expert offloading, and zero retraining.
"""

from __future__ import annotations

import importlib.util
import os
from typing import Any, Dict, Optional, Tuple, Union

from phoenix.core.hardware_check import HardwareChecker
from phoenix.services.llm.local import LocalLLM
from phoenix.services.observability.logger import get_logger

logger = get_logger("Phoenix AI.LLM.LocalMoE")


MOE_MODEL_PRESETS: Dict[str, Dict[str, Any]] = {
    "Qwen/Qwen1.5-MoE-A2.7B-Chat": {
        "aliases": ("qwen-moe", "qwen1.5-moe", "qwen1.5-moe-a2.7b", "qwen/qwen1.5-moe-a2.7b"),
        "hf_repo": "Qwen/Qwen1.5-MoE-A2.7B-Chat",
        "gguf_repo": "tensorblock/Qwen1.5-MoE-A2.7B-Chat-GGUF",
        "gguf_filename": "*Q4_K_M.gguf",
        "total_params": "14.3B",
        "active_params": "2.7B",
        "num_experts": 60,
        "experts_per_tok": 4,
        "vram_4bit_gb": 9.2,
        "vram_peak_gb": 16.0,
        "ram_gguf_q4_gb": 8.5,
        "trust_remote_code": False,
    },
    "allenai/OLMoE-1B-7B-0924-Instruct": {
        "aliases": ("olmoe", "olmoe-1b-7b", "allenai/olmoe"),
        "hf_repo": "allenai/OLMoE-1B-7B-0924-Instruct",
        "gguf_repo": "allenai/OLMoE-1B-7B-0924-Instruct-GGUF",
        "gguf_filename": "*Q4_K_M.gguf",
        "total_params": "6.9B",
        "active_params": "1.3B",
        "num_experts": 64,
        "experts_per_tok": 8,
        "vram_4bit_gb": 4.6,
        "vram_peak_gb": 6.8,
        "ram_gguf_q4_gb": 4.4,
        "trust_remote_code": False,
    },
    "ibm-granite/granite-3.1-1b-a400m-instruct": {
        "aliases": ("granite-moe-1b", "granite-1b-moe", "granite-400m"),
        "hf_repo": "ibm-granite/granite-3.1-1b-a400m-instruct",
        "total_params": "1.3B",
        "active_params": "0.4B",
        "vram_fp16_gb": 2.7,
        "vram_4bit_gb": 1.2,
        "vram_peak_gb": 2.7,
        "trust_remote_code": False,
    },
    "ibm-granite/granite-3.1-3b-a800m-instruct": {
        "aliases": ("granite-moe", "granite-moe-3b", "granite-3b-moe", "granite-800m"),
        "hf_repo": "ibm-granite/granite-3.1-3b-a800m-instruct",
        "total_params": "3.3B",
        "active_params": "0.8B",
        "vram_fp16_gb": 6.6,
        "vram_4bit_gb": 2.4,
        "vram_peak_gb": 6.6,
        "trust_remote_code": False,
    },
    "mistralai/Mixtral-8x7B-Instruct-v0.1": {
        "aliases": ("mixtral", "mixtral-8x7b"),
        "hf_repo": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "total_params": "46.7B",
        "active_params": "12.9B",
        "num_experts": 8,
        "experts_per_tok": 2,
        "vram_4bit_gb": 26.0,
        "vram_peak_gb": 30.0,
        "ram_gguf_q4_gb": 26.5,
        "trust_remote_code": False,
    },
    "deepseek-ai/deepseek-moe-16b-chat": {
        "aliases": ("deepseek-moe", "deepseek-moe-16b"),
        "hf_repo": "deepseek-ai/deepseek-moe-16b-chat",
        "total_params": "16.4B",
        "active_params": "2.8B",
        "num_experts": 64,
        "experts_per_tok": 6,
        "vram_4bit_gb": 10.5,
        "vram_peak_gb": 17.5,
        "ram_gguf_q4_gb": 10.0,
        "trust_remote_code": True,
    },
}


def _resolve_moe_preset(model_name: Optional[str]) -> Tuple[str, Dict[str, Any]]:
    raw = (model_name or "").strip()
    if not raw:
        canonical = "Qwen/Qwen1.5-MoE-A2.7B-Chat"
        return canonical, dict(MOE_MODEL_PRESETS[canonical])

    if raw in MOE_MODEL_PRESETS:
        return raw, dict(MOE_MODEL_PRESETS[raw])

    low = raw.lower()
    for canonical, meta in MOE_MODEL_PRESETS.items():
        if canonical.lower() == low or low in meta.get("aliases", ()):
            return canonical, dict(meta)

    return raw, {}


class LocalMoELLM(LocalLLM):
    """
    Drop-in Local Mixture-of-Experts (MoE) LLM supporting native MoE architectures
    (Qwen1.5-MoE-A2.7B, OLMoE-1B-7B, Mixtral-8x7B, DeepSeek-MoE, custom HuggingFace/GGUF MoEs)
    across GPU (4-bit NF4 + seamless CPU expert offload) and CPU (auto GGUF Q4_K_M / disk offload).
    """

    def __init__(
        self,
        model: Optional[str] = "Qwen/Qwen1.5-MoE-A2.7B-Chat",
        backend: str = "auto",
        quantization: str = "auto",
        device: str = "auto",
        gguf_filename: Optional[str] = None,
        n_ctx: int = 4096,
        n_gpu_layers: Optional[int] = None,
        n_threads: Optional[int] = None,
        temperature: float = 0.1,
        top_p: float = 0.9,
        base_url: str = "http://localhost:11434",
        keep_alive: str = "30m",
        trust_remote_code: Optional[bool] = None,
        allow_auto_load: Optional[bool] = None,
        prefer_gguf_on_cpu: bool = True,
        cpu_offload: Union[bool, str] = "auto",
        offload_folder: Optional[str] = None,
    ):
        resolved_model, preset = _resolve_moe_preset(model)
        self._moe_preset: Dict[str, Any] = preset
        self.requested_model: str = model or resolved_model
        self.prefer_gguf_on_cpu: bool = bool(prefer_gguf_on_cpu)
        self.cpu_offload: Union[bool, str] = cpu_offload
        self.offload_folder: str = offload_folder or os.path.join(
            os.path.expanduser("~"), ".cache", "phoenix", "moe_offload"
        )
        self._hybrid_offload_used: bool = False

        effective_trust = (
            bool(trust_remote_code)
            if trust_remote_code is not None
            else bool(preset.get("trust_remote_code", False))
        )

        # Pick optimal local backend & artifact before initializing base state
        chosen_model = resolved_model
        chosen_backend = (backend or "auto").lower()
        chosen_gguf = gguf_filename
        chosen_gpu_layers = n_gpu_layers

        gpu_info = HardwareChecker.check_gpu()
        detected_device = gpu_info.get("device", "cpu") if (device or "auto").lower() == "auto" else device.lower()
        has_llama_cpp = importlib.util.find_spec("llama_cpp") is not None

        if chosen_backend == "auto" and not chosen_gguf and not str(chosen_model).lower().endswith(".gguf"):
            if detected_device == "cpu" and self.prefer_gguf_on_cpu and preset.get("gguf_repo"):
                if has_llama_cpp:
                    chosen_model = preset["gguf_repo"]
                    chosen_gguf = preset.get("gguf_filename", "*Q4_K_M.gguf")
                    chosen_backend = "llama_cpp"
                    logger.info(
                        f"LocalMoELLM CPU mode: using quantized GGUF '{chosen_model}' ({chosen_gguf}) "
                        "for memory-efficient MoE inference."
                    )
                else:
                    logger.info(
                        f"LocalMoELLM CPU tip: '{resolved_model}' is a sparse MoE ({preset.get('total_params', '?')} total / "
                        f"{preset.get('active_params', '?')} active). Install `llama-cpp-python` to auto-run Q4_K_M GGUF with ~3x less RAM."
                    )

        # For small sparse MoEs (<= 4B total) whose full FP16 fits comfortably inside half of GPU VRAM,
        # native FP16/BF16 avoids per-expert 4-bit NF4 dequantization overhead and runs ~3-5x faster.
        chosen_quant = (quantization or "auto").lower()
        if chosen_quant == "auto" and detected_device == "cuda" and chosen_backend != "llama_cpp":
            vram_gb = float(gpu_info.get("vram_gb") or 0.0)
            est_fp16 = float(preset.get("vram_fp16_gb") or 0.0)
            if 0.0 < est_fp16 <= vram_gb * 0.50:
                chosen_quant = "bf16" if gpu_info.get("bf16_supported") else "fp16"

        # Adjust GGUF GPU layers conservatively if VRAM is tighter than MoE 4-bit footprint
        if detected_device == "cuda" and chosen_gpu_layers is None and preset.get("vram_4bit_gb"):
            vram_gb = float(gpu_info.get("vram_gb") or 0.0)
            est_vram = float(preset["vram_4bit_gb"])
            if 0.0 < vram_gb < est_vram and (chosen_backend == "llama_cpp" or chosen_gguf):
                # Hybrid GPU + CPU layer split so MoE runs without CUDA OOM
                ratio = max(0.25, min(0.85, (vram_gb * 0.8) / est_vram))
                chosen_gpu_layers = max(8, int(28 * ratio))

        super().__init__(
            model=chosen_model,
            backend=chosen_backend,
            quantization=chosen_quant,
            device=device,
            gguf_filename=chosen_gguf,
            n_ctx=n_ctx,
            n_gpu_layers=chosen_gpu_layers,
            n_threads=n_threads,
            temperature=temperature,
            top_p=top_p,
            base_url=base_url,
            keep_alive=keep_alive,
            trust_remote_code=effective_trust,
            allow_auto_load=allow_auto_load,
        )

    def _customize_transformers_load_kwargs(
        self,
        load_kwargs: Dict[str, Any],
        hw: Dict[str, Any],
        *,
        device: str,
        has_accelerate: bool,
    ) -> None:
        """Apply MoE-safe GPU+CPU+disk expert offloading with multi-GPU scratch headroom."""
        if not self.cpu_offload:
            return

        ram_info = HardwareChecker.check_ram()
        avail_ram_gb = max(2, int(float(ram_info.get("available_gb", 8.0)) * 0.65))
        vram_gb = float(hw.get("vram_gb") or 0.0)
        gpu_count = int(hw.get("gpu_count") or (1 if vram_gb > 0 else 0))
        est_needed = float(self._moe_preset.get("vram_4bit_gb") or 0.0)
        est_peak = float(self._moe_preset.get("vram_peak_gb") or est_needed or 0.0)
        force_offload = self.cpu_offload is True

        need_gpu_offload = (
            device == "cuda"
            and vram_gb > 0
            and (
                force_offload
                or gpu_count > 1
                or (est_peak > 0 and vram_gb < est_peak * 1.15)
                or (est_peak == 0.0 and vram_gb < 10.0)
            )
        )

        if need_gpu_offload:
            qcfg = load_kwargs.get("quantization_config")
            if qcfg is not None and hasattr(qcfg, "llm_int8_enable_fp32_cpu_offload"):
                try:
                    qcfg.llm_int8_enable_fp32_cpu_offload = True
                except Exception:
                    pass

        if not has_accelerate:
            return

        if need_gpu_offload:
            os.makedirs(self.offload_folder, exist_ok=True)
            load_kwargs["device_map"] = "auto"
            load_kwargs["offload_folder"] = self.offload_folder
            load_kwargs["offload_state_dict"] = True
            load_kwargs["offload_buffers"] = True
            load_kwargs["max_memory"] = self._build_safe_max_memory(
                hw, safety_ratio=0.72, min_reserve_gb=3.0, cpu_ratio=0.65
            )
            self._hybrid_offload_used = True
        elif device == "cpu":
            # Prevent large MoE CPU tensors from crashing system RAM
            os.makedirs(self.offload_folder, exist_ok=True)
            load_kwargs["device_map"] = "auto"
            load_kwargs["offload_folder"] = self.offload_folder
            load_kwargs["offload_state_dict"] = True
            load_kwargs["max_memory"] = {"cpu": f"{avail_ram_gb}GiB"}
            self._hybrid_offload_used = True

    def get_moe_info(self) -> Dict[str, Any]:
        """Return resolved MoE architecture specs, backend, device, and expert routing configuration."""
        cfg = getattr(self.hf_model, "config", None) if self.hf_model is not None else None
        archs = list(getattr(cfg, "architectures", []) or []) if cfg is not None else []

        num_experts = (
            getattr(cfg, "num_experts", None)
            or getattr(cfg, "num_local_experts", None)
            or getattr(cfg, "n_routed_experts", None)
            or self._moe_preset.get("num_experts")
        )
        active_per_tok = (
            getattr(cfg, "num_experts_per_tok", None)
            or self._moe_preset.get("experts_per_tok")
        )
        return {
            "requested_model": self.requested_model,
            "model": self.model,
            "gguf_filename": self.gguf_filename,
            "backend": self.active_backend or self._resolve_backend(),
            "device": self.active_device if self.active_backend else self.device,
            "architecture": archs[0] if archs else None,
            "num_experts": num_experts,
            "experts_per_token": active_per_tok,
            "total_params": self._moe_preset.get("total_params"),
            "active_params": self._moe_preset.get("active_params"),
            "hybrid_offload": self._hybrid_offload_used,
            "initialized": self.client is not None,
        }


def local_moe_llm(
    model: Optional[str] = "Qwen/Qwen1.5-MoE-A2.7B-Chat",
    **kwargs: Any,
) -> LocalMoELLM:
    """Create a LocalMoELLM instance configured automatically for local CPU or GPU MoE inference."""
    return LocalMoELLM(model=model, **kwargs)
