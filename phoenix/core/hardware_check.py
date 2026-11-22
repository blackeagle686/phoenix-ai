import os
import platform
import shutil
import subprocess
import sys
from typing import Any, Dict, Optional, Tuple

from phoenix.services.observability.logger import get_logger

try:
    import psutil
except ImportError:
    psutil = None

logger = get_logger("Phoenix AI.Hardware")


class HardwareChecker:
    """
    Inspects CPU, RAM, disk, and GPU (CUDA / MPS) resources and recommends
    optimal local inference settings (backend, quantization, threads).
    """

    MIN_RAM_GB = 8
    RECOMMENDED_RAM_GB = 16
    MIN_DISK_GB = 10
    _gpu_cache: Optional[Dict[str, Any]] = None

    @classmethod
    def check_all(cls, silent: bool = False) -> Tuple[bool, Dict[str, Any]]:
        results = {
            "ram": cls.check_ram(),
            "cpu": cls.check_cpu(),
            "disk": cls.check_disk(),
            "os": cls.check_os(),
            "gpu": cls.check_gpu(),
        }
        results["recommendation"] = cls.recommend_local_profile(results)
        all_ok = results["ram"]["ok"] and results["disk"]["ok"]
        if not silent:
            cls.print_report(results)
        return all_ok, results

    @classmethod
    def check_cpu(cls) -> Dict[str, Any]:
        logical = os.cpu_count() or 2
        physical = psutil.cpu_count(logical=False) if psutil else max(1, logical // 2)
        physical = physical or max(1, logical // 2)
        return {
            "physical_cores": physical,
            "logical_cores": logical,
            "recommended_threads": max(1, min(physical, 16)),
            "ok": True,
        }

    @classmethod
    def check_ram(cls) -> Dict[str, Any]:
        if psutil is not None:
            total_ram = psutil.virtual_memory().total / (1024 ** 3)
            avail_ram = psutil.virtual_memory().available / (1024 ** 3)
        else:
            total_ram = 16.0
            avail_ram = 8.0
        return {
            "value": f"{total_ram:.2f} GB",
            "total_gb": round(total_ram, 2),
            "available_gb": round(avail_ram, 2),
            "ok": total_ram >= cls.MIN_RAM_GB,
            "recommended": total_ram >= cls.RECOMMENDED_RAM_GB,
            "message": "Insufficient RAM for stable local LLM execution." if total_ram < cls.MIN_RAM_GB else "RAM is sufficient.",
        }

    @classmethod
    def check_disk(cls) -> Dict[str, Any]:
        _, _, free = shutil.disk_usage(".")
        free_gb = free / (1024 ** 3)
        return {
            "value": f"{free_gb:.2f} GB Free",
            "free_gb": round(free_gb, 2),
            "ok": free_gb >= cls.MIN_DISK_GB,
            "message": "Low disk space. Model downloads might fail." if free_gb < cls.MIN_DISK_GB else "Disk space is sufficient.",
        }

    @classmethod
    def check_os(cls) -> Dict[str, Any]:
        return {
            "value": f"{platform.system()} {platform.release()}",
            "ok": True,
        }

    @classmethod
    def check_gpu(cls, use_cache: bool = True) -> Dict[str, Any]:
        if use_cache and cls._gpu_cache is not None:
            cached = dict(cls._gpu_cache)
            if "gpus" in cached and isinstance(cached["gpus"], list):
                cached["gpus"] = [dict(g) for g in cached["gpus"]]
            return cached

        has_cuda = False
        has_mps = False
        vram_gb = 0.0
        free_vram_gb = 0.0
        total_vram_gb = 0.0
        total_free_vram_gb = 0.0
        gpu_name = ""
        gpu_count = 0
        gpus = []
        bf16_supported = False

        # Avoid expensive `import torch` unless torch is already imported in this process
        torch_mod = sys.modules.get("torch")
        if torch_mod is not None:
            try:
                if torch_mod.cuda.is_available():
                    has_cuda = True
                    dev_count = max(1, int(torch_mod.cuda.device_count()))
                    for dev_idx in range(dev_count):
                        props = torch_mod.cuda.get_device_properties(dev_idx)
                        tot_gb = round(props.total_memory / (1024 ** 3), 2)
                        free_gb = tot_gb
                        try:
                            free_bytes, _ = torch_mod.cuda.mem_get_info(dev_idx)
                            free_gb = round(free_bytes / (1024 ** 3), 2)
                        except Exception:
                            pass
                        gpus.append({
                            "index": dev_idx,
                            "name": props.name,
                            "total_gb": tot_gb,
                            "free_gb": free_gb,
                        })
                    gpu_count = len(gpus)
                    gpu_name = gpus[0]["name"]
                    vram_gb = gpus[0]["total_gb"]
                    free_vram_gb = gpus[0]["free_gb"]
                    total_vram_gb = round(sum(g["total_gb"] for g in gpus), 2)
                    total_free_vram_gb = round(sum(g["free_gb"] for g in gpus), 2)
                    # Require native hardware BF16 (Ampere sm_80+); avoid slow software-emulated BF16 on Turing/T4 GPUs
                    try:
                        native_bf16_cc = all(
                            int(getattr(torch_mod.cuda.get_device_properties(i), "major", 0) or 0) >= 8
                            for i in range(dev_count)
                        )
                    except Exception:
                        native_bf16_cc = False
                    bf16_fn = getattr(torch_mod.cuda, "is_bf16_supported", None)
                    if callable(bf16_fn) and native_bf16_cc:
                        try:
                            bf16_supported = bool(bf16_fn(including_emulation=False))
                        except TypeError:
                            bf16_supported = bool(bf16_fn())
                    else:
                        bf16_supported = False
                elif hasattr(torch_mod.backends, "mps") and torch_mod.backends.mps.is_available():
                    has_mps = True
                    gpu_name = "Apple Silicon (MPS)"
                    gpu_count = 1
            except Exception:
                pass

        if not has_cuda and not has_mps and shutil.which("nvidia-smi"):
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader,nounits"],
                    capture_output=True,
                    text=True,
                    timeout=2,
                    check=True,
                )
                lines = [ln.strip() for ln in (out.stdout or "").splitlines() if ln.strip()]
                for dev_idx, line in enumerate(lines):
                    parts = [p.strip() for p in line.split(",")]
                    if parts:
                        has_cuda = True
                        tot_gb = round(int(parts[1]) / 1024.0, 2) if len(parts) > 1 and parts[1].isdigit() else 0.0
                        free_gb = round(int(parts[2]) / 1024.0, 2) if len(parts) > 2 and parts[2].isdigit() else tot_gb
                        gpus.append({
                            "index": dev_idx,
                            "name": parts[0],
                            "total_gb": tot_gb,
                            "free_gb": free_gb,
                        })
                if gpus:
                    gpu_count = len(gpus)
                    gpu_name = gpus[0]["name"]
                    vram_gb = gpus[0]["total_gb"]
                    free_vram_gb = gpus[0]["free_gb"]
                    total_vram_gb = round(sum(g["total_gb"] for g in gpus), 2)
                    total_free_vram_gb = round(sum(g["free_gb"] for g in gpus), 2)
            except Exception:
                pass

        if not has_cuda and not has_mps and platform.system() == "Darwin" and platform.machine() == "arm64":
            has_mps = True
            gpu_name = "Apple Silicon (MPS)"
            gpu_count = 1

        device = "cuda" if has_cuda else ("mps" if has_mps else "cpu")
        if has_cuda and gpu_count > 1:
            label = f"CUDA ({gpu_count}x {gpu_name}, {total_vram_gb} GB Total VRAM)"
        elif has_cuda:
            label = f"CUDA ({gpu_name}, {vram_gb} GB VRAM)"
        else:
            label = gpu_name if has_mps else "CPU only"

        result = {
            "value": label,
            "ok": True,
            "has_gpu": has_cuda or has_mps,
            "device": device,
            "gpu_name": gpu_name,
            "vram_gb": vram_gb,
            "free_vram_gb": free_vram_gb,
            "total_vram_gb": total_vram_gb or vram_gb,
            "total_free_vram_gb": total_free_vram_gb or free_vram_gb,
            "gpu_count": gpu_count,
            "gpus": gpus,
            "bf16_supported": bf16_supported,
        }
        cls._gpu_cache = result
        copy_res = dict(result)
        copy_res["gpus"] = [dict(g) for g in gpus]
        return copy_res

    @classmethod
    def recommend_local_profile(cls, results: Dict[str, Any]) -> Dict[str, Any]:
        gpu = results.get("gpu", {})
        cpu = results.get("cpu", {})
        device = gpu.get("device", "cpu")
        vram_gb = gpu.get("vram_gb", 0.0)

        if device == "cuda":
            quant = "4bit" if vram_gb < 16.0 else "bf16"
            return {
                "device": "cuda",
                "quantization": quant,
                "n_gpu_layers": -1,
                "n_threads": cpu.get("recommended_threads", 4),
            }
        if device == "mps":
            return {
                "device": "mps",
                "quantization": "fp16",
                "n_gpu_layers": -1,
                "n_threads": cpu.get("recommended_threads", 4),
            }
        return {
            "device": "cpu",
            "quantization": "8bit",
            "n_gpu_layers": 0,
            "n_threads": cpu.get("recommended_threads", 4),
        }

    @classmethod
    def print_report(cls, results: Dict[str, Any]) -> None:
        print("\n" + "=" * 50)
        print("PHOENIX AI: HARDWARE RESOURCE REPORT")
        print("=" * 50)
        status = "[v] OK" if results["ram"]["ok"] else "[!] WARN"
        print(f"{status} RAM: {results['ram']['value']} - {results['ram']['message']}")
        status = "[v] OK" if results["disk"]["ok"] else "[!] WARN"
        print(f"{status} DISK: {results['disk']['value']} - {results['disk']['message']}")
        print(f"[*] GPU: {results['gpu']['value']}")
        print(f"[*] CPU: {results['cpu']['physical_cores']} physical / {results['cpu']['logical_cores']} logical cores")
        print(f"[*] OS:  {results['os']['value']}")
        rec = results.get("recommendation", {})
        print(f"[*] Recommended Local Profile: device={rec.get('device')}, quant={rec.get('quantization')}, threads={rec.get('n_threads')}")
        print("=" * 50 + "\n")


if __name__ == "__main__":
    HardwareChecker.check_all()
