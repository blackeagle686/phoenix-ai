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
            return dict(cls._gpu_cache)

        has_cuda = False
        has_mps = False
        vram_gb = 0.0
        gpu_name = ""
        bf16_supported = False

        # Avoid expensive `import torch` unless torch is already imported in this process
        torch_mod = sys.modules.get("torch")
        if torch_mod is not None:
            try:
                if torch_mod.cuda.is_available():
                    has_cuda = True
                    props = torch_mod.cuda.get_device_properties(0)
                    gpu_name = props.name
                    vram_gb = round(props.total_memory / (1024 ** 3), 2)
                    bf16_supported = bool(getattr(torch_mod.cuda, "is_bf16_supported", lambda: False)())
                elif hasattr(torch_mod.backends, "mps") and torch_mod.backends.mps.is_available():
                    has_mps = True
                    gpu_name = "Apple Silicon (MPS)"
            except Exception:
                pass

        if not has_cuda and not has_mps and shutil.which("nvidia-smi"):
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                    capture_output=True,
                    text=True,
                    timeout=2,
                    check=True,
                )
                line = (out.stdout or "").strip().splitlines()[0]
                parts = [p.strip() for p in line.split(",")]
                if parts:
                    has_cuda = True
                    gpu_name = parts[0]
                    if len(parts) > 1 and parts[1].isdigit():
                        vram_gb = round(int(parts[1]) / 1024.0, 2)
            except Exception:
                pass

        if not has_cuda and not has_mps and platform.system() == "Darwin" and platform.machine() == "arm64":
            has_mps = True
            gpu_name = "Apple Silicon (MPS)"

        device = "cuda" if has_cuda else ("mps" if has_mps else "cpu")
        label = (
            f"CUDA ({gpu_name}, {vram_gb} GB VRAM)"
            if has_cuda
            else (gpu_name if has_mps else "CPU only")
        )
        result = {
            "value": label,
            "ok": True,
            "has_gpu": has_cuda or has_mps,
            "device": device,
            "gpu_name": gpu_name,
            "vram_gb": vram_gb,
            "bf16_supported": bf16_supported,
        }
        cls._gpu_cache = result
        return dict(result)

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
