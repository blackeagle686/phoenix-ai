import os
import sys
import psutil
import platform
import shutil
from phoenix.services.observability.logger import get_logger

logger = get_logger("Phoenix AI.Hardware")

class HardwareChecker:
    """
    Checks system resources to ensure suitability for local AI operations.
    """
    MIN_RAM_GB = 8
    RECOMMENDED_RAM_GB = 16
    MIN_DISK_GB = 10

    @classmethod
    def check_all(cls, silent=False):
        results = {
            "ram": cls.check_ram(),
            "disk": cls.check_disk(),
            "os": cls.check_os(),
            "gpu": cls.check_gpu()
        }
        
        all_ok = results["ram"]["ok"] and results["disk"]["ok"]
        
        if not silent:
            cls.print_report(results)
            
        return all_ok, results

    @classmethod
    def check_ram(cls):
        total_ram = psutil.virtual_memory().total / (1024**3)
        return {
            "value": f"{total_ram:.2f} GB",
            "ok": total_ram >= cls.MIN_RAM_GB,
            "recommended": total_ram >= cls.RECOMMENDED_RAM_GB,
            "message": "Insufficient RAM for stable local LLM execution." if total_ram < cls.MIN_RAM_GB else "RAM is sufficient."
        }

    @classmethod
    def check_disk(cls):
        total, used, free = shutil.disk_usage(".")
        free_gb = free / (1024**3)
        return {
            "value": f"{free_gb:.2f} GB Free",
            "ok": free_gb >= cls.MIN_DISK_GB,
            "message": "Low disk space. Model downloads might fail." if free_gb < cls.MIN_DISK_GB else "Disk space is sufficient."
        }

    @classmethod
    def check_os(cls):
        return {
            "value": f"{platform.system()} {platform.release()}",
            "ok": True # All OS supported for SDK, though local LLM might vary
        }

    @classmethod
    def check_gpu(cls):
        # Basic check for NVIDIA/CUDA availability
        has_cuda = False
        try:
            # Check for nvidia-smi
            import subprocess
            subprocess.run(["nvidia-smi"], capture_output=True, check=True)
            has_cuda = True
        except:
            pass
            
        return {
            "value": "CUDA Available" if has_cuda else "No NVIDIA GPU detected (CPU only mode)",
            "ok": True, # CPU fallback is always supported
            "has_gpu": has_cuda
        }

    @classmethod
    def print_report(cls, results):
        print("\n" + "="*50)
        print("PHOENIX AI: HARDWARE RESOURCE REPORT")
        print("="*50)
        
        # RAM
        status = "[v] OK" if results["ram"]["ok"] else "[!] WARN"
        print(f"{status} RAM: {results['ram']['value']} - {results['ram']['message']}")
        if results["ram"]["ok"] and not results["ram"]["recommended"]:
            print(f"      Note: {cls.RECOMMENDED_RAM_GB}GB+ is recommended for optimal performance.")
            
        # Disk
        status = "[v] OK" if results["disk"]["ok"] else "[!] WARN"
        print(f"{status} DISK: {results['disk']['value']} - {results['disk']['message']}")
        
        # GPU
        print(f"[*] GPU: {results['gpu']['value']}")
        
        # OS
        print(f"[*] OS: {results['os']['value']}")
        
        print("="*50)
        if not results["ram"]["ok"] or not results["disk"]["ok"]:
            print("[!] CRITICAL: Your hardware may struggle with local AI models.")
            print("    Recommendation: Use remote (OpenAI/Cloud) providers instead.")
        else:
            print("[v] System is ready for Phoenix AI SDK.")
        print("="*50 + "\n")

if __name__ == "__main__":
    HardwareChecker.check_all()

