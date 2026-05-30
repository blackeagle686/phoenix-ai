import os
import subprocess
import sys

def run_test_script(script_path):
    print(f"\n{'='*50}")
    print(f"RUNNING: {script_path}")
    print(f"{'='*50}")
    
    result = subprocess.run([sys.executable, script_path], capture_output=False)
    if result.returncode == 0:
        print(f"\n[SUCCESS] {script_path} passed.")
    else:
        print(f"\n[FAILURE] {script_path} failed with exit code {result.returncode}.")
        sys.exit(1)

def main():
    test_dir = "tests/arch_test"
    scripts = [
        "test_stm_cell.py",
        "test_stm_manager.py",
        "test_hybrid_manager.py",
        "test_thinker.py",
        "test_planner.py",
        "test_analyzer.py"
    ]
    
    for script in scripts:
        full_path = os.path.join(test_dir, script)
        if os.path.exists(full_path):
            run_test_script(full_path)
        else:
            print(f"[WARNING] Script {full_path} not found.")

if __name__ == "__main__":
    main()

