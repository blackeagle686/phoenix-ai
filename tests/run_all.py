import subprocess
import os
import sys

def run_test(file):
    print(f"\n>>> RUNNING: {file}")
    result = subprocess.run([sys.executable, file], capture_output=False)
    return result.returncode == 0

def main():
    test_dir = "tests"
    test_files = [f for f in os.listdir(test_dir) if f.startswith("test_") and f.endswith(".py")]
    test_files.sort()

    print("="*60)
    print("PHOENIX AI: RUNNING ALL SERVICE TESTS")
    print("="*60)

    results = {}
    for f in test_files:
        path = os.path.join(test_dir, f)
        results[f] = run_test(path)

    print("\n" + "="*60)
    print("TEST SUITE SUMMARY")
    print("="*60)
    for f, success in results.items():
        status = "[v] PASSED" if success else "[!] FAILED"
        print(f"{status} - {f}")
    print("="*60)

if __name__ == "__main__":
    main()

