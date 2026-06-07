import asyncio
import os
import sys

# Add the project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from phoenix.framework.agent.runtime.restricted_python import RestrictedPythonRuntime
from phoenix.framework.agent.tools.code import CodeExecutionTool, CommandExecutionTool

async def main():
    print("="*60)
    print("🛡️ Testing Restricted Python Runtime (Zero-Dependency Sandbox)")
    print("="*60)

    runtime = RestrictedPythonRuntime()
    
    print("\n[1] Testing Safe Python Code...")
    safe_code = "result = 10 + 20\nprint('Calculating...', result)"
    res = await runtime.execute_code(safe_code)
    print(f"Code:\n{safe_code}")
    print(f"Result (Success={res.success}): {res.output}")

    print("\n[2] Testing Dangerous Python Code (Importing OS)...")
    dangerous_code = "import os\nos.system('echo Hacked!')"
    res = await runtime.execute_code(dangerous_code)
    print(f"Code:\n{dangerous_code}")
    print(f"Result (Success={res.success}): {res.error}")
    
    print("\n[3] Testing Dangerous Python Code (Using eval/exec)...")
    eval_code = "eval('1+1')"
    res = await runtime.execute_code(eval_code)
    print(f"Code:\n{eval_code}")
    print(f"Result (Success={res.success}): {res.error}")

    print("\n[4] Testing Command Execution (Safe)...")
    res = await runtime.execute_command("echo Hello from Sandbox")
    print(f"Command: echo Hello from Sandbox")
    print(f"Result (Success={res.success}): {res.output.strip()}")

    print("\n[5] Testing Command Execution (Dangerous Pattern)...")
    dangerous_cmd = "rm -rf /"
    res = await runtime.execute_command(dangerous_cmd)
    print(f"Command: {dangerous_cmd}")
    print(f"Result (Success={res.success}): {res.error}")

    print("\n[6] Testing via CodeExecutionTool...")
    tool = CodeExecutionTool(runtime=runtime)
    tool_res = await tool.execute("import subprocess\nsubprocess.call(['ls'])")
    print(f"Tool Output: {tool_res.error}")

if __name__ == "__main__":
    asyncio.run(main())
