import unittest
import os
import asyncio
from phoenix.framework.agent.tools.code import PythonAnalyzerTool
from phoenix.framework.agent.tools.patch import MultiBlockUpdateTool

class TestEngineeringTools(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.test_file = "test_script.py"
        self.content = """
class MyClass:
    def method_one(self):
        return 1

def top_function():
    return 2
"""
        with open(self.test_file, "w") as f:
            f.write(self.content)

    def tearDown(self):
        if os.path.exists(self.test_file):
            os.remove(self.test_file)

    async def test_python_analyzer(self):
        print("\n--- Testing PythonAnalyzerTool ---")
        tool = PythonAnalyzerTool()
        print(f"[*] Analyzing file: {self.test_file}")
        result = await tool.execute(file_path=self.test_file)
        
        self.assertTrue(result.success)
        data = result.output
        print(f"[v] Found Classes: {list(data['classes'].keys())}")
        print(f"[v] Found Functions: {[f['name'] for f in data['functions']]}")
        self.assertIn("MyClass", data["classes"])
        self.assertEqual(data["functions"][0]["name"], "top_function")

    async def test_multi_block_update(self):
        print("\n--- Testing MultiBlockUpdateTool ---")
        tool = MultiBlockUpdateTool()
        edits = [
            {"target": "return 1", "replacement": "return 100"},
            {"target": "return 2", "replacement": "return 200"}
        ]
        
        print(f"[*] Applying {len(edits)} edits to {self.test_file}...")
        result = await tool.execute(file_path=self.test_file, edits=edits)
        self.assertTrue(result.success)
        
        with open(self.test_file, "r") as f:
            new_content = f.read()
        print("[v] File updated successfully.")
        self.assertIn("return 100", new_content)
        self.assertIn("return 200", new_content)

if __name__ == "__main__":
    unittest.main()
