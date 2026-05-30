import unittest
import os
from phoenix.framework.agent.tools.io import FileReadTool, FileWriteTool, FileSearchTool

class TestIOTools(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.test_file = "io_test.txt"
        self.content = "hello phoenix\nthis is a test file"
        with open(self.test_file, "w") as f:
            f.write(self.content)

    def tearDown(self):
        if os.path.exists(self.test_file):
            os.remove(self.test_file)

    async def test_file_read(self):
        print("\n--- Testing FileReadTool ---")
        tool = FileReadTool()
        print(f"[*] Reading {self.test_file}...")
        result = await tool.execute(file_path=self.test_file)
        self.assertTrue(result.success)
        print(f"[v] Content Read: {result.output}")
        joined_content = "".join([c['content_block'] for c in result.output['content']])
        self.assertEqual(joined_content, self.content)

    async def test_file_write(self):
        print("\n--- Testing FileWriteTool ---")
        tool = FileWriteTool()
        new_file = "new_io_test.txt"
        print(f"[*] Writing to {new_file}...")
        result = await tool.execute(file_path=new_file, content="new content")
        self.assertTrue(result.success)
        self.assertTrue(os.path.exists(new_file))
        print("[v] File written and verified.")
        os.remove(new_file)

    async def test_file_search(self):
        print("\n--- Testing FileSearchTool ---")
        tool = FileSearchTool()
        print(f"[*] Searching for 'phoenix' in {self.test_file}...")
        result = await tool.execute(path=self.test_file, pattern="phoenix")
        self.assertTrue(result.success)
        print(f"[v] Search Result: {result.output}")
        self.assertEqual(result.output['total_matches'], 1)
        self.assertEqual(result.output['matches'][0]['line_number'], 1)
        self.assertEqual(result.output['matches'][0]['line_content'], "hello phoenix")

if __name__ == "__main__":
    unittest.main()

