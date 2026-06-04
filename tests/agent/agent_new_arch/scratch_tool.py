import asyncio
from phoenix.framework.agent.tools.io import FileWriteTool

async def test_tool():
    tool = FileWriteTool()
    res = await tool.execute(file_path="summary__AA.txt", content="This is a test summary.")
    print(res)

asyncio.run(test_tool())
