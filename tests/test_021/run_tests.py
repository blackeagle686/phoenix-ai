import asyncio
import sys
import os

# Add the project root to sys.path
# This script is in tests/test_021/run_tests.py, so the root is two levels up.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from tests.test_021.test_chatbot import test_chatbot_full_flow
from tests.test_021.test_agent import test_agent_full_flow
from phoenix.core.config import config

async def run_all_v021_tests():
    print("🚀 Starting Phoenix Framework v0.2.1 Integration Tests")
    
    try:
        await test_chatbot_full_flow()
        await test_agent_full_flow()
        print("\\n✨ ALL INTEGRATION TESTS PASSED ✨")
    except Exception as e:
        print(f"\\n💥 TEST SUITE FAILED: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(run_all_v021_tests())

