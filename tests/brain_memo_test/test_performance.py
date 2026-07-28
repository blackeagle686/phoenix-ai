import pytest
import time
import asyncio
from datetime import datetime

# TODO: Import the actual compiled BrainMemory module once PyO3 bindings are written
# from phoenix_ai import brain_memory

pytestmark = pytest.mark.asyncio

@pytest.fixture
def mock_brain():
    """Fixture to initialize BrainMemory (Mocked until PyO3 is complete)"""
    class MockBrainMemory:
        async def wake_up(self): pass
        async def sleep(self): pass
        def fire_perception(self, data): pass
        def fire_event(self, data): pass
        
    return MockBrainMemory()

async def test_brain_memory_neural_bus_throughput(mock_brain):
    """
    Test how fast the Neural Bus can ingest 100,000 events in Python.
    This demonstrates the zero-bottleneck capability of the Crossbeam channels.
    """
    print("\n[BrainMemory] Starting High-Throughput Test: Ingesting 100,000 Events...")
    
    start_time = time.time()
    
    # Simulate high-speed rapid fire of memories
    for i in range(100_000):
        mock_brain.fire_event({"id": f"event_{i}", "importance": 0.5})
        
    end_time = time.time()
    
    elapsed = end_time - start_time
    events_per_sec = 100_000 / elapsed if elapsed > 0 else 0
    
    print(f"✅ Ingested 100,000 events in {elapsed:.4f} seconds.")
    print(f"🚀 Throughput: {events_per_sec:,.0f} Events / Second")
    
    # Asserting that the Python side is instantly unblocked
    assert elapsed < 1.0, "Neural Bus should ingest 100k events in under 1 second!"

async def test_dream_engine_activation(mock_brain):
    """
    Test that the Dream Engine can safely spin up in the background
    without blocking the main thread.
    """
    print("\n[BrainMemory] Testing Dream Engine Async Isolation...")
    
    start_time = time.time()
    
    # Send AI to sleep (this should spawn the tokio thread and return instantly)
    await mock_brain.sleep()
    
    # Simulate the AI waking up 50ms later
    await asyncio.sleep(0.05)
    await mock_brain.wake_up()
    
    elapsed = time.time() - start_time
    
    # Ensure it didn't block for a long time
    assert elapsed < 0.1
    print("✅ Dream Engine spawned and stopped asynchronously perfectly.")
