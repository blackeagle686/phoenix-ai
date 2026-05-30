def print_step(msg: str):
    print(f"\n[TEST STEP] {msg}...")

class MockLLM:
    async def generate(self, prompt, **kwargs):
        return "Mocked response"
    async def init(self):
        pass

class MockVectorDB:
    async def add(self, **kwargs): pass
    async def search(self, **kwargs): return []
    async def get_by_metadata(self, **kwargs): return []
    async def delete(self, **kwargs): pass

