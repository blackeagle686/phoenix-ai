import unittest

from phoenix.services.cache.semantic import SemanticCache


class FakeEmbeddings:
    """
    Deterministic embedding stub for semantic cache testing.
    """

    def __init__(self, mapping=None):
        self.mapping = mapping or {}

    def embed_query(self, text: str):
        if text in self.mapping:
            return self.mapping[text]
        # Default orthogonal-ish fallback for unknown prompts.
        return [0.0, 1.0, 0.0]


class TestSemanticCache(unittest.IsolatedAsyncioTestCase):
    async def test_cache_miss_when_empty(self):
        embeddings = FakeEmbeddings({"hello": [1.0, 0.0, 0.0]})
        cache = SemanticCache(embeddings=embeddings, threshold=0.95)

        result = await cache.get_similar("hello")
        self.assertIsNone(result)

    async def test_exact_prompt_hit(self):
        embeddings = FakeEmbeddings(
            {
                "prompt-a": [1.0, 0.0, 0.0],
                "prompt-a-variant": [1.0, 0.0, 0.0],
            }
        )
        cache = SemanticCache(embeddings=embeddings, threshold=0.95)

        await cache.add("prompt-a", "response-a")
        result = await cache.get_similar("prompt-a-variant")

        self.assertEqual(result, "response-a")

    async def test_threshold_respected_for_low_similarity(self):
        embeddings = FakeEmbeddings(
            {
                "prompt-a": [1.0, 0.0, 0.0],
                "very-different": [0.0, 1.0, 0.0],
            }
        )
        cache = SemanticCache(embeddings=embeddings, threshold=0.95)

        await cache.add("prompt-a", "response-a")
        result = await cache.get_similar("very-different")

        self.assertIsNone(result)

    async def test_returns_first_matching_cached_response(self):
        embeddings = FakeEmbeddings(
            {
                "prompt-a": [1.0, 0.0, 0.0],
                "prompt-b": [0.99, 0.01, 0.0],
                "query": [1.0, 0.0, 0.0],
            }
        )
        cache = SemanticCache(embeddings=embeddings, threshold=0.95)

        await cache.add("prompt-a", "response-a")
        await cache.add("prompt-b", "response-b")

        # Current implementation returns first matching item in insertion order.
        result = await cache.get_similar("query")
        self.assertEqual(result, "response-a")


if __name__ == "__main__":
    unittest.main()

