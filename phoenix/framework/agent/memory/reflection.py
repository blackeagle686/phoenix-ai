from typing import Any, List


class ReflectionMemory:
    """Stores and consolidates task reflections and lessons learned."""

    def __init__(self, consolidation_threshold: int = 5):
        self.reflections: List[str] = []
        self.consolidation_threshold = consolidation_threshold

    def add_reflection(self, reflection: str):
        if reflection:
            self.reflections.append(str(reflection))

    def get_reflections(self) -> str:
        if not self.reflections:
            return ""
        return "Lessons Learned:\n" + "\n".join([f"- {x}" for x in self.reflections])

    async def consolidate(self, llm: Any):
        if not self.reflections or llm is None or not hasattr(llm, "generate"):
            return None
        joined = "\n".join(f"- {r}" for r in self.reflections)
        prompt = (
            "Consolidate the following agent reflections and lessons learned into a concise set of core insights:\n\n"
            f"{joined}"
        )
        try:
            consolidated = await llm.generate(prompt)
            if consolidated:
                self.reflections = [str(consolidated).strip()]
                return self.reflections[0]
        except Exception:
            pass
        return None

    def clear(self):
        self.reflections.clear()
