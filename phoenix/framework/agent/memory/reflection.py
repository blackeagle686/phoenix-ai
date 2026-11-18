import inspect
from typing import Any, List
from phoenix.framework.agent.memory.budget import truncate_text


class ReflectionMemory:
    """Bounded, deduplicated store for high-signal reflections and lessons learned."""

    def __init__(self, consolidation_threshold: int = 5, max_reflections: int = 10, max_item_chars: int = 240):
        self.reflections: List[str] = []
        self.consolidation_threshold = max(2, consolidation_threshold)
        self.max_reflections = max(self.consolidation_threshold, max_reflections)
        self.max_item_chars = max(80, max_item_chars)

    def add_reflection(self, reflection: str):
        if not reflection:
            return
        clean = truncate_text(" ".join(str(reflection).split()), self.max_item_chars)
        norm = clean.lower()
        if any(r.lower() == norm for r in self.reflections):
            return
        self.reflections.append(clean)
        if len(self.reflections) > self.max_reflections:
            del self.reflections[: len(self.reflections) - self.max_reflections]

    def get_reflections(self, limit: int = 5, max_chars: int = 750) -> str:
        if not self.reflections:
            return ""
        recent = self.reflections[-max(1, limit) :]
        text = "Lessons Learned:\n" + "\n".join(f"- {x}" for x in recent)
        return truncate_text(text, max_chars=max_chars, mode="tail_ellipsis")

    async def consolidate(self, llm: Any):
        if not self.reflections or llm is None or not hasattr(llm, "generate"):
            return None
        joined = "\n".join(f"- {r}" for r in self.reflections)
        prompt = (
            "Distill the following reflections into 1-3 concise, actionable rules or lessons:\n\n"
            f"{joined}"
        )
        try:
            sig = inspect.signature(llm.generate)
            if "max_tokens" in sig.parameters or any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
            ):
                consolidated = await llm.generate(prompt, max_tokens=220)
            else:
                consolidated = await llm.generate(prompt)
            if consolidated:
                self.reflections = [truncate_text(str(consolidated).strip(), self.max_item_chars * 2)]
                return self.reflections[0]
        except Exception:
            pass
        return None

    def clear(self):
        self.reflections.clear()
