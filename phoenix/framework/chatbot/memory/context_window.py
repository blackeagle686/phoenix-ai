
from typing import List, Dict

class ContextWindow:
    """
    Manages a sliding token/character window over conversation history.
    Ensures the active context never exceeds the LLM's token budget.
    """
    def __init__(self, max_chars: int = 8000):
        self.max_chars = max_chars

    def trim(self, messages: List[Dict]) -> List[Dict]:
        """
        Trims the oldest messages from the list until it fits within max_chars.
        Always preserves the system prompt (first message) if present.
        """
        total = sum(len(m.get("content", "")) for m in messages)
        system = [m for m in messages if m.get("role") == "system"]
        rest = [m for m in messages if m.get("role") != "system"]

        while total > self.max_chars and len(rest) > 1:
            removed = rest.pop(0)
            total -= len(removed.get("content", ""))

        return system + rest

