
from typing import List, Dict, Optional

class ConversationSummarizer:
    """
    Automatically condenses old conversation history using the LLM
    when the context window limit is approaching.
    """
    def __init__(self, llm, threshold_chars: int = 6000):
        self.llm = llm
        self.threshold = threshold_chars

    async def maybe_summarize(self, messages: List[Dict]) -> List[Dict]:
        """
        If total chars exceed threshold, summarize older messages into one block.
        Returns a compressed list of messages.
        """
        total = sum(len(m.get("content", "")) for m in messages)
        if total <= self.threshold:
            return messages

        system_msgs = [m for m in messages if m.get("role") == "system"]
        convo_msgs = [m for m in messages if m.get("role") != "system"]

        # Keep the last 4 messages intact, summarize the rest
        to_summarize = convo_msgs[:-4]
        to_keep = convo_msgs[-4:]

        if not to_summarize:
            return messages

        history_text = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in to_summarize])
        prompt = f"Summarize this conversation concisely in 2-3 sentences:\n{history_text}"

        try:
            summary = await self.llm.generate(prompt)
            summary_msg = {"role": "system", "content": f"[Earlier conversation summary]: {summary}"}
            return system_msgs + [summary_msg] + to_keep
        except Exception:
            return messages  # Graceful fallback

