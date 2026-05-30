from typing import Dict, Any
from .base import BaseReflector
from ..utils import parse_llm_json

class Reflector(BaseReflector):
    """
    Evaluates the outcome of actions and determines task completion or replanning.
    """

    def __init__(self, llm, profile=None):
        super().__init__(llm, profile=profile)

    async def reflect(self, objective: str, action: Dict[str, Any], result: str) -> Dict[str, Any]:
        system_prompt = """
        You are the 'Reflector' module of an autonomous agent.
        Your job is to evaluate the result of the last action taken towards achieving the objective.
        
        Determine:
        1. Is the objective fully accomplished?
        2. What should be learned or saved from this action?
        
        Respond strictly with JSON:
        {
            "is_complete": boolean,
            "reflection": "A short summary of what was learned or what needs to be done next."
        }
        """
        
        if self.profile:
            system_prompt += f"\n\n{self.profile.to_prompt_string()}"
            
        full_prompt = f"{system_prompt}\n\nObjective: {objective}\nAction Taken: {action}\nResult: {result}\n\nReflection (JSON only):"
        response = await self.llm.generate(full_prompt, session_id=None)
        
        data = parse_llm_json(response)
        if not data:
            return {"is_complete": False, "reflection": "Failed to parse reflection"}
            
        return {
            "is_complete": bool(data.get("is_complete", False)),
            "reflection": str(data.get("reflection", ""))
        }

