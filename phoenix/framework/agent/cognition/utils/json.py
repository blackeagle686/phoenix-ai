import json
import re
from typing import Any, Dict

def parse_llm_json(raw_output: str) -> Dict[str, Any]:
    """
    Safely parses JSON from LLM output, handling markdown blocks and fallback.
    """
    try:
        # Extract json blocks if any
        match = re.search(r'```(?:json)?(.*?)```', raw_output, re.DOTALL)
        if match:
            raw_output = match.group(1)
        
        return json.loads(raw_output.strip())
    except (json.JSONDecodeError, AttributeError):
        return {}

def safe_parse_thinker_output(raw_output: str) -> Dict[str, Any]:
    """Fallback for thinker-specific output."""
    data = parse_llm_json(raw_output)
    if not data:
        return {
            "main_objective": "",
            "sub_objectives": [],
            "context_memory": [],
            "summary_answer": raw_output
        }
    return data

