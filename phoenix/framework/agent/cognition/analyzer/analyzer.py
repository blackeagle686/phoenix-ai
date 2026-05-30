import os
from typing import List, Dict, Any, Optional
from .base import BaseAnalyzer
from ..utils import parse_llm_json

class Analyzer(BaseAnalyzer):
    """
    Scans the repository and identifies relevant files and architectural patterns 
    based on the user's prompt to assist the Planner.
    Includes simple caching for analyzed workspace files.
    """
    
    def __init__(self, llm, profile=None):
        super().__init__(llm, profile=profile)
        self._file_cache: Optional[List[str]] = None
        self._last_root: Optional[str] = None

    async def analyze_workspace(self, prompt: str, root_dir: str = ".") -> Dict[str, Any]:
        """
        Analyzes the root directory and identifies relevant files for the prompt.
        Uses cached file list if available for the same root_dir.
        """
        if self._file_cache is None or self._last_root != root_dir:
            # Refresh cache
            files = []
            for root, dirs, filenames in os.walk(root_dir):
                if any(p in root for p in [".git", "__pycache__", "venv", ".env", "node_modules", ".gemini"]):
                    continue
                for f in filenames:
                    files.append(os.path.join(root, f))
                    if len(files) > 200: # Slightly higher limit for cache
                        break
                if len(files) > 200:
                    break
            self._file_cache = files
            self._last_root = root_dir
        else:
            files = self._file_cache

        files_str = "\n".join(files[:100]) # Still limit LLM context
        
        system_prompt = """
        You are the 'Analyzer' module of an autonomous agent.
        Your job is to look at the list of files in the project and identify which ones are most relevant 
        to the user's request. Also, try to identify the tech stack or core architecture.
        
        Respond with exactly this JSON format:
        {
            "relevant_files": ["file1", "file2"],
            "tech_stack": "e.g. Python/FastAPI",
            "summary": "Brief architectural summary"
        }
        """

        if self.profile:
            system_prompt += f"\n\n{self.profile.to_prompt_string()}"
            
        full_prompt = f"{system_prompt}\n\nFiles in workspace (sample):\n{files_str}\n\nUser Prompt: {prompt}\n\nAnalysis (JSON):"
        response = await self.llm.generate(full_prompt, session_id=None, max_tokens=150)
        
        data = parse_llm_json(response)
        if not data:
            return {
                "relevant_files": [],
                "tech_stack": "Unknown",
                "summary": "Failed to analyze workspace"
            }
        return data

