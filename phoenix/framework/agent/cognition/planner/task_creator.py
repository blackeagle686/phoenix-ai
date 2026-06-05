"""
    Task creator is responsible for creating tasks based on the objective.
    It uses the LLM to generate tasks from the objective.

"""

import json
from typing import Any, Optional, List, Dict
from uuid import uuid4

from phoenix.framework.agent.cognition.planner.schema import (
    Problem, Solution, Task, TaskType, TaskPriority, TaskStatus, 
    SolutionType, ProblemComplexity
)

from phoenix.framework.agent.cognition.reflector.schema import BaseReflectorMeta
from phoenix.framework.agent.cognition.utils import parse_llm_json

class TaskCreator:
    def __init__(self, llm: Any, tools: Optional[Any] = None, cache: Optional[Any] = None ):
        self.llm = llm
        self.tools = tools
        self.cache = cache
        self.problems = []

        if not cache: 
            self.memory = {}


    def _get_default_reflector(self) -> BaseReflectorMeta:
        return BaseReflectorMeta(
            rating=5, 
            feedback="Initial generation, unreflected.", 
            confidence=0.5, 
            reasoning="Created automatically by TaskCreator."
        )

    async def create_problem(self, objective: str) -> Problem:
        prompt = f"""
        Given the following objective, identify the core problem and propose potential solutions.
        Objective: {objective}

        Valid complexity values: "low", "medium", "high", "extreme"

        Respond ONLY in valid JSON matching this exact structure (no comments):
        {{
            "description": "A very brief, 1-sentence description of the core problem",
            "complexity": "low",
            "best_solution_index": 0
        }}
        """
        response = await self.llm.generate(prompt, max_tokens=300)
        data = parse_llm_json(response) or {}
        
        desc = data.get("description", f"Problem for objective: {objective}")
        comp_str = data.get("complexity", "medium").lower()
        try:
            complexity = ProblemComplexity(comp_str)
        except ValueError:
            complexity = ProblemComplexity.MEDIUM

        solutions = []

        dummy_solution = Solution(
            id=uuid4(),
            description="Pending solution generation",
            solution_type=SolutionType.PLAN,
            content="Pending content",
            reflector_result=self._get_default_reflector()
        )

        problem = Problem(
            id=uuid4(),
            description=desc,
            solution=solutions,
            best_solution=dummy_solution,
            complexity=complexity,
            reflector_result=self._get_default_reflector()
        )
        self.problems.append(problem)
        return problem

    async def create_solution(self, problem: Problem, variant_index: int = 1) -> Solution:
        prompt = f"""
        Given the following problem, generate a concise strategy to solve it. DO NOT WRITE CODE. 
        This is solution variant #{variant_index}. Provide a unique, high-level approach.
        
        Problem Description: {problem.description}
        Complexity: {problem.complexity.value}
        
        Valid solution_type values: "plan", "code", "terminal", "network", "mission", "fastanswer", "other"
        
        Respond ONLY in valid JSON matching this exact structure (no comments):
        {{
            "description": "A very short, 1-sentence description of the solution",
            "solution_type": "plan",
            "content": "A high-level outline of the approach (max 3 bullet points, NO CODE)."
        }}
        """
        response = await self.llm.generate(prompt, max_tokens=500)
        data = parse_llm_json(response) or {}
        
        print(f"DEBUG - LLM Raw Response for create_solution (Variant {variant_index}):\n{response}\n")
        print(f"DEBUG - Parsed Data:\n{data}\n")

        stype_str = data.get("solution_type", "other").lower()
        try:
            stype = SolutionType(stype_str)
        except ValueError:
            stype = SolutionType.OTHER

        return Solution(
            id=uuid4(),
            description=data.get("description", "Generated solution for the problem"),
            solution_type=stype,
            content=data.get("content", "Detailed solution steps"),
            reflector_result=self._get_default_reflector()
        )

    async def solve_all_problems(self, solutions_count_for_each_problem=3): 
        import asyncio
        for problem in self.problems:
            tasks = [self.create_solution(problem, variant_index=i+1) for i in range(solutions_count_for_each_problem)]
            solutions = await asyncio.gather(*tasks, return_exceptions=True)
            
            valid_solutions = [s for s in solutions if isinstance(s, Solution)]
            problem.solution = valid_solutions
            if valid_solutions:
                problem.best_solution = valid_solutions[0]

    async def create_task(self, objective: str, user_prompt: str) -> Task:
        """Creates a structured task based on objective and prompt."""
        problem = await self.create_problem(objective)
        await self.solve_all_problems()
        
        registered_tools = []
        if self.tools:
            if hasattr(self.tools, "tools"):
                registered_tools = list(self.tools.tools.keys())
            elif hasattr(self.tools, "get_all_tools_info"):
                try:
                    registered_tools = [t.get("name") for t in self.tools.get_all_tools_info() if t and "name" in t]
                except Exception:
                    pass

        prompt = f"""
        Given the user prompt, objective, and the identified problem, formulate a structured execution task.
        User Prompt: {user_prompt}
        Objective: {objective}
        Problem Description: {problem.description}
        Chosen Solution: {problem.best_solution.content}

        Available Tools: {json.dumps(registered_tools)}

        Valid task_type values: "read", "write", "search", "update", "delete", "block_read", "block_write", "mmap_io", "net_send", "net_recv", "ipc_pipe", "ipc_share", "rpc_call", "batch_load", "tensor_stream", "vector_search", "vram_shuttle", "token_stream", "dma_transfer", "interrupt_req", "port_in", "port_out", "mem_mapped_in", "bus_broadcast", "bus_listen", "adc_sample", "dac_actuate", "pwm_output", "sensor_poll", "watchdog_ping", "other"
        Valid priority values: "critical", "high", "medium", "low"

        Respond ONLY in valid JSON matching this exact structure (no comments):
        {{
            "task_title": "Short descriptive title (max 5 words)",
            "description": "Brief, actionable description of the task (1-2 sentences)",
            "task_type": "other",
            "priority": "medium",
            "dependencies": [],
            "tools_required": ["tool_name1", "tool_name2"]
        }}
        """
        response = await self.llm.generate(prompt, max_tokens=500)
        data = parse_llm_json(response) or {}

        ttype_str = data.get("task_type", "other").lower()
        try:
            task_type = TaskType(ttype_str)
        except ValueError:
            task_type = TaskType.OTHER

        tprio_str = data.get("priority", "medium").lower()
        try:
            priority = TaskPriority(tprio_str)
        except ValueError:
            priority = TaskPriority.MEDIUM

        tools_required = [t for t in data.get("tools_required", []) if t in registered_tools]
        payload = {
            "tools_required": tools_required,
            "solution_context": problem.best_solution.content
        }

        task = Task(
            prompt_id=uuid4(),
            task_id=str(uuid4()),
            dependencies=data.get("dependencies", []),
            task_type=task_type,
            priority=priority,
            status=TaskStatus.PENDING,
            task_title=data.get("task_title", "Generated Task"),
            description=data.get("description", f"Task for: {objective}"),
            task_summary=None,
            complexity=problem.complexity,
            problems=[problem],
            repeat_count=1,
            payload=payload,
            created_by="TaskCreator",
            reflector_result=self._get_default_reflector()
        )
        
        return task




if __name__ == "__main__":
    import sys
    import os
    import asyncio
    
    # Add the project root to sys.path so 'phoenix' can be imported
    
    from phoenix.services.llm.openai import OpenAILLM

    llm = OpenAILLM(
        api_key="ak_2yp3Xw1Ny7ky2pF7er9x93ZO9jj6G",
        model="LongCat-2.0-Preview",
        base_url="https://api.longcat.chat/openai"
    )

    class MockTools:
        def __init__(self):
            # Simulating a tool registry
            self.tools = {
                "file_write": "Write code or text to a specific file path",
                "file_read": "Read content from a specific file path",
                "execute_terminal": "Run terminal commands like npm install or python execution",
                "web_search": "Search the web for up to date information", 
                "folder_create": "Create a new folder",
                
            }

    async def run_test():
        await llm.init()
        
        mock_tools = MockTools()
        test_task = TaskCreator(llm=llm, tools=mock_tools)
        
        objective = "create a modular django project with modular folders structure and modular plugins"
        prompt = "hello"
        task = await test_task.create_task(objective, prompt)
        
        print("\n" + "="*60)
        print("GENERATED TASK (Readable Format)")
        print("="*60)
        # Pydantic V2 .model_dump_json() handles UUIDs and Enums
        print(task.model_dump_json(indent=5))
        print("="*60 + "\n")

    asyncio.run(run_test())