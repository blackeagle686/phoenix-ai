from dataclasses import dataclass, field
from typing import List, Dict
from .task import FileTask, Task

@dataclass
class ThinkerOutput:
    main_objective: str
    sub_objectives: List[str] = field(default_factory=list)
    context_memory: List[str] = field(default_factory=list)
    summary_answer: str = ""
    files: Dict[str, FileTask] = field(default_factory=dict)
    tasks: Dict[str, Task] = field(default_factory=dict)

