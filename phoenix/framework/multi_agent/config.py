from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field
from phoenix.framework.agent.core.profile import AgentProfile

class AgentConfig(BaseModel):
    """Configuration for an individual Phoenix Agent."""
    name: str = Field(..., description="Unique name/ID for the agent within the team")
    profile: Optional[Union[AgentProfile, str, Dict[str, Any]]] = None
    llm_type: str = "openai"  # "openai", "local", etc.
    llm_params: Dict[str, Any] = Field(default_factory=dict)
    
    # Component overrides (factory names or direct classes)
    component_overrides: Dict[str, Any] = Field(default_factory=dict)
    
    # Metadata for orchestration
    roles: List[str] = Field(default_factory=list, description="Roles this agent fulfills in the multi-agent system")

class MultiAgentConfig(BaseModel):
    """Configuration for a team of agents."""
    team_name: str = "PhoenixTeam"
    agents: List[AgentConfig]
    shared_memory: bool = False
    max_parallel_agents: int = 5

