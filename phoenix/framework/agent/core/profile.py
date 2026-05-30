import json
from typing import List, Optional, Dict
from pydantic import BaseModel, Field

class Identity(BaseModel):
    name: str = Field(..., description="The agent's name")
    id: str = Field(..., description="Unique identifier for the agent")

class Role(BaseModel):
    title: str = Field(..., description="The agent's title")
    mission: str = Field(..., description="Main mission or purpose in the system")

class Personality(BaseModel):
    communication_tone: str = Field(..., description="Tone of communication")
    response_style: str = Field(..., description="Style of response")

class AgentProfile(BaseModel):
    identity: Identity
    role: Role
    personality: Personality
    rules: List[str] = Field(default_factory=list, description="Strict behavioral constraints")
    capabilities: List[str] = Field(default_factory=list, description="Core skills of the agent")
    tool_access: List[str] = Field(default_factory=list, description="System tools the agent can use")

    @classmethod
    def from_json(cls, file_path: str) -> "AgentProfile":
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)

    @classmethod
    def from_dict(cls, data: Dict) -> "AgentProfile":
        return cls(**data)

    def to_prompt_string(self) -> str:
        """
        Formats the profile into a readable string to be injected into system prompts.
        """
        rules_str = "\n".join([f"- {rule}" for rule in self.rules])
        capabilities_str = "\n".join([f"- {cap}" for cap in self.capabilities])
        tools_str = "\n".join([f"- {tool}" for tool in self.tool_access])

        return f"""
[AGENT PROFILE]
Identity: {self.identity.name} (ID: {self.identity.id})
Role: {self.role.title}
Mission: {self.role.mission}

[PERSONALITY]
Communication Tone: {self.personality.communication_tone}
Response Style: {self.personality.response_style}

[STRICT RULES]
{rules_str}

[CAPABILITIES]
{capabilities_str}

[TOOL ACCESS OVERVIEW]
{tools_str}
"""

