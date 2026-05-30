import uuid
from datetime import datetime
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field
from enum import Enum


class MessageType(str, Enum):
    """Types of messages agents can exchange."""
    TASK = "task"               # Assign a task to another agent
    ALERT = "alert"             # Critical system alert requiring immediate action
    REPORT = "report"           # Status report or result delivery
    REQUEST = "request"         # Request information or action from another agent
    RESPONSE = "response"       # Response to a previous request
    HEARTBEAT = "heartbeat"     # Periodic health/status check


class Priority(str, Enum):
    """Message priority levels for OS-grade operations."""
    CRITICAL = "critical"       # Immediate action required (e.g., system failure)
    HIGH = "high"               # Important but not emergency
    NORMAL = "normal"           # Standard operations
    LOW = "low"                 # Background/informational


class AgentMessage(BaseModel):
    """
    Standardized communication protocol for inter-agent messaging.
    Every message between agents in the multi-agent system MUST use this format
    to ensure reliable, structured communication at the OS level.
    """
    # Routing
    sender: str = Field(..., description="Name of the sending agent")
    receiver: str = Field(default="*", description="Target agent name, or '*' for broadcast")
    channel: str = Field(default="general", description="Topic channel (e.g., 'monitoring', 'security', 'cli')")
    
    # Classification
    msg_type: MessageType = Field(default=MessageType.TASK, description="Type of message")
    priority: Priority = Field(default=Priority.NORMAL, description="Priority level")
    
    # Payload
    payload: Dict[str, Any] = Field(default_factory=dict, description="Structured data payload")
    summary: str = Field(default="", description="Human-readable summary of the message")
    
    # Tracking
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique message ID")
    correlation_id: Optional[str] = Field(default=None, description="Links request-response pairs")
    timestamp: datetime = Field(default_factory=datetime.now, description="When the message was created")
    
    # Status
    acknowledged: bool = Field(default=False, description="Whether the receiver has acknowledged this message")

    def create_response(self, payload: Dict[str, Any], summary: str = "") -> 'AgentMessage':
        """Create a response message linked to this message via correlation_id."""
        return AgentMessage(
            sender=self.receiver,
            receiver=self.sender,
            channel=self.channel,
            msg_type=MessageType.RESPONSE,
            priority=self.priority,
            payload=payload,
            summary=summary,
            correlation_id=self.message_id,
        )

    def to_agent_prompt(self) -> str:
        """Format this message as a prompt string to feed into an agent's run() method."""
        lines = [
            f"[INCOMING MESSAGE from {self.sender}]",
            f"Type: {self.msg_type.value} | Priority: {self.priority.value} | Channel: {self.channel}",
        ]
        if self.summary:
            lines.append(f"Summary: {self.summary}")
        if self.payload:
            lines.append(f"Data: {self.payload}")
        lines.append(f"Message ID: {self.message_id}")
        if self.correlation_id:
            lines.append(f"In response to: {self.correlation_id}")
        return "\n".join(lines)

