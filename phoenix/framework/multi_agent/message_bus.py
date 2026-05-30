import asyncio
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Awaitable
from phoenix.framework.multi_agent.protocol import AgentMessage, Priority
from phoenix.services.observability.logger import get_logger

logger = get_logger("Phoenix AI.MessageBus")


class MessageBus:
    """
    Async pub/sub message bus for inter-agent communication.
    Agents subscribe to channels and receive structured AgentMessages.
    Supports direct messaging, channel broadcasts, and priority-based ordering.
    """

    def __init__(self):
        # channel -> list of subscriber agent names
        self._subscriptions: Dict[str, List[str]] = defaultdict(list)
        # agent_name -> list of pending messages (inbox)
        self._inboxes: Dict[str, List[AgentMessage]] = defaultdict(list)
        # agent_name -> list of async callbacks for real-time message handling
        self._listeners: Dict[str, List[Callable[[AgentMessage], Awaitable[None]]]] = defaultdict(list)
        # Message history for audit/debugging
        self._history: List[AgentMessage] = []
        self._lock = asyncio.Lock()

    def subscribe(self, agent_name: str, channel: str) -> None:
        """Subscribe an agent to a specific channel."""
        if agent_name not in self._subscriptions[channel]:
            self._subscriptions[channel].append(agent_name)
            logger.info(f"Agent '{agent_name}' subscribed to channel '{channel}'")

    def unsubscribe(self, agent_name: str, channel: str) -> None:
        """Unsubscribe an agent from a channel."""
        if agent_name in self._subscriptions[channel]:
            self._subscriptions[channel].remove(agent_name)
            logger.info(f"Agent '{agent_name}' unsubscribed from channel '{channel}'")

    def on_message(self, agent_name: str, callback: Callable[[AgentMessage], Awaitable[None]]) -> None:
        """
        Register a real-time async callback for when a message arrives for an agent.
        Used for event-driven architectures where agents react immediately.
        """
        self._listeners[agent_name].append(callback)

    async def publish(self, message: AgentMessage) -> None:
        """
        Publish a message to the bus.
        - If receiver is '*', deliver to all subscribers of the message's channel.
        - If receiver is a specific agent name, deliver directly to that agent.
        """
        async with self._lock:
            self._history.append(message)

            if message.receiver == "*":
                # Broadcast to all subscribers of the channel
                subscribers = self._subscriptions.get(message.channel, [])
                for agent_name in subscribers:
                    # Don't send back to the sender
                    if agent_name != message.sender:
                        self._inboxes[agent_name].append(message)
                        await self._notify_listeners(agent_name, message)
                logger.info(
                    f"[{message.priority.value.upper()}] {message.sender} -> channel '{message.channel}' "
                    f"({message.msg_type.value}): {message.summary or 'no summary'}"
                )
            else:
                # Direct message to a specific agent
                self._inboxes[message.receiver].append(message)
                await self._notify_listeners(message.receiver, message)
                logger.info(
                    f"[{message.priority.value.upper()}] {message.sender} -> {message.receiver} "
                    f"({message.msg_type.value}): {message.summary or 'no summary'}"
                )

    async def _notify_listeners(self, agent_name: str, message: AgentMessage) -> None:
        """Trigger all registered callbacks for an agent."""
        for callback in self._listeners.get(agent_name, []):
            try:
                await callback(message)
            except Exception as e:
                logger.error(f"Listener error for agent '{agent_name}': {e}")

    def get_messages(self, agent_name: str, channel: str = None, priority: Priority = None) -> List[AgentMessage]:
        """
        Retrieve pending messages for an agent.
        Optionally filter by channel and/or priority.
        Messages are returned sorted by priority (critical first).
        """
        messages = self._inboxes.get(agent_name, [])

        if channel:
            messages = [m for m in messages if m.channel == channel]
        if priority:
            messages = [m for m in messages if m.priority == priority]

        # Sort by priority: critical > high > normal > low
        priority_order = {Priority.CRITICAL: 0, Priority.HIGH: 1, Priority.NORMAL: 2, Priority.LOW: 3}
        messages.sort(key=lambda m: priority_order.get(m.priority, 99))

        return messages

    def consume_messages(self, agent_name: str, channel: str = None) -> List[AgentMessage]:
        """
        Retrieve and remove all pending messages for an agent (like popping from a queue).
        Returns messages sorted by priority.
        """
        messages = self.get_messages(agent_name, channel=channel)
        # Remove consumed messages from inbox
        consumed_ids = {m.message_id for m in messages}
        self._inboxes[agent_name] = [m for m in self._inboxes[agent_name] if m.message_id not in consumed_ids]
        return messages

    def get_history(self, channel: str = None, limit: int = 50) -> List[AgentMessage]:
        """Retrieve message history for auditing/debugging."""
        history = self._history
        if channel:
            history = [m for m in history if m.channel == channel]
        return history[-limit:]

    def clear_inbox(self, agent_name: str) -> None:
        """Clear all pending messages for an agent."""
        self._inboxes[agent_name] = []

