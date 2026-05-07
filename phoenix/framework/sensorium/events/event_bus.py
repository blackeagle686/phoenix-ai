import asyncio
from typing import Any, Callable, Dict, List, Coroutine
from ..core.models import DeviceEvent

class EventBus:
    """
    High-performance Async Event Bus for Sensorium.
    Uses direct callback execution for low latency.
    """
    def __init__(self):
        self._listeners: Dict[str, List[Callable[[DeviceEvent], Any]]] = {}

    def subscribe(self, event_name: str, callback: Callable[[DeviceEvent], Any]) -> None:
        """Subscribe to an event."""
        if event_name not in self._listeners:
            self._listeners[event_name] = []
        self._listeners[event_name].append(callback)

    def unsubscribe(self, event_name: str, callback: Callable[[DeviceEvent], Any]) -> None:
        """Unsubscribe from an event."""
        if event_name in self._listeners:
            try:
                self._listeners[event_name].remove(callback)
            except ValueError:
                pass

    def emit(self, event_name: str, event: DeviceEvent) -> None:
        """
        Emit an event to all subscribers.
        Executes callbacks asynchronously to avoid blocking the emitter.
        """
        if event_name not in self._listeners:
            return

        for callback in self._listeners[event_name]:
            if asyncio.iscoroutinefunction(callback):
                asyncio.create_task(callback(event))
            else:
                # Run synchronous callbacks in a separate thread to avoid blocking the event loop.
                # This is crucial for high-performance Sensorium operations.
                asyncio.create_task(asyncio.to_thread(callback, event))

    async def emit_wait(self, event_name: str, event: DeviceEvent) -> None:
        """Emit and wait for all subscribers to finish (if async)."""
        if event_name not in self._listeners:
            return

        tasks = []
        for callback in self._listeners[event_name]:
            if asyncio.iscoroutinefunction(callback):
                tasks.append(callback(event))
            else:
                tasks.append(asyncio.to_thread(callback, event))
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
