import asyncio
import time
from typing import Any, Callable, Dict, List, Optional, Awaitable
from dataclasses import dataclass, field
from phoenix.services.observability.logger import get_logger

logger = get_logger("Phoenix AI.StateStore")


@dataclass
class StateEntry:
    """A single entry in the shared state store."""
    key: str
    value: Any
    owner: str                              # Which agent set this value
    updated_at: float = field(default_factory=time.time)
    version: int = 1                        # Incremented on each update


class SharedStateStore:
    """
    Thread-safe, async key-value store for real-time OS state management.
    Agents can read, write, and watch for changes to shared state.
    
    Used for sharing system facts (cpu_usage, disk_space, active_processes)
    between agents without requiring explicit message passing.
    """

    def __init__(self):
        self._store: Dict[str, StateEntry] = {}
        self._watchers: Dict[str, List[Callable[[StateEntry], Awaitable[None]]]] = {}
        self._lock = asyncio.Lock()

    async def set(self, key: str, value: Any, owner: str) -> StateEntry:
        """
        Set a value in the shared state.
        If the key already exists, update it and increment the version.
        Triggers any registered watchers for this key.
        """
        async with self._lock:
            if key in self._store:
                entry = self._store[key]
                entry.value = value
                entry.owner = owner
                entry.updated_at = time.time()
                entry.version += 1
            else:
                entry = StateEntry(key=key, value=value, owner=owner)
                self._store[key] = entry

            logger.debug(f"State updated: {key}={value} (by {owner}, v{entry.version})")

        # Trigger watchers outside the lock to prevent deadlocks
        await self._notify_watchers(key, entry)
        return entry

    async def get(self, key: str) -> Optional[StateEntry]:
        """Get a state entry by key. Returns None if not found."""
        return self._store.get(key)

    async def get_value(self, key: str, default: Any = None) -> Any:
        """Get just the value for a key. Returns default if not found."""
        entry = self._store.get(key)
        return entry.value if entry else default

    async def get_all(self) -> Dict[str, StateEntry]:
        """Get the entire state store."""
        return dict(self._store)

    async def get_by_owner(self, owner: str) -> Dict[str, StateEntry]:
        """Get all state entries set by a specific agent."""
        return {k: v for k, v in self._store.items() if v.owner == owner}

    async def delete(self, key: str) -> bool:
        """Remove a key from the state store."""
        if key in self._store:
            del self._store[key]
            logger.debug(f"State deleted: {key}")
            return True
        return False

    def watch(self, key: str, callback: Callable[[StateEntry], Awaitable[None]]) -> None:
        """
        Register a watcher callback for a specific key.
        The callback will be triggered whenever the key's value changes.
        
        Perfect for reactive OS patterns:
            state_store.watch("cpu_usage", handle_cpu_change)
        """
        if key not in self._watchers:
            self._watchers[key] = []
        self._watchers[key].append(callback)
        logger.debug(f"Watcher registered for key '{key}'")

    def unwatch(self, key: str, callback: Callable = None) -> None:
        """Remove a watcher. If callback is None, remove all watchers for the key."""
        if key in self._watchers:
            if callback:
                self._watchers[key] = [cb for cb in self._watchers[key] if cb != callback]
            else:
                del self._watchers[key]

    async def _notify_watchers(self, key: str, entry: StateEntry) -> None:
        """Trigger all registered watcher callbacks for a key."""
        for callback in self._watchers.get(key, []):
            try:
                await callback(entry)
            except Exception as e:
                logger.error(f"Watcher error for key '{key}': {e}")

    def snapshot(self) -> Dict[str, Any]:
        """
        Get a plain dict snapshot of all current state values.
        Useful for injecting into agent prompts as context.
        """
        return {k: {"value": v.value, "owner": v.owner, "version": v.version} for k, v in self._store.items()}

