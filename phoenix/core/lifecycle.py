import asyncio
from typing import List, Callable, Awaitable
from phoenix.services.observability.logger import get_logger

logger = get_logger("Phoenix AI.Lifecycle")

class LifecycleManager:
    def __init__(self):
        self._startup_hooks: List[Callable[[], Awaitable[None]]] = []
        self._shutdown_hooks: List[Callable[[], Awaitable[None]]] = []

    def on_startup(self, hook: Callable[[], Awaitable[None]]):
        self._startup_hooks.append(hook)

    def on_shutdown(self, hook: Callable[[], Awaitable[None]]):
        self._shutdown_hooks.append(hook)

    async def _execute_hooks(self, hooks: List[Callable], name: str, parallel: bool = False):
        if not hooks:
            return

        logger.info(f"Executing {len(hooks)} {name} hooks (parallel={parallel}).")
        
        if parallel:
            # Wrap in safe runners to ensure one failure doesn't cancel others
            async def safe_run(h):
                h_name = getattr(h, "__name__", "unnamed_hook")
                try:
                    await h()
                    logger.debug(f"Successfully executed {name} hook: {h_name}")
                except Exception as e:
                    logger.error(f"FAILED {name} hook '{h_name}': {e}", extra={"metadata": {"error": str(e), "hook": h_name}})
            
            await asyncio.gather(*(safe_run(h) for h in hooks))
        else:
            for hook in hooks:
                h_name = getattr(hook, "__name__", "unnamed_hook")
                try:
                    await hook()
                except Exception as e:
                    logger.error(f"FAILED {name} hook '{h_name}': {e}")

    async def startup(self, parallel: bool = False):
        """Execute all registered startup hooks."""
        await self._execute_hooks(self._startup_hooks, "startup", parallel)

    async def shutdown(self, parallel: bool = False):
        """Execute all registered shutdown hooks."""
        await self._execute_hooks(self._shutdown_hooks, "shutdown", parallel)

lifecycle = LifecycleManager()


