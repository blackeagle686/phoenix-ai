import asyncio
import logging
from typing import Dict, Any, Optional, Callable

logger = logging.getLogger(__name__)

class StreamManager:
    """
    Manages high-bandwidth data streams (Video/Audio/Radar).
    Optimized for zero-copy or low-latency processing.
    """
    def __init__(self, max_buffer_size: int = 10):
        self._streams: Dict[str, asyncio.Queue] = {}
        self._max_buffer_size = max_buffer_size
        self._active = True

    async def create_stream(self, stream_id: str) -> asyncio.Queue:
        """Create a new stream buffer."""
        if stream_id in self._streams:
            return self._streams[stream_id]
        
        queue = asyncio.Queue(maxsize=self._max_buffer_size)
        self._streams[stream_id] = queue
        return queue

    async def push_frame(self, stream_id: str, frame: Any) -> bool:
        """
        Push a new frame into the stream.
        If buffer is full, drops the oldest frame to maintain real-time speed.
        """
        if stream_id not in self._streams:
            return False
        
        queue = self._streams[stream_id]
        
        if queue.full():
            try:
                # Drop oldest frame
                queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
        
        await queue.put(frame)
        return True

    async def get_frame(self, stream_id: str, timeout: Optional[float] = None) -> Optional[Any]:
        """Retrieve the next frame from the stream."""
        if stream_id not in self._streams:
            return None
        
        try:
            if timeout:
                return await asyncio.wait_for(self._streams[stream_id].get(), timeout)
            return await self._streams[stream_id].get()
        except asyncio.TimeoutError:
            return None
        except Exception as e:
            logger.error(f"Stream read error: {e}")
            return None

    def close_stream(self, stream_id: str):
        """Remove a stream."""
        if stream_id in self._streams:
            del self._streams[stream_id]

