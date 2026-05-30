import time
import uuid
from typing import Optional, Dict, Any
from phoenix.services.observability.logger import get_logger

logger = get_logger("Phoenix AI.Telemetry")

class Telemetry:
    def __init__(self):
        self.active_spans = {}

    def start_span(self, name: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        span_id = str(uuid.uuid4())
        self.active_spans[span_id] = {
            "name": name,
            "start_time": time.perf_counter(),
            "metadata": metadata or {}
        }
        return span_id

    def end_span(self, span_id: str, status: str = "success", error: Optional[str] = None, usage: Optional[Dict[str, int]] = None):
        if span_id not in self.active_spans:
            return

        span = self.active_spans.pop(span_id)
        duration = (time.perf_counter() - span["start_time"]) * 1000 # convert to ms
        
        metadata = span["metadata"]
        metadata.update({
            "status": status,
            "duration_ms": round(duration, 2),
            "usage": usage or {}
        })
        if error:
            metadata["error"] = error

        message = f"Request Completed: {span['name']} | Status: {status} | Latency: {metadata['duration_ms']}ms"
        if usage:
            message += f" | Tokens: {usage.get('total_tokens', 0)}"

        if status == "error":
            logger.error(message, extra={"metadata": metadata})
        else:
            logger.info(message, extra={"metadata": metadata})

tracer = Telemetry()

