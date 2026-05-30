import json
import logging
import os
from datetime import datetime

class StructuredFormatter(logging.Formatter):
    def format(self, record):
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "name": record.name,
            "level": record.levelname,
            "message": record.getMessage(),
        }
        if hasattr(record, "metadata"):
            log_data["metadata"] = record.metadata
        
        # In a real prod env, we might return json.dumps(log_data)
        # For this SDK, we'll keep it readable but rich
        meta_str = f" | META: {json.dumps(record.metadata)}" if hasattr(record, "metadata") else ""
        return f"{log_data['timestamp']} - {log_data['name']} - {log_data['level']} - {log_data['message']}{meta_str}"

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(StructuredFormatter())
        logger.addHandler(handler)
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logger.setLevel(level)
    return logger

