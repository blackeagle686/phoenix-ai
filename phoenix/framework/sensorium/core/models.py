import time
from typing import Any, Dict, Optional

class SensorData:
    """
    Standardized data format for all sensor readings.
    Optimized with __slots__ for maximum memory efficiency and speed.
    """
    __slots__ = ['device_id', 'type', 'value', 'unit', 'timestamp', 'metadata']
    
    def __init__(self, 
                 device_id: str, 
                 type: str, 
                 value: Any, 
                 unit: Optional[str] = None, 
                 timestamp: Optional[float] = None, 
                 metadata: Optional[Dict[str, Any]] = None):
        self.device_id = device_id
        self.type = type
        self.value = value
        self.unit = unit
        self.timestamp = timestamp if timestamp is not None else time.time()
        self.metadata = metadata if metadata is not None else {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "device_id": self.device_id,
            "type": self.type,
            "value": self.value,
            "unit": self.unit,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }

class DeviceEvent:
    """
    Standardized event object for the Sensorium Event Bus.
    Optimized with __slots__.
    """
    __slots__ = ['event_name', 'source_id', 'data', 'timestamp']
    
    def __init__(self, 
                 event_name: str, 
                 source_id: str, 
                 data: Any, 
                 timestamp: Optional[float] = None):
        self.event_name = event_name
        self.source_id = source_id
        self.data = data
        self.timestamp = timestamp if timestamp is not None else time.time()
