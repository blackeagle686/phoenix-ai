import time
from typing import Any, Dict, Optional
from dataclasses import dataclass, field

@dataclass
class SensorData:
    """
    Standardized data format for all sensor readings.
    Optimized with __slots__ for memory efficiency.
    """
    __slots__ = ['device_id', 'type', 'value', 'unit', 'timestamp', 'metadata']
    
    device_id: str
    type: str
    value: Any
    unit: Optional[str]
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "device_id": self.device_id,
            "type": self.type,
            "value": self.value,
            "unit": self.unit,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }

@dataclass
class DeviceEvent:
    """
    Standardized event object for the Sensorium Event Bus.
    """
    __slots__ = ['event_name', 'source_id', 'data', 'timestamp']
    
    event_name: str
    source_id: str
    data: Any
    timestamp: float = field(default_factory=time.time)
