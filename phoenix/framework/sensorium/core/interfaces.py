from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from enum import Enum

class DeviceStatus(Enum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    READY = "ready"
    BUSY = "busy"
    ERROR = "error"

class DeviceInterface(ABC):
    """
    Base interface for all Phoenix Sensorium hardware devices.
    Optimized for high-performance async interaction.
    """
    
    def __init__(self, device_id: str, metadata: Optional[Dict[str, Any]] = None):
        self.device_id = device_id
        self.metadata = metadata or {}
        self.status = DeviceStatus.DISCONNECTED

    @abstractmethod
    async def connect(self) -> bool:
        """Initialize connection to the hardware."""
        pass

    @abstractmethod
    async def disconnect(self) -> bool:
        """Safely close the connection."""
        pass

    @abstractmethod
    async def read(self) -> Dict[str, Any]:
        """Read data from the sensor/device."""
        pass

    @abstractmethod
    async def write(self, data: Any) -> bool:
        """Send data/command to the device (if applicable)."""
        pass

    def get_info(self) -> Dict[str, Any]:
        """Return device metadata and current status."""
        return {
            "device_id": self.device_id,
            "status": self.status.value,
            "metadata": self.metadata
        }

