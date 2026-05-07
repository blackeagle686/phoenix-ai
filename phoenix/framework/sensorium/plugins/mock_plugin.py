import asyncio
from typing import Any
from .base import DevicePlugin
from .metadata import PluginMetadata
from ..core.capabilities import DeviceCapability
from ..core.models import SensorData

class MockSensorPlugin(DevicePlugin):
    """
    A mock sensor plugin for testing the Sensorium SDK.
    """
    def __init__(self, device_id: str = "mock_01"):
        metadata = PluginMetadata(
            name="Mock Sensor",
            version="1.0.0",
            author="Phoenix Team",
            description="A virtual sensor that generates random data.",
            hardware_requirements=["None"]
        )
        capabilities = [DeviceCapability.READ, DeviceCapability.TRIGGER]
        super().__init__(device_id, metadata, capabilities)
        self.value = 25.0

    async def connect(self) -> bool:
        await asyncio.sleep(0.1) # Simulate hardware latency
        return True

    async def read(self) -> SensorData:
        self.value += 0.1
        return SensorData(
            device_id=self.device_id,
            type="temperature",
            value=round(self.value, 2),
            unit="C"
        )

    async def disconnect(self) -> bool:
        return True

    async def write(self, data: Any) -> bool:
        return True
