import asyncio
import logging
from typing import Dict, List, Optional, Type
from .interfaces import DeviceInterface, DeviceStatus
from .container import Container
from .models import DeviceEvent
from ..registry.device_registry import DeviceRegistry
from ..events.event_bus import EventBus

logger = logging.getLogger(__name__)

class DeviceManager:
    """
    The central orchestrator for the Sensorium Hardware SDK.
    Manages device lifecycles, health, and event routing.
    """
    def __init__(self, container: Optional[Container] = None):
        self.container = container or Container()
        self.registry = DeviceRegistry()
        self.event_bus = EventBus()
        
        # Register core services in the container
        self.container.register("registry", self.registry)
        self.container.register("event_bus", self.event_bus)
        self.container.register("device_manager", self)

    async def add_device(self, name: str, device: DeviceInterface, auto_connect: bool = True) -> bool:
        """Register and optionally connect a device."""
        self.registry.register(name, device)
        if auto_connect:
            return await self.connect_device(name)
        return True

    async def connect_device(self, name: str) -> bool:
        """Connect a specific device by name."""
        device = self.registry.get(name)
        if not device:
            logger.error(f"Device '{name}' not found.")
            return False
        
        device.status = DeviceStatus.CONNECTING
        success = await device.connect()
        
        if success:
            device.status = DeviceStatus.READY
            self.event_bus.emit("device_connected", DeviceEvent("device_connected", name, {"status": "ready"}))
        else:
            device.status = DeviceStatus.ERROR
            self.event_bus.emit("device_error", DeviceEvent("device_error", name, {"reason": "connection_failed"}))
            
        return success

    async def disconnect_device(self, name: str) -> bool:
        """Disconnect a specific device."""
        device = self.registry.get(name)
        if not device:
            return False
        
        success = await device.disconnect()
        if success:
            device.status = DeviceStatus.DISCONNECTED
            self.event_bus.emit("device_disconnected", DeviceEvent("device_disconnected", name, {}))
        return success

    async def shutdown(self) -> None:
        """Shutdown all devices and clean up."""
        devices = self.registry.list_devices()
        tasks = [self.disconnect_device(name) for name in devices]
        await asyncio.gather(*tasks, return_exceptions=True)
        logger.info("Sensorium Hardware SDK shutdown complete.")

    def get_device(self, name: str) -> Optional[DeviceInterface]:
        """Get a device instance from the registry."""
        return self.registry.get(name)
