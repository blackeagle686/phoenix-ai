from typing import Dict, List, Optional
from ..core.interfaces import DeviceInterface

class DeviceRegistry:
    """
    Registry for managing device instances.
    Provides fast lookup and thread-safe registration.
    """
    def __init__(self):
        self._devices: Dict[str, DeviceInterface] = {}

    def register(self, name: str, device: DeviceInterface) -> None:
        """Register a new device instance."""
        if not isinstance(device, DeviceInterface):
            raise TypeError(f"Object {device} must implement DeviceInterface")
        self._devices[name] = device

    def unregister(self, name: str) -> Optional[DeviceInterface]:
        """Remove a device from the registry."""
        return self._devices.pop(name, None)

    def get(self, name: str) -> Optional[DeviceInterface]:
        """Retrieve a device by name."""
        return self._devices.get(name)

    def list_devices(self) -> List[str]:
        """List all registered device names."""
        return list(self._devices.keys())

    def get_all(self) -> Dict[str, DeviceInterface]:
        """Get all registered devices."""
        return self._devices.copy()

