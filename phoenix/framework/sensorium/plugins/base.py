from typing import List, Set, Any, Dict, Optional
from ..core.interfaces import DeviceInterface
from ..core.capabilities import DeviceCapability
from .metadata import PluginMetadata

class DevicePlugin(DeviceInterface):
    """
    Base class for all Sensorium plugins.
    Combines hardware interface with metadata and capabilities.
    """
    def __init__(self, 
                 device_id: str, 
                 metadata: PluginMetadata,
                 capabilities: List[DeviceCapability]):
        super().__init__(device_id)
        self.plugin_metadata = metadata
        self.capabilities: Set[DeviceCapability] = set(capabilities)

    def has_capability(self, capability: DeviceCapability) -> bool:
        """Check if the device supports a specific capability."""
        return capability in self.capabilities

    def get_plugin_info(self) -> Dict[str, Any]:
        """Return detailed plugin information."""
        info = self.get_info()
        info.update({
            "plugin": {
                "name": self.plugin_metadata.name,
                "version": self.plugin_metadata.version,
                "author": self.plugin_metadata.author
            },
            "capabilities": [c.name for c in self.capabilities]
        })
        return info
