from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class PluginMetadata:
    """
    Metadata for a Sensorium Hardware Plugin.
    """
    __slots__ = ['name', 'version', 'author', 'description', 'hardware_requirements', 'min_phoenix_version']
    
    name: str
    version: str
    author: str
    description: str
    hardware_requirements: List[str] = field(default_factory=list)
    min_phoenix_version: str = "0.2.0"
