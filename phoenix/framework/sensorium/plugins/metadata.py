from typing import List, Optional

class PluginMetadata:
    """
    Metadata for a Sensorium Hardware Plugin.
    Optimized with __slots__.
    """
    __slots__ = ['name', 'version', 'author', 'description', 'hardware_requirements', 'min_phoenix_version']
    
    def __init__(self, 
                 name: str, 
                 version: str, 
                 author: str, 
                 description: str, 
                 hardware_requirements: Optional[List[str]] = None, 
                 min_phoenix_version: str = "0.2.0"):
        self.name = name
        self.version = version
        self.author = author
        self.description = description
        self.hardware_requirements = hardware_requirements if hardware_requirements is not None else []
        self.min_phoenix_version = min_phoenix_version
