import importlib
import inspect
import os
import logging
from typing import Dict, List, Type, Any
from .base import DevicePlugin

logger = logging.getLogger(__name__)

class PluginLoader:
    """
    Dynamic loader for Sensorium hardware plugins.
    Scans directories and imports classes that inherit from DevicePlugin.
    """
    def __init__(self, plugin_dir: str):
        self.plugin_dir = plugin_dir
        self._loaded_plugins: Dict[str, Type[DevicePlugin]] = {}

    def discover_plugins(self) -> Dict[str, Type[DevicePlugin]]:
        """
        Scan the plugin directory and import all valid DevicePlugin classes.
        """
        if not os.path.exists(self.plugin_dir):
            logger.warning(f"Plugin directory {self.plugin_dir} does not exist.")
            return {}

        # Ensure the directory is on the python path
        # (This is a simplified version, usually we'd use pkgutil)
        for filename in os.listdir(self.plugin_dir):
            if filename.endswith(".py") and filename != "__init__.py":
                module_name = filename[:-3]
                try:
                    # Construct full path or relative import
                    # Assuming plugins are in phoenix.framework.sensorium.plugins.installed
                    # For now, let's just handle it as a dynamic import from path
                    module_path = os.path.join(self.plugin_dir, filename)
                    spec = importlib.util.spec_from_file_location(module_name, module_path)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)

                    for name, obj in inspect.getmembers(module):
                        if (inspect.isclass(obj) and 
                            issubclass(obj, DevicePlugin) and 
                            obj is not DevicePlugin):
                            
                            self._loaded_plugins[name] = obj
                            logger.info(f"Discovered plugin: {name}")
                
                except Exception as e:
                    logger.error(f"Failed to load plugin {module_name}: {e}")

        return self._loaded_plugins

    def get_plugins(self) -> Dict[str, Type[DevicePlugin]]:
        """Return all discovered plugins."""
        return self._loaded_plugins
