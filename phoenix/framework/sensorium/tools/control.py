from phoenix.framework.agent.tools.base import BaseTool, ToolResult
from phoenix.framework.sensorium.core.manager import DeviceManager

class DeviceControlTool(BaseTool):
    """
    A critical bridge tool that allows the SensoriumAgent to execute commands 
    on physical hardware or actuators (like Defense Batteries, Drones, Motors) 
    via the unified DeviceManager.
    """
    def __init__(self, device_manager: DeviceManager):
        self.name = "execute_device_command"
        self.description = (
            "Executes a command on a specific hardware device or actuator. "
            "Requires 'device_id' and 'command' strings. Extra arguments can be passed."
        )
        self.device_manager = device_manager

    async def execute(self, device_id: str, command: str, **kwargs) -> ToolResult:
        device = self.device_manager.get_device(device_id)
        if not device:
            return ToolResult(success=False, output="", error=f"Device '{device_id}' not found or offline.")
            
        # Construct the command payload
        payload = {"command": command}
        payload.update(kwargs)
        
        try:
            # Route the command to the hardware via the DeviceManager
            success = await device.write(payload)
            if success:
                return ToolResult(success=True, output=f"Command '{command}' successfully sent to and executed by '{device_id}'.")
            else:
                return ToolResult(success=False, output="", error=f"Device '{device_id}' rejected or failed to execute command '{command}'.")
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))
