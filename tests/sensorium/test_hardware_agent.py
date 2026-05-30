import asyncio
import os
import sys
from typing import Any, Dict

# Ensure we can import from the project root
sys.path.append(os.getcwd())

from phoenix.services.llm.openai import OpenAILLM
from phoenix.framework.agent.core.agent import Agent
from phoenix.framework.agent.tools.base import tool, ToolResult
from phoenix.framework.sensorium.core.manager import DeviceManager
from phoenix.framework.sensorium.plugins.mock_plugin import MockSensorPlugin

async def run_hardware_agent_test():
    print("--- Starting Sensorium Hardware Agent Test ---")

    # 1. Initialize Sensorium Hardware SDK
    manager = DeviceManager()
    
    # 2. Add Mock Hardware
    mock_sensor = MockSensorPlugin(device_id="thermometer_01")
    await manager.add_device("main_sensor", mock_sensor)
    print(f"Registered device: {mock_sensor.get_info()}")

    # 3. Define Tools for the Agent
    @tool(name="read_temperature", description="Reads the current temperature from the hardware sensor.")
    async def read_temperature():
        device = manager.get_device("main_sensor")
        if not device:
            return "Sensor not found."
        data = await device.read()
        return f"Temperature: {data.value}{data.unit}"

    @tool(name="list_connected_hardware", description="Lists all hardware devices connected to the Sensorium SDK.")
    def list_hardware():
        devices = manager.registry.list_devices()
        return {"connected_devices": devices}

    # 4. Initialize LLM with User Credentials
    # NOTE: These are the credentials provided by the user
    llm = OpenAILLM(
        api_key="ak_2yp3Xw1Ny7ky2pF7er9x93ZO9jj6G",
        base_url="https://api.longcat.chat/openai",
        model="LongCat-Flash-Lite"
    )

    # 5. Initialize Agent
    agent = Agent(llm=llm)
    
    # 6. Register Hardware Tools
    agent.register_tool(read_temperature)
    agent.register_tool(list_hardware)

    print("Agent initialized with hardware tools.")

    # 7. Run Test Query
    query = "Check the hardware status. What devices are connected and what is the current temperature?"
    print(f"\nUser Query: {query}")
    
    print("\nAgent Thinking...")
    try:
        # Run in 'plan' mode to force tool usage
        response = await agent.run(query, mode="plan")
        print(f"\nAgent Response:\n{response}")
    except Exception as e:
        print(f"\nError during agent execution: {e}")
    finally:
        # 8. Shutdown hardware
        await manager.shutdown()

if __name__ == "__main__":
    asyncio.run(run_hardware_agent_test())

