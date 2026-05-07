import asyncio
import os
import sys

# Ensure we can import from the project root
sys.path.append(os.getcwd())

from phoenix.framework.sensorium.core.manager import DeviceManager
from phoenix.framework.sensorium.plugins.mock_plugin import MockSensorPlugin
from phoenix.framework.sensorium.protocols.serial_protocol import SerialProtocol

async def test_hardware_foundation():
    print("--- Testing Sensorium Hardware Foundation ---")
    
    manager = DeviceManager()
    
    # 1. Test Event Bus
    events_received = []
    def on_event(event):
        print(f"Event Captured: {event.event_name} from {event.source_id}")
        events_received.append(event)
    
    manager.event_bus.subscribe("device_connected", on_event)
    
    # 2. Test Device Lifecycle
    print("\nConnecting Mock Sensor...")
    mock = MockSensorPlugin(device_id="sensor_X")
    await manager.add_device("X", mock)
    
    # 3. Test Reading
    print("\nReading from device...")
    for _ in range(3):
        data = await mock.read()
        print(f"Data: {data.value} {data.unit} at {data.timestamp}")
        await asyncio.sleep(0.5)

    # 4. Test Serial Protocol Mocking (if possible)
    print("\nTesting Protocol Abstraction...")
    # Since we don't have real hardware, we just check if it initializes
    serial = SerialProtocol(port="/dev/ttyMock", baudrate=115200)
    print(f"Serial Protocol created for {serial.port}")
    
    # 5. Shutdown
    print("\nShutting down manager...")
    await manager.shutdown()
    
    print("\nTest Complete.")

if __name__ == "__main__":
    asyncio.run(test_hardware_foundation())
