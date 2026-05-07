from enum import Enum, auto

class DeviceCapability(Enum):
    """
    Standard capabilities that a device can report.
    Used by agents to determine how to interact with hardware.
    """
    READ = auto()         # Can read data (sensors)
    WRITE = auto()        # Can receive commands (actuators)
    STREAM = auto()       # Supports real-time data streaming
    TRIGGER = auto()      # Can trigger external events
    GPIO = auto()         # Supports direct GPIO pin control
    SERIAL = auto()       # Uses serial communication
    NETWORK = auto()      # Communicates over network (MQTT/HTTP)
    VISION = auto()       # Provides image/video data
    AUDIO = auto()        # Provides audio data
    POSITION = auto()     # Provides GPS/IMU data
