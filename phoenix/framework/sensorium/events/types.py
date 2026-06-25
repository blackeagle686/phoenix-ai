from enum import Enum

class SystemEvent(str, Enum):
    """Core SDK system events."""
    STARTUP = "sys.startup"
    SHUTDOWN = "sys.shutdown"
    FATAL_ERROR = "sys.fatal_error"

class DeviceLifecycleEvent(str, Enum):
    """Standard lifecycle events for any hardware/simulator."""
    CONNECTED = "dev.connected"
    DISCONNECTED = "dev.disconnected"
    CONNECTING = "dev.connecting"
    ERROR = "dev.error"
    CALIBRATING = "dev.calibrating"
    READY = "dev.ready"

class SensorEvent(str, Enum):
    """Standard events for read-only sensors (IR, Radar, Sonar)."""
    DATA_READY = "sensor.data_ready"
    THRESHOLD_WARNING = "sensor.threshold_warning"
    THRESHOLD_CRITICAL = "sensor.threshold_critical"
    SIGNAL_LOST = "sensor.signal_lost"

class ActuatorEvent(str, Enum):
    """Standard events for devices that receive commands (Motors, Relays)."""
    COMMAND_RECEIVED = "actuator.command_received"
    COMMAND_EXECUTED = "actuator.command_executed"
    COMMAND_FAILED = "actuator.command_failed"
    OVERLOAD_WARNING = "actuator.overload_warning"

class VehicleEvent(str, Enum):
    """Standard events for autonomous vehicles (Drones, Boats, Aircraft)."""
    TELEMETRY_UPDATE = "vehicle.telemetry_update"
    WAYPOINT_REACHED = "vehicle.waypoint_reached"
    LOW_BATTERY = "vehicle.low_battery"
    COLLISION_WARNING = "vehicle.collision_warning"
    MODE_CHANGED = "vehicle.mode_changed"
