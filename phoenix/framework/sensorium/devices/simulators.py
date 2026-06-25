import asyncio
import random
from typing import Any, Callable, Dict, Optional, Union, Awaitable

from phoenix.framework.sensorium.core.interfaces import DeviceInterface, DeviceStatus
from phoenix.framework.sensorium.events.event_bus import EventBus
from phoenix.framework.sensorium.core.models import DeviceEvent
from phoenix.framework.sensorium.events.types import SensorEvent, VehicleEvent


class BaseSimulator(DeviceInterface):
    """
    Base class for all Sensorium Simulators.
    Highly reusable SDK component: Users can inject custom state_callback and 
    command_callback to easily integrate external simulator libraries 
    (like ROS, AirSim, CARLA) without writing new classes from scratch.
    """
    def __init__(
        self, 
        device_id: str, 
        event_bus: Optional[EventBus] = None, 
        emit_interval: float = 2.0, 
        metadata: Optional[Dict[str, Any]] = None,
        state_callback: Optional[Callable[[], Union[Dict[str, Any], Awaitable[Dict[str, Any]]]]] = None,
        command_callback: Optional[Callable[[Any], Union[bool, Awaitable[bool]]]] = None,
        event_type: str = SensorEvent.DATA_READY
    ):
        super().__init__(device_id, metadata)
        self.event_bus = event_bus
        self.emit_interval = emit_interval
        self.event_type = event_type
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self.state: Dict[str, Any] = {}
        
        # Integration hooks for external libraries
        self.state_callback = state_callback
        self.command_callback = command_callback

    async def connect(self) -> bool:
        self.status = DeviceStatus.READY
        self._running = True
        if self.event_bus:
            self._task = asyncio.create_task(self._simulation_loop())
        return True

    async def disconnect(self) -> bool:
        self._running = False
        if self._task:
            self._task.cancel()
        self.status = DeviceStatus.DISCONNECTED
        return True

    async def read(self) -> Dict[str, Any]:
        """Read current state. Updates on-demand if no background loop is running."""
        if not self.event_bus:
            await self._update_state()
        return self.state

    async def write(self, data: Any) -> bool:
        """Send a command. Uses custom callback if provided, otherwise default internal logic."""
        if self.command_callback:
            if asyncio.iscoroutinefunction(self.command_callback):
                return await self.command_callback(data)
            return self.command_callback(data)
        
        return await self._handle_command(data)

    async def _handle_command(self, data: Any) -> bool:
        """Override in actuating simulators for default behavior."""
        return True

    async def _simulation_loop(self):
        """Background loop to emit events."""
        while self._running:
            await self._update_state()
            if self.event_bus and self.state:
                event = DeviceEvent(self.event_type, self.device_id, self.state)
                self.event_bus.emit(self.event_type, event)
            await asyncio.sleep(self.emit_interval)

    async def _update_state(self):
        """Updates internal state using external callback or internal fallback."""
        if self.state_callback:
            if asyncio.iscoroutinefunction(self.state_callback):
                self.state = await self.state_callback()
            else:
                self.state = self.state_callback()
        else:
            self._generate_internal_state()

    def _generate_internal_state(self):
        """Override in subclasses to generate default random/simulated data."""
        pass


# ==========================================
# 1. Sensors Simulators (Read-Only)
# ==========================================

class IRSimulator(BaseSimulator):
    """Simulates an Infrared (Proximity/Heat) Sensor"""
    def _generate_internal_state(self):
        proximity = round(random.uniform(0.5, 20.0), 2)
        heat = round(random.uniform(20.0, 45.0), 1)
        
        if proximity < 3.0:
            status_msg = f"WARNING: Object close at {proximity}m, Heat Signature: {heat}C"
        else:
            status_msg = f"Clear path. Nearest object {proximity}m."
            
        self.state = {
            "type": "ir", 
            "value": status_msg, 
            "raw": {"proximity_m": proximity, "temp_c": heat}
        }


class RadarSimulator(BaseSimulator):
    """Simulates an active Radar system"""
    def _generate_internal_state(self):
        targets_count = random.randint(0, 3)
        if targets_count > 0:
            targets = []
            for _ in range(targets_count):
                targets.append({
                    "distance_km": round(random.uniform(1, 100), 1),
                    "bearing_deg": round(random.uniform(0, 360), 1),
                    "velocity_kts": round(random.uniform(0, 400), 1)
                })
            status_msg = f"{targets_count} bogies detected on radar."
            self.state = {"type": "radar", "value": status_msg, "raw": {"targets": targets}}
        else:
            self.state = {"type": "radar", "value": "Radar scope clear. No targets.", "raw": {"targets": []}}


class SonarSimulator(BaseSimulator):
    """Simulates an acoustic Sonar system for underwater detection"""
    def _generate_internal_state(self):
        depth = round(random.uniform(10.0, 500.0), 1)
        contacts = random.randint(0, 2)
        
        status_msg = f"Sea depth: {depth}m. "
        status_msg += f"{contacts} biological/mechanical contacts detected." if contacts > 0 else "No contacts."
        
        self.state = {
            "type": "sonar", 
            "value": status_msg, 
            "raw": {"depth_m": depth, "contacts": contacts}
        }


# ==========================================
# 2. Vehicles / Actuators Simulators
# ==========================================

class DroneSimulator(BaseSimulator):
    """Simulates a UAV / Drone telemetry and controls"""
    def __init__(self, *args, **kwargs):
        kwargs.setdefault('event_type', VehicleEvent.TELEMETRY_UPDATE)
        super().__init__(*args, **kwargs)
        self.altitude = 0.0
        self.battery = 100.0
        self.mode = "grounded"

    def _generate_internal_state(self):
        if self.mode == "flying":
            self.battery -= random.uniform(0.1, 0.5)
            self.altitude += random.uniform(-0.5, 0.5)
            if self.battery <= 0:
                self.mode = "crashed"
                self.altitude = 0.0

        self.state = {
            "type": "drone", 
            "value": f"Mode: {self.mode}, Altitude: {round(self.altitude, 1)}m, Battery: {round(self.battery, 1)}%", 
            "raw": {"altitude": self.altitude, "battery": self.battery, "mode": self.mode}
        }

    async def _handle_command(self, data: Any) -> bool:
        command = data.get("command")
        if command == "takeoff" and self.mode == "grounded":
            self.mode = "flying"
            self.altitude = 15.0
            print(f"[DRONE SIM] Executing command: TAKEOFF")
        elif command == "land" and self.mode == "flying":
            self.mode = "grounded"
            self.altitude = 0.0
            print(f"[DRONE SIM] Executing command: LAND")
        elif command == "rtl": # Return to Launch
            self.mode = "grounded"
            self.altitude = 0.0
            print(f"[DRONE SIM] Executing command: RTL (Returning Home)")
        return True


class AircraftSimulator(BaseSimulator):
    """Simulates Fixed-Wing Aircraft telemetry and controls"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.speed = 0.0
        self.altitude = 0.0
        self.heading = 0.0
        
    def _generate_internal_state(self):
        if self.speed > 0:
            self.altitude += random.uniform(-10.0, 10.0)
            self.heading = (self.heading + random.uniform(-1.0, 1.0)) % 360
        
        self.state = {
            "type": "aircraft", 
            "value": f"Airspeed: {round(self.speed,1)} kts, Altitude: {round(self.altitude,1)} ft, Heading: {round(self.heading,1)}°", 
            "raw": {"speed_kts": self.speed, "alt_ft": self.altitude, "heading": self.heading}
        }

    async def _handle_command(self, data: Any) -> bool:
        action = data.get("command")
        if action == "throttle_up":
            self.speed += 50.0
            if self.altitude == 0: self.altitude = 5000.0
        elif action == "throttle_down":
            self.speed = max(0.0, self.speed - 50.0)
            if self.speed == 0: self.altitude = 0.0
        elif action == "steer":
            angle = data.get("angle", 10.0)
            self.heading = (self.heading + angle) % 360
        return True


class BoatSimulator(BaseSimulator):
    """Simulates a Naval Vessel / Boat telemetry and helm controls"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.speed = 0.0
        self.heading = 0.0
        
    def _generate_internal_state(self):
        if self.speed > 0:
            self.heading = (self.heading + random.uniform(-2.0, 2.0)) % 360
            
        self.state = {
            "type": "boat", 
            "value": f"Speed: {round(self.speed,1)} knots, Heading: {round(self.heading,1)}°", 
            "raw": {"speed": self.speed, "heading": self.heading}
        }

    async def _handle_command(self, data: Any) -> bool:
        action = data.get("command")
        if action == "ahead":
            self.speed += 5.0
            print("[BOAT SIM] Engine: Ahead")
        elif action == "stop":
            self.speed = 0.0
            print("[BOAT SIM] Engine: Stop")
        elif action == "port":
            self.heading = (self.heading - 15.0) % 360
            print("[BOAT SIM] Helm: Hard to Port")
        elif action == "starboard":
            self.heading = (self.heading + 15.0) % 360
            print("[BOAT SIM] Helm: Hard to Starboard")
        return True


class DefenseBatterySimulator(BaseSimulator):
    """Simulates an Air Defense Battery (e.g., Patriot / Iron Dome)"""
    def __init__(self, *args, **kwargs):
        kwargs.setdefault('event_type', VehicleEvent.TELEMETRY_UPDATE)
        
        # Unifying devices: We pass metadata to define its capabilities!
        default_metadata = {
            "role": "actuator",
            "class": "air_defense",
            "ammo": 12,
            "status": "safe"
        }
        if 'metadata' in kwargs and kwargs['metadata']:
            default_metadata.update(kwargs['metadata'])
        kwargs['metadata'] = default_metadata
        
        super().__init__(*args, **kwargs)
        self.ammo = self.metadata["ammo"]
        self.mode = "safe" # safe, armed, tracking
        self.active_target = None
        
    def _generate_internal_state(self):
        self.state = {
            "type": "defense_battery", 
            "value": f"Mode: {self.mode.upper()}, Ammo: {self.ammo}, Target: {self.active_target or 'None'}", 
            "raw": {"mode": self.mode, "ammo": self.ammo, "target": self.active_target}
        }

    async def _handle_command(self, data: Any) -> bool:
        action = data.get("command")
        if action == "arm":
            self.mode = "armed"
            print(f"[DEFENSE BATTERY] ⚠️ SYSTEM ARMED!")
        elif action == "safe":
            self.mode = "safe"
            self.active_target = None
            print(f"[DEFENSE BATTERY] 🟢 System Safe.")
        elif action == "track":
            target = data.get("target_id", "UNKNOWN")
            self.mode = "tracking"
            self.active_target = target
            print(f"[DEFENSE BATTERY] 🎯 Tracking target: {target}")
        elif action == "fire":
            if self.mode != "armed" and self.mode != "tracking":
                print(f"[DEFENSE BATTERY] ❌ Cannot fire! System is in {self.mode} mode.")
                return False
            if self.ammo <= 0:
                print(f"[DEFENSE BATTERY] ❌ Cannot fire! Out of ammo.")
                return False
                
            self.ammo -= 1
            print(f"[DEFENSE BATTERY] 🚀 MISSILE FIRED at {self.active_target or 'blind trajectory'}! (Ammo left: {self.ammo})")
            
            # Auto-safe after fire if no target remains
            self.active_target = None
            self.mode = "armed"
        return True

