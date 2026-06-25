import asyncio
import os
import sys

# Ensure SDK is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from phoenix.framework.sensorium.core.manager import DeviceManager
from phoenix.framework.sensorium.devices.simulators import RadarSimulator
from phoenix.framework.sensorium.agent import SensoriumAgent
from phoenix.framework.sensorium.events.types import SensorEvent

class SkyGuardSystem:
    """
    A real-time AI Air Defense monitoring system.
    Monitors a radar simulator and instantly reports threats to the Commander via AI.
    """
    def __init__(self):
        self.manager = DeviceManager()
        self.agent = SensoriumAgent(device_manager=self.manager)
        self.is_alerting = False

    async def start(self):
        print("🛡️ [SKY GUARD] System Initializing...")
        
        # 1. Setup Radar (scanning every 1 second for fast real-time response)
        # Using the BaseSimulator's emit_interval argument
        radar = RadarSimulator(device_id="sky_radar_north", event_bus=self.manager.event_bus, emit_interval=1.0)
        await self.manager.add_device("sky_radar_north", radar)
        
        # 2. Subscribe to radar events (Event-Driven Architecture)
        self.manager.event_bus.subscribe(SensorEvent.DATA_READY, self._on_radar_event)
        
        print("🛰️ [SKY GUARD] Radar online. Scanning airspace...\n")
        
        try:
            # Keep the system running infinitely
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            print("\n[SKY GUARD] Shutting down...")
            await self.manager.shutdown()

    async def _on_radar_event(self, event):
        # We only care about our specific radar
        if event.source_id != "sky_radar_north":
            return
            
        data = event.data
        targets = data.get("raw", {}).get("targets", [])
        
        # Real-time condition: Only alert if targets exist AND we aren't already alerting
        if len(targets) > 0 and not self.is_alerting:
            self.is_alerting = True
            print(f"\n🚨 [CRITICAL ALERT] {len(targets)} bogies detected! Handing over to AI Commander...")
            
            # Format target details for the AI prompt
            targets_info = "\n".join([f"- Target {i+1}: Dist {t['distance_km']}km, Bearing {t['bearing_deg']}°, Speed {t['velocity_kts']}kts" for i, t in enumerate(targets)])
            
            prompt = (
                f"You are the Tactical AI for Sky Guard. The radar just detected {len(targets)} unidentified targets in the airspace.\n"
                f"Target Details:\n{targets_info}\n\n"
                f"Generate a VERY SHORT, URGENT, military-style alert report for the Human Commander. "
                f"List the threats clearly. Do not plan actions, just report the situation."
            )
            
            # Using fast_ans mode! We bypass the slow planning loop for immediate real-time reporting
            report = await self.agent.run(prompt, mode="fast_ans")
            
            print("\n" + "="*60)
            print("📜 AI TACTICAL REPORT:")
            print("="*60)
            print(report)
            print("="*60 + "\n")
            
            # Cooldown to avoid spamming alerts for the same targets
            print("⏳ [SKY GUARD] Cooldown for 5 seconds before next alert cycle...")
            await asyncio.sleep(5) 
            self.is_alerting = False
            print("🛰️ [SKY GUARD] Resuming normal radar scan...\n")

if __name__ == "__main__":
    system = SkyGuardSystem()
    asyncio.run(system.start())
