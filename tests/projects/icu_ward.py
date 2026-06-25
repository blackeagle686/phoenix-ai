import asyncio
import os
import sys

# Ensure SDK is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from phoenix.framework.sensorium.core.manager import DeviceManager
from phoenix.framework.sensorium.devices.simulators import ICUMonitorSimulator
from phoenix.framework.sensorium.agent import SensoriumAgent
from phoenix.framework.sensorium.events.types import SensorEvent

class ICUWardSystem:
    """
    A real-time AI Medical Monitoring System for an Intensive Care Unit.
    Monitors multiple patient beds concurrently and alerts the AI Doctor if a crisis occurs.
    """
    def __init__(self):
        self.manager = DeviceManager()
        self.agent = SensoriumAgent(device_manager=self.manager)
        self.alerting_patients = set()

    async def start(self):
        print("🏥 [ICU WARD AI] System Initializing...")
        
        # Setup 3 ICU beds with Patient names in Metadata
        beds = [
            ("bed_01", "John Doe"),
            ("bed_02", "Jane Smith"),
            ("bed_03", "Ali Omar")
        ]
        
        for bed_id, patient_name in beds:
            monitor = ICUMonitorSimulator(
                device_id=bed_id, 
                event_bus=self.manager.event_bus, 
                emit_interval=2.0, # Check vitals every 2 seconds
                metadata={"patient_id": patient_name}
            )
            await self.manager.add_device(bed_id, monitor)
        
        # Subscribe to all sensor events
        self.manager.event_bus.subscribe(SensorEvent.DATA_READY, self._on_vitals_update)
        
        print("🩺 [ICU WARD AI] Monitoring patient vitals across all beds...\n")
        
        try:
            # Keep the system running infinitely
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            print("\n[ICU WARD AI] Shutting down...")
            await self.manager.shutdown()

    async def _on_vitals_update(self, event):
        data = event.data
        if data.get("type") != "icu_monitor":
            return
            
        raw_vitals = data.get("raw", {})
        patient_id = raw_vitals.get("patient_id")
        is_critical = raw_vitals.get("is_critical")
        
        # Real-time condition: trigger AI Doctor only on emergencies
        if is_critical and patient_id not in self.alerting_patients:
            self.alerting_patients.add(patient_id)
            print(f"\n🚨 [CODE BLUE] Critical Vitals Detected for {patient_id} (Bed: {event.source_id})!")
            
            prompt = (
                f"You are the AI ICU Doctor. An emergency was just detected for patient '{patient_id}' at {event.source_id}.\n"
                f"Current Vitals:\n"
                f"- Heart Rate: {raw_vitals['heart_rate_bpm']} BPM\n"
                f"- Oxygen (SpO2): {raw_vitals['oxygen_saturation_pct']}%\n"
                f"- Blood Pressure (Sys): {raw_vitals['blood_pressure_sys']} mmHg\n\n"
                f"Generate an URGENT, professional medical alert report. "
                f"Identify the likely medical crisis (e.g., Hypoxia, Tachycardia) and recommend immediate nursing actions. "
                f"Keep it under 4 sentences."
            )
            
            # Using fast_ans mode for rapid medical response without the planning overhead
            report = await self.agent.run(prompt, mode="fast_ans")
            
            print("\n" + "🔴"*25)
            print("🩺 AI MEDICAL EMERGENCY REPORT:")
            print("🔴"*25)
            print(report)
            print("🔴"*25 + "\n")
            
            # Cooldown for this specific patient so we don't spam the console while nurses intervene
            await asyncio.sleep(15) 
            self.alerting_patients.remove(patient_id)
            print(f"✅ [ICU WARD AI] Resumed normal monitoring for {patient_id}...\n")

if __name__ == "__main__":
    system = ICUWardSystem()
    asyncio.run(system.start())
