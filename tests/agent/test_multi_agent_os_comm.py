import asyncio
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from phoenix.framework.multi_agent.manager import MultiAgentManager
from phoenix.framework.multi_agent.config import MultiAgentConfig, AgentConfig
from phoenix.framework.multi_agent.protocol import AgentMessage, MessageType, Priority

async def main():
    print("--- Testing OS-Grade Inter-Agent Communication ---")

    # 1. Configuration
    team_config = MultiAgentConfig(
        team_name="Hashira OS Core",
        agents=[
            AgentConfig(
                name="Giyu_Monitor",
                llm_type="openai", 
                profile={
                    "identity": {"name": "Giyu", "id": "giyu-01"}, 
                    "role": {"title": "Monitor", "mission": "Monitor system resources"},
                    "personality": {"communication_tone": "serious", "response_style": "short"}
                }
            ),
            AgentConfig(
                name="Gyomei_Admin",
                llm_type="openai",
                profile={
                    "identity": {"name": "Gyomei", "id": "gyomei-01"}, 
                    "role": {"title": "System Admin", "mission": "Manage system processes"},
                    "personality": {"communication_tone": "calm", "response_style": "detailed"}
                }
            )
        ]
    )

    manager = MultiAgentManager(team_config)
    print("\n[✓] Manager initialized with MessageBus and StateStore.")

    # 2. Test State Store (Reactive OS Pattern)
    print("\n--- Testing Shared State Store ---")
    
    # Register a watcher for CPU spikes
    async def on_cpu_change(entry):
        print(f"   [WATCHER] Detected CPU change: {entry.value}% (Owner: {entry.owner})")
        if entry.value > 90.0:
            print(f"   [WATCHER ALERT] CPU is critically high!")
            
    manager.state_store.watch("cpu_usage", on_cpu_change)
    
    # Simulate Giyu monitoring the CPU
    await manager.state_store.set("cpu_usage", 45.2, owner="Giyu_Monitor")
    await manager.state_store.set("cpu_usage", 62.8, owner="Giyu_Monitor")
    await manager.state_store.set("cpu_usage", 95.5, owner="Giyu_Monitor")  # Should trigger alert

    # 3. Test Message Bus (Pub/Sub)
    print("\n--- Testing Async Message Bus ---")
    
    # Subscribe Gyomei to the monitoring channel
    manager.bus.subscribe("Gyomei_Admin", "monitoring")
    
    # Create a structured OS alert
    alert_msg = AgentMessage(
        sender="Giyu_Monitor",
        receiver="*",  # Broadcast to channel
        channel="monitoring",
        msg_type=MessageType.ALERT,
        priority=Priority.CRITICAL,
        payload={"metric": "cpu_usage", "value": 95.5, "action_needed": "kill_rogue_process"},
        summary="CPU usage is critical!"
    )
    
    # Publish the alert
    print("   [Giyu_Monitor] Publishing CRITICAL alert to 'monitoring' channel...")
    await manager.bus.publish(alert_msg)
    
    # Check Gyomei's inbox
    pending = manager.bus.get_messages("Gyomei_Admin")
    print(f"   [Gyomei_Admin] Inbox size: {len(pending)}")
    if pending:
        msg = pending[0]
        print(f"   [Gyomei_Admin] Received message: {msg.summary} (Priority: {msg.priority.value})")
        print("   [Gyomei_Admin] Prompt representation sent to agent:")
        print("--------------------------------------------------")
        print(msg.to_agent_prompt())
        print("--------------------------------------------------")

    print("\n[SUCCESS] OS-Grade Communication tests completed.")

if __name__ == "__main__":
    asyncio.run(main())

