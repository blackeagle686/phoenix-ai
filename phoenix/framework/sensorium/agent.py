import asyncio
from typing import Optional, Dict, Any, Union

from phoenix.framework.agent.core.agent import Agent
from phoenix.framework.agent.core.loop import AgentLoop
from phoenix.framework.agent.core.profile import AgentProfile
from phoenix.framework.agent.cognition.schema import ReflectorInputSchema, ReflectorType
from phoenix.framework.sensorium.core.manager import DeviceManager


class SensoriumLoop(AgentLoop):
    """
    A custom execution loop tailored for Sensorium.
    It mandates human approval before the Actor executes any planned actions 
    or tools on physical hardware, ensuring safe interactions.
    """
    async def run(self, prompt: str, memory, session_id: str, max_iterations: int = 15) -> str:
        # Step 1: Think & Plan based on sensor data and user prompt
        plan = await self.planner.generate_initial_plan(prompt, memory, session_id)
        self.memory_manager.set_session(session_id, plan.objective)
        memory.session.set("current_objective", plan.objective)

        for task in plan.tasks:
            self.memory_manager.create_cell(
                task_id=task.task_id,
                description=task.description,
                task_type=getattr(task, "task_type", "other"),
                priority=getattr(task, "priority", "medium"),
                dependencies=getattr(task, "dependencies", [])
            )

        final_answer = ""
        iterations = 0

        for task in plan.tasks:
            if task.status == "done":
                continue

            cell = self.memory_manager.get_cell(task.task_id)

            if not self.memory_manager.are_dependencies_met(task.task_id):
                cell.log_error("Dependencies not met, skipping")
                continue

            cell.task_status = "in_progress"
            task_complete = False

            while not task_complete and iterations < max_iterations and not cell.is_exhausted():
                cell.increment_attempt()
                context = self.memory_manager.get_task_context(task.task_id)

                # --- HUMAN APPROVAL STEP (The Core of Sensorium Safe Execution) ---
                print(f"\n[SENSORIUM AGENT] 🧠 Planned Action: {task.description}")
                
                # Using run_in_executor to not block the async event loop with input()
                loop = asyncio.get_running_loop()
                approval = await loop.run_in_executor(
                    None, 
                    input, 
                    f"[SENSORIUM AGENT] 🛑 Do you approve executing this task on hardware? (y/n/skip): "
                )
                
                if approval.lower() == 'skip':
                    print("[SENSORIUM AGENT] ⏭️ Task skipped by human.")
                    task.status = "skipped"
                    task_complete = True
                    final_answer += f"Task {task.task_id} skipped by human.\n"
                    break
                elif approval.lower() != 'y':
                    print("[SENSORIUM AGENT] 🚫 Action rejected by human. Stopping execution loop.")
                    final_answer += f"\nExecution stopped due to human rejection on task {task.task_id}."
                    await memory.add_interaction(session_id, "assistant", final_answer)
                    return final_answer

                # --- EXECUTION ---
                print("[SENSORIUM AGENT] ✅ Action approved. Executing...")
                actor_output = await self.actor.generate_and_execute(task, context=context)

                result_data = actor_output.dict() if hasattr(actor_output, "dict") else {"output": str(actor_output)}
                cell.add_runtime_result(result_data)

                # --- REFLECTION ---
                ref_context = self.memory_manager.build_retry_context(task.task_id)
                ref_input = ReflectorInputSchema(
                    reflector_type=ReflectorType.TASK,
                    target_id=task.task_id,
                    target_content=result_data,
                    context=ref_context
                )
                
                reflection = await self.reflector.reflect(ref_input)
                cell.add_reflection(reflection.dict() if hasattr(reflection, "dict") else {"feedback": str(reflection)})

                iterations += 1

                if reflection.is_task_complete:
                    task_complete = True
                    task.status = "done"
                    self.memory_manager.mark_complete(task.task_id)
                    final_answer += f"Task {task.task_id} completed. Rating: {reflection.rating}/10.\n"

            if not task_complete and cell.is_exhausted():
                final_answer += f"Task {task.task_id} failed after {cell.attempts} attempts.\n"

        if iterations >= max_iterations:
            final_answer += "\nMax iterations reached before completing all tasks."

        await memory.add_interaction(session_id, "assistant", final_answer)
        return final_answer


class SensoriumAgent(Agent):
    """
    A specialized Agent for the Sensorium hardware framework.
    It acts as the brain for physical devices, focusing on:
    1. Reading and interpreting sensor data (IR, Radar, Sonar/Sound).
    2. Fusing sensor data with user prompts.
    3. Mandating human-in-the-loop approval before emitting hardware commands.
    """
    def __init__(self, device_manager: DeviceManager, profile: Optional[Union[AgentProfile, str, dict]] = None, **kwargs):
        # Override the loop class to our custom SensoriumLoop for human approval
        kwargs['loop_cls'] = SensoriumLoop
        super().__init__(profile=profile, **kwargs)
        
        self.device_manager = device_manager
        
        # State repository for latest sensor readings
        self.latest_sensor_data = {
            "ir": "No data yet",
            "radar": "No data yet",
            "sonar": "No data yet"
        }
        
        from phoenix.framework.sensorium.events.types import SensorEvent, VehicleEvent
        # Subscribe to EventBus to receive real-time sensor and telemetry updates
        self.device_manager.event_bus.subscribe(SensorEvent.DATA_READY, self._on_sensor_update)
        self.device_manager.event_bus.subscribe(VehicleEvent.TELEMETRY_UPDATE, self._on_sensor_update)

    def _on_sensor_update(self, event):
        """Callback to update internal state when new sensor data arrives."""
        sensor_type = event.data.get("type")
        value = event.data.get("value")
        if sensor_type in self.latest_sensor_data:
            self.latest_sensor_data[sensor_type] = value

    def get_environment_context(self) -> str:
        """Format the current sensor readings into a descriptive context for the LLM."""
        context = "--- CURRENT SENSORIUM ENVIRONMENT DATA ---\n"
        context += f"- IR Sensor (Proximity/Heat): {self.latest_sensor_data['ir']}\n"
        context += f"- Radar (Motion/Distance): {self.latest_sensor_data['radar']}\n"
        context += f"- Sonar/Sound (Acoustic): {self.latest_sensor_data['sonar']}\n"
        context += "Note: VLM/Camera vision is currently disabled to conserve resources.\n"
        context += "------------------------------------------\n"
        return context

    async def run(self, prompt: str, session_id: str = None, max_iterations: int = 15, mode: str = "plan") -> str:
        """
        Override the run method to inject real-time environment context into the prompt.
        Forces mode='plan' to ensure proper thought-process and tool usage.
        """
        # Prefix the user's prompt with the current state of all sensors
        env_context = self.get_environment_context()
        enriched_prompt = (
            f"{env_context}\n"
            f"User Request: {prompt}\n\n"
            f"System Instruction: Analyze the sensor data and the user's request. "
            f"Plan the necessary actions, prepare the required commands for the physical devices, "
            f"and formulate your response."
        )
        
        # Pass the enriched prompt to the underlying Agent framework
        return await super().run(prompt=enriched_prompt, session_id=session_id, max_iterations=max_iterations, mode="plan")
