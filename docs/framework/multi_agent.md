# Multi-Agent Framework Layer

The Phoenix Multi-Agent layer allows you to orchestrate teams of autonomous agents to solve complex tasks through collaboration, parallel execution, and structured pipelines.

## Key Concepts

### 1. The Multi-Agent Manager
The `MultiAgentManager` is the central orchestrator. It manages the lifecycle of multiple agents and provides high-level patterns for agent interaction.

### 2. Orchestration Patterns

#### 📡 Broadcast (Parallel)
The same prompt is sent to all agents in the team simultaneously. This is ideal for brainstorming, getting multiple perspectives, or distributed processing.
- **Method:** `await manager.broadcast(prompt)`
- **Behavior:** Parallel execution using `asyncio.gather`.

#### 🔗 Pipeline (Sequential)
Agents are executed in a predefined sequence, where the output of one agent becomes the input for the next. This is perfect for review cycles or multi-stage workflows.
- **Method:** `await manager.run_pipeline(prompt, ["AgentA", "AgentB"])`
- **Behavior:** Sequential hand-off.

#### 🎯 Targeted
Send a prompt directly to a specific agent within the team.
- **Method:** `await manager.run_targeted("AgentName", prompt)`

#### 🤖 Autonomous Routing (Lead Manager)
The manager acts as a "Lead", automatically reading the prompt, evaluating the team's capabilities, and delegating the task to the most qualified agent.
- **Method:** `await manager.run_autonomous(prompt)`
- **Behavior:** Dynamic, LLM-based delegation.

#### 🔄 Review Loop (Self-Correcting)
Run a resilient, self-correcting loop between a "Doer" and a "Reviewer". If the Reviewer rejects the output, it sends feedback back to the Doer to fix the issues, repeating until approved or `max_loops` is reached.
- **Method:** `await manager.run_with_review(prompt, doer="Giyu", reviewer="Shinobu")`
- **Behavior:** Iterative execution and feedback generation.

---

## 🛠️ Configuration

Multi-agent teams are defined using a structured configuration schema. 
**Pro-Tip**: Enable `shared_memory` to give all agents access to the same conversation history!

```python
from phoenix.framework import MultiAgentConfig, AgentConfig

config = MultiAgentConfig(
    team_name="Hashira-OS Team",
    shared_memory=True, # All agents will now share the same HybridMemory instance
    agents=[
        AgentConfig(
            name="Giyu",
            profile="profiles/giyu.json",  # Identity & Role
            llm_type="openai"
        ),
        AgentConfig(
            name="Shinobu",
            profile="profiles/shinobu.json",
            llm_type="local"  # Mixing different LLMs is supported!
        )
    ]
)
```

---

## 🚀 Example Usage

```python
from phoenix.framework import MultiAgentManager

# 1. Initialize the team
manager = MultiAgentManager(config)

# 2. Run a "Dev & Review" pipeline
result = await manager.run_pipeline(
    prompt="Design a secure authentication module",
    agent_sequence=["Giyu", "Shinobu"]
)

print(f"Final Outcome: {result}")
```

---

## 🧠 Full Complex Example

Here is a complete, runable example demonstrating how to set up a powerful 3-agent team ("Architect", "Coder", and "Reviewer") that collaborates to solve a complex software engineering problem. 

In this example, the **Architect** designs the solution, the **Coder** writes the implementation, and the **Reviewer** performs a security and code quality audit. Because each agent inherits the full Phoenix cognitive loop, they autonomously utilize tools (like web search or file writing) if required.

```python
import asyncio
from phoenix.framework.multi_agent.manager import MultiAgentManager
from phoenix.framework.multi_agent.config import MultiAgentConfig, AgentConfig

async def run_hashira_team():
    # 1. Define the specific Agent Profiles
    architect_config = AgentConfig(
        name="Tengen_Architect",
        llm_type="openai",
        profile={
            "identity": {"name": "Tengen", "id": "hashira-arch-01"},
            "role": {"title": "System Architect", "mission": "Design robust and scalable software architectures."},
            "personality": {"communication_tone": "flamboyant and direct", "response_style": "detailed"},
            "rules": ["Always outline a clear step-by-step plan."],
            "capabilities": ["System Design", "Architecture"],
            "tool_access": ["web_search"]
        }
    )

    coder_config = AgentConfig(
        name="Giyu_Coder",
        llm_type="openai", 
        profile={
            "identity": {"name": "Giyu", "id": "hashira-code-02"},
            "role": {"title": "Lead Developer", "mission": "Write precise, bug-free Python code based on architectural plans."},
            "personality": {"communication_tone": "calm and quiet", "response_style": "code-heavy"},
            "rules": ["Follow PEP 8.", "Do not write bloated code.", "Always include type hints."],
            "capabilities": ["Python", "Algorithms", "Implementation"],
            "tool_access": ["python_repl", "file_write"]
        }
    )

    reviewer_config = AgentConfig(
        name="Shinobu_Reviewer",
        llm_type="openai",
        profile={
            "identity": {"name": "Shinobu", "id": "hashira-rev-03"},
            "role": {"title": "Security & Code Reviewer", "mission": "Find bugs, security flaws, and test the provided code."},
            "personality": {"communication_tone": "cheerful but sharp", "response_style": "analytical"},
            "rules": ["Be thorough in code reviews.", "Highlight any potential edge cases."],
            "capabilities": ["Security audit", "Code review", "Testing"],
            "tool_access": ["python_repl"]
        }
    )

    # 2. Group them into a Team Configuration
    team_config = MultiAgentConfig(
        team_name="Hashira-OS Task Force",
        agents=[architect_config, coder_config, reviewer_config]
    )

    # 3. Initialize the Manager
    manager = MultiAgentManager(team_config)

    # 4. Define the complex problem
    problem_statement = (
        "We need a Python implementation of a Thread-Safe LRU (Least Recently Used) Cache. "
        "It needs to have `get`, `put`, and `get_cache_size` methods. "
        "Please provide the full code and a security/performance review."
    )
    
    # 5. Run the collaborative pipeline!
    # Tengen creates the plan -> Giyu writes the code -> Shinobu reviews it.
    print("Launching Pipeline...")
    final_result = await manager.run_pipeline(
        prompt=problem_statement, 
        agent_sequence=["Tengen_Architect", "Giyu_Coder", "Shinobu_Reviewer"]
    )
    
    print("\n--- Final Output from Shinobu (Reviewer) ---")
    print(final_result)

if __name__ == "__main__":
    asyncio.run(run_hashira_team())
```

---

## 🏗️ Architecture

Every agent created by the manager follows the full **Phoenix Agent Architecture**:
- **Thinker:** Objective deconstruction.
- **Planner:** Multi-step tool selection.
- **Actor:** Parallel tool execution.
- **Reflector:** Success verification and learning.

By leveraging the `AgentProfile` system, each agent in the team maintains its unique personality and follows its specific set of rules during the entire collaboration.

---

## 📡 OS-Grade Communication Layer (Message Bus & State Store)

When building complex OS-level architectures (like Hashira-OS), passing raw strings between agents is dangerous. The `MultiAgentManager` provides a strict, typed communication infrastructure to guarantee reliability.

### 1. Structured Message Protocol (`AgentMessage`)
Agents communicate using structured JSON payloads rather than plain text. This allows agents to reliably trigger alerts, requests, and tasks based on strict priorities.

```python
from phoenix.framework import AgentMessage, MessageType, Priority

msg = AgentMessage(
    sender="Giyu_Monitor",
    receiver="Gyomei_Admin",
    channel="monitoring",
    msg_type=MessageType.ALERT,
    priority=Priority.CRITICAL,
    payload={"metric": "cpu_usage", "value": 95.2},
    summary="CPU usage exceeded safe threshold!"
)
```

### 2. Async Message Bus
The manager hosts a central pub/sub bus (`manager.bus`). Agents can subscribe to channels and listen for structured events.

```python
# 1. Subscribe Gyomei to the monitoring channel
manager.bus.subscribe("Gyomei_Admin", channel="monitoring")

# 2. Giyu publishes an alert (receiver="*" means broadcast to channel)
await manager.bus.publish(msg)

# 3. Start the event loop so the manager automatically routes the message
# to Gyomei's cognition loop!
await manager.run_event_loop()
```

### 3. Shared State Store
When agents need to share real-time OS facts (like CPU usage, active processes, system locks) without spamming the message bus, they use the `SharedStateStore`.

```python
# Giyu updates the real-time CPU usage
await manager.state_store.set("cpu_usage", 85.5, owner="Giyu_Monitor")

# Gyomei can instantly read it
cpu = await manager.state_store.get_value("cpu_usage")

# Reactive OS Pattern: Gyomei can WATCH the state and react instantly!
async def on_cpu_spike(entry):
    if entry.value > 90.0:
        print(f"ALERT: CPU spiked to {entry.value}%!")

manager.state_store.watch("cpu_usage", on_cpu_spike)
```

---

## 🔌 Registering Pre-Built Agents

If you have already built fully customized agents with their own cognition modules, custom tools, and custom loops, you can plug them directly into the `MultiAgentManager` **without** needing to use `AgentConfig` at all.

### Option 1: `from_agents()` (Recommended)

Create the manager directly from your existing agent instances:

```python
import asyncio
from phoenix.framework.multi_agent.manager import MultiAgentManager

# Import your custom agent factories
from giyu.agent import get_giyu_agent
from gyomei.agent import get_gyomei_agent

async def main():
    # 1. Initialize your pre-built agents as usual
    giyu = await get_giyu_agent()
    gyomei = await get_gyomei_agent()

    # 2. Create the manager directly from them
    manager = MultiAgentManager.from_agents({
        "Giyu": giyu,
        "Gyomei": gyomei
    }, shared_memory=True)  # Optional: share memory between them

    # 3. Use any orchestration pattern!
    result = await manager.run_pipeline(
        prompt="Refactor the auth module and audit it for security",
        agent_sequence=["Giyu", "Gyomei"]
    )

    # Or let the manager decide who should handle it
    result = await manager.run_autonomous("Check the server logs for anomalies")

    # Or use the self-correcting review loop
    result = await manager.run_with_review(
        prompt="Write a rate limiter middleware",
        doer="Giyu",
        reviewer="Gyomei",
        max_loops=3
    )

asyncio.run(main())
```

### Option 2: `register_agent()` (Dynamic)

Add agents to an existing manager at runtime:

```python
manager = MultiAgentManager()  # Create an empty manager

# Build and register agents one by one
giyu = await get_giyu_agent()
manager.register_agent("Giyu", giyu)

gyomei = await get_gyomei_agent()
manager.register_agent("Gyomei", gyomei)

# Remove agents dynamically
manager.remove_agent("Gyomei")
```

> [!IMPORTANT]
> Each agent keeps its own custom cognition (Thinker, Planner, Reflector, Loop), tools, and LLM configuration. The `MultiAgentManager` does **not** override any of these — it only orchestrates the communication between them.
