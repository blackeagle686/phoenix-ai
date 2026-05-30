# Phoenix AI SDK: Autonomous Agent Guide

The Phoenix Agent framework provides a state-of-the-art, schema-compliant, autonomous loop architecture driven by **Plan-Execute-Reflect** cycles. It supports structured tool schema execution, Redis short-term persistence, vector DB long-term memory, and custom cognition pipelines.

---

## Getting Started

### 1. Minimal Initialization (Default Settings)
By default, the Agent uses `OpenAILLM` (via the environment config `OPENAI_API_KEY` and base URL), registers default File I/O tools, and starts a memory-resident context window backed by Redis:

```python
import asyncio
from phoenix.framework.agent import Agent

async def main():
    # Instantiates the agent with default configuration
    agent = Agent()
    
    # Run the autonomous execution loop
    response = await agent.run(
        prompt="Write a Python script to './test.py' that prints 'Hello Phoenix'. Then read the file to verify it.",
        session_id="session_001"
    )
    print(f"Agent Final Report:\n{response}")

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 🛠️ Tool System & Custom Tool Creation

All agent tools use Pydantic models to enforce strict parameter typing, preventing prompt-injection parameter errors.

### Registering a Custom Tool

You can register custom tools using either the `@tool` decorator (cleanest) or by constructing a `Tool` object explicitly.

#### A. Using the `@tool` Decorator (Recommended)
Import the `@tool` decorator, decorate your function with name/description details, and register it directly:

```python
import asyncio
from phoenix.framework.agent import Agent
from phoenix.framework.agent.tools import tool

# 1. Define custom tool using the @tool decorator
@tool(name="custom_math", description="Calculates the square of a given number. Input: 'number' (int).")
def custom_math_tool(number: int):
    return f"The square of {number} is {number ** 2}"

async def main():
    agent = Agent()
    
    # 2. Register the tool
    agent.register_tool(custom_math_tool)
    
    # 3. Run the agent
    reply = await agent.run("Calculate the square of 9.")
    print(reply)

if __name__ == "__main__":
    asyncio.run(main())
```

#### B. Subclassing `BaseTool` (OOP Style)
For more complex tools that require initialization logic or state management, you can inherit from `BaseTool` directly:

```python
from phoenix.framework.agent import Agent
from phoenix.framework.agent.tools import BaseTool, ToolResult

class UserRoleTool(BaseTool):
    name = "check_user_role"
    description = "Verify user permissions. Input: 'username' (str)."

    async def execute(self, username: str, **kwargs) -> ToolResult:
        try:
            role = "Administrator" if username == "admin" else "Standard User"
            return ToolResult(success=True, output=role)
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))

agent = Agent()
agent.register_tool(UserRoleTool())
```

---

## ⚡ Execution Modes

### 1. Standard Execution (`run`)
Performs standard blocking loop execution, returning a structured markdown report containing the tool actions executed and their statuses:

```python
report = await agent.run("Create a new directory './data'")
```

### 2. Streaming execution (`run_stream`)
Yields real-time events, such as status indicators (thinking, executing specific tools, reflecting) and response text chunks:

```python
async for event in agent.run_stream("Create 'config.json'"):
    event_type = event.get("type") # "status" | "chunk"
    content = event.get("content")
    
    if event_type == "status":
        print(f"\n[Status Change]: {content}")
    elif event_type == "chunk":
        print(content, end="", flush=True)
```

---

## 🧠 Memory Persistence (Redis & Vector DB)

The Phoenix agent uses a **Hybrid Memory System**:
- **Short-Term (STM)**: Tracks immediate conversation turns. Stored in Redis cache under `stm[{session_id}]`.
- **Long-Term (LTM)**: Stores permanent logs and reflections in Chroma/Qdrant databases.
- **Session State**: Stores temporary execution variables under `session[{session_id}]` in Redis.
- **Reflections**: Stores loop lessons-learned under `reflection[{session_id}]` in Redis.

If Redis or the Vector database is offline, the memory managers automatically fallback to memory-resident dictionaries, ensuring seamless operation.

---

## 🏗️ Custom Cognition Pipelines

You can easily customize or extend the default Plan-Execute-Reflect pipeline by registering custom cognitive brain handlers:

```python
from phoenix.framework.agent import Agent

agent = Agent()

# 1. Define custom cognitive step logic
def security_scan_handler(state: dict) -> dict:
    """Scans planner target paths for unsafe operations."""
    print("Performing security audit...")
    planner_output = state.get("planner_output", {})
    # Manipulate or inspect state
    return state

# 2. Register step inside loop
agent.register_brain("security_scan", security_scan_handler)

# 3. Supply custom cognition schema JSON to run
agent.set_cognition_pipeline(
    bootstrap_pipeline_path="./custom_bootstrap.json"
)
```
