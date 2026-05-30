# Building Specialized Agents in Phoenix

The Phoenix framework makes it incredibly simple to build specialized autonomous agents. By modifying an agent's `AgentProfile` and providing it with a specific set of `Tools`, you can fundamentally change its behavior and capabilities.

Here are three common scenarios:

---

## Scenario 1: Call Center / Customer Support Assistant

A Call Center agent needs to be polite, empathetic, and possess the ability to interface with your internal business systems (like a CRM or order database). We do this by creating **Custom Tools** using the `@tool` decorator.

```python
import asyncio
from phoenix.main import init_phoenix
from phoenix.framework.agent import Agent
from phoenix.framework.agent.core.profile import AgentProfile, Identity, Role, Personality
from phoenix.framework.agent.tools.base import tool

# 1. Initialize the SDK
init_phoenix()

# 2. Define custom business tools for the Call Center
@tool(name="check_order_status", description="Fetches the shipping status of an order using the order_id.")
async def check_order_status(order_id: str) -> str:
    # In reality, this would query your SQL DB or Shopify API
    return f"Order '{order_id}' is currently Out for Delivery."

@tool(name="issue_refund", description="Initiates a refund for a specific order_id.")
async def issue_refund(order_id: str, reason: str) -> str:
    return f"Refund processed successfully for order '{order_id}'. Reason logged: {reason}."

# 3. Create the Persona Profile
profile = AgentProfile(
    identity=Identity(name="SupportBot", id="agent-support-1"),
    role=Role(title="Customer Support Representative", mission="Empathize with users and interface with internal business systems to resolve customer queries."),
    personality=Personality(communication_tone="Polite and warm", response_style="Empathetic and structured"),
    rules=[
        "Always greet the user warmly.",
        "Before discussing shipping or refunds, you must ask the user for their Order ID.",
        "Only use tools when you have the necessary information."
    ]
)

# 4. Instantiate the Agent
agent = Agent(profile=profile, tools=[check_order_status, issue_refund])

# Run it
async def run():
    response = await agent.run("Hi, where is my order? My ID is #99281")
    print(response)

asyncio.run(run())
```

---

## Scenario 2: Data Analysis Agent

A Data Analysis agent doesn't need external web APIs, but it *does* need the ability to write and execute Python code, read datasets, and save output charts. Phoenix provides these built-in tools out of the box.

```python
import asyncio
from phoenix.main import init_phoenix
from phoenix.framework.agent import Agent
from phoenix.framework.agent.core.profile import AgentProfile, Identity, Role, Personality
from phoenix.framework.agent.tools.code import CodeExecutionTool
from phoenix.framework.agent.tools.io import FileReadTool, FileWriteTool

init_phoenix()

# 1. Create the Persona Profile
profile = AgentProfile(
    identity=Identity(name="DataAnalyst", id="agent-analyst-1"),
    role=Role(title="Senior Data Scientist", mission="Analyze datasets, write pandas operations, and generate summary reports."),
    personality=Personality(communication_tone="Analytical and professional", response_style="Concise and fact-based"),
    rules=[
        "Write and execute Python code using your python_repl tool to perform Pandas operations.",
        "After analyzing, write your final reports to a markdown file."
    ]
)

# 2. Instantiate with Code & IO Tools
agent = Agent(
    profile=profile,
    tools=[
        CodeExecutionTool(), # Allows running isolated python code
        FileReadTool(),      # Allows reading CSV/JSON files
        FileWriteTool()      # Allows saving reports
    ]
)

# Run it
async def run():
    await agent.run("Can you read 'sales_data.csv', calculate the average revenue per region, and save the summary to 'report.md'?")

asyncio.run(run())
```

---

## Scenario 3: Autonomous Coder / Developer Agent

A Software Engineer agent requires deep access to the project's file system, the ability to execute terminal commands (like running `pytest`), and advanced tools to patch/edit code files safely.

```python
import asyncio
from phoenix.main import init_phoenix
from phoenix.framework.agent import Agent
from phoenix.framework.agent.core.profile import AgentProfile, Identity, Role, Personality

# Import all advanced development tools
from phoenix.framework.agent.tools.code import CommandExecutionTool, PythonAnalyzerTool, CodeCompileTool
from phoenix.framework.agent.tools.io import FileReadTool, FileEditTool, FileSearchTool
from phoenix.framework.agent.tools.patch import MultiBlockUpdateTool

init_phoenix()

# 1. Create the Persona Profile
profile = AgentProfile(
    identity=Identity(name="SeniorDev", id="agent-dev-1"),
    role=Role(title="Principal Software Engineer", mission="Review codebase, modify source files, and run tests to implement new features."),
    personality=Personality(communication_tone="Professional", response_style="Technical and direct"),
    rules=[
        "Before modifying code, search the workspace and read existing files to understand the architecture.",
        "Make code edits carefully, run tests using the terminal tool to verify your changes, and fix any bugs you encounter."
    ]
)

# 2. Instantiate with a full suite of Developer Tools
agent = Agent(
    profile=profile,
    tools=[
        CommandExecutionTool(), # Run bash/shell commands (e.g. pytest)
        FileSearchTool(),       # Grep functionality
        FileReadTool(),         # Read file contents
        MultiBlockUpdateTool(), # Safe code-patching tool
        FileEditTool(),         # Overwrite or append files
        PythonAnalyzerTool(),   # AST analyzer for mapping classes/functions
        CodeCompileTool()       # Syntax checking
    ]
)

# Run it
async def run():
    await agent.run("Please add a new user authentication endpoint to the FastAPI app in src/api.py, then run the tests to ensure it works.")

asyncio.run(run())
```

---

### Summary

The flexibility of the framework comes down to **Tool Selection**. 

* Need an agent to browse the internet? Give it the `WebSearchTool()`. 
* Need an agent to interact with a smart home? Write a custom `@tool` that triggers your IoT API.
* Need the agent to speak? Connect it to your Audio pipeline tools. 

Because the `AgentLoop` automatically parses the schemas for any tool you pass in the `tools=[]` list, the LLM will intuitively figure out how to use them to accomplish the objective.

