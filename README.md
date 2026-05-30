# Phoenix AI

<p align="center">
  <img src="https://raw.githubusercontent.com/blackeagle686/phoenix-ai/master/docs/statics/images/phx-light-circle.png" alt="Phoenix AI Logo" width="500">
</p>

<p align="center">
  <strong>Advanced AI Infrastructure SDK for Autonomous Agents, Chatbots, and Production Backend Services.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License: MIT">
  <img src="https://img.shields.io/badge/framework-FastAPI%20%7C%20Django-red.svg" alt="Frameworks">
  <img src="https://img.shields.io/badge/AI-Agents%20%7C%20RAG-orange.svg" alt="AI SDK">
</p>

---

Whether you are building with FastAPI, Django, or a custom event-driven service, **Phoenix AI** eliminates repetitive backend setup and provides a highly-optimized orchestration layer for large language models, computer vision, and physical hardware.

## Why Phoenix AI?

- **Autonomous Agents**: Single-line creation of thinking, planning, and executing AI agents.
- **High-Level ChatBot**: Turnkey conversational AI with native RAG, Vision, and Memory.
- **"Everything as a Service"**: A unified Dependency Injection container handles Vector DBs, Redis caching, and LLMs seamlessly.
- **Fail-Loud & Recover**: Auto-fallbacks from Local (Ollama/Transformers) to Cloud (OpenAI) to prevent system crashes.
- **Native PyTorch & 4-bit Quantization**: Run local models directly on your GPU without external servers, automatically optimized for low-VRAM machines.
- **Sensorium (Embodied AI)**: Plug your AI directly into the physical world via IoT, MQTT, and Arduino plugins.

---

## 📦 Installation

Choose the installation tier that matches your project needs:

```bash
# Core Backend Framework
pip install phx-ashborn

# Core + ChatBot & Conversation Memory
pip install "phx-ashborn[chatbot]"

# Core + Autonomous Agents & Planners
pip install "phx-ashborn[agent]"

# Full Suite (Everything including Local AI Inference)
pip install "phx-ashborn[full]"
```

> [!NOTE]
> If you're running locally, copy the `.env.example` to `.env` and configure your API keys (e.g., `OPENAI_API_KEY`). For full system deployment (including Redis setup), use the provided `./install.sh` or `install.bat` scripts.

---

##  Quickstarts & Core Features

### 1.  Autonomous Agents
The Phoenix Agent is a high-speed cognitive engine capable of understanding complex problems, scanning codebases, planning multi-step solutions, and executing parallel tools.

```python
import asyncio
from phoenix import Agent

async def agent_demo():
    # Initialize a high-speed Agent with default tools and memory
    agent = Agent()
    
    # Or inject custom tools instantly!
    from phoenix.framework.agent import tool
    
    @tool(name="custom_math", description="Calculates squares. Input: 'number' (int)")
    def math_tool(number: int):
        return f"The square is {number ** 2}"
        
    agent.register_tool(math_tool)
    
    # Run a complex engineering task
    # The agent will: Think -> Analyze -> Plan -> Execute Tools -> Reflect
    result = await agent.run(
        "Find the redundant code in the memory module and optimize it.", 
        mode="plan"
    )
    
    print(f"Agent Execution Report: {result}")

asyncio.run(agent_demo())
```

> [!TIP]
> Phoenix Agents feature **Intelligent Auto-Routing**. By default (`mode="auto"`), the agent analyzes your prompt to decide whether to give a blazing-fast direct answer or spin up its heavy planning loop for complex operations!

### 2. Multi-Modal ChatBot
Need a powerful conversational interface without the complexity of building agent loops? The `ChatBot` builder abstracts away RAG, Vision (VLM), and Session Memory into a fluent API.

```python
from phoenix import ChatBot
import asyncio

async def chatbot_demo():
    # Build a complete AI ChatBot in one line
    bot = (ChatBot(local=False, vlm=True)
           .with_rag(["./docs", "./src"])       # Ingest folders automatically
           .with_memory()                       # Enable conversation history
           .with_security(mode="strict")        # Prompt-injection protection
           .with_system_prompt("You are a helpful Python expert.")
           .build())

    # Multi-modal interaction out of the box
    response = await bot.chat(
        prompt="Explain this architecture diagram.", 
        image_path="architecture.png"
    )
    print(response)

asyncio.run(chatbot_demo())
```

### 3. RAG Pipeline (Retrieval-Augmented Generation)
The `RAGPipeline` handles document extraction, intelligent chunking, and vector storage (ChromaDB/Qdrant) across PDFs, Code files, SQL, APIs, and GitHub repos.

```python
import asyncio
from phoenix import init_phoenix, startup_phoenix, get_rag_pipeline

async def rag_demo():
    # Initialize the core framework services
    init_phoenix()
    await startup_phoenix()
    
    rag = get_rag_pipeline()

    # 1. Ingest local directories (supports .pdf, .docx, .py, .go, etc.)
    await rag.ingest("./my_project")

    # 2. Ingest remote web pages
    await rag.ingest_url("https://example.com/api-docs")
    
    # 3. Clone and index a GitHub repository on the fly
    await rag.ingest_github("https://github.com/blackeagle686/phoenix-ai.git")

    # 4. Query the knowledge base (Automatic Source Citations included!)
    answer = await rag.query("How do I extend the caching layer?")
    print(answer)

asyncio.run(rag_demo())
```

---

## 🏗️ Advanced Architecture

###  Multi-Agent Orchestration
Define dynamic teams of specialized agents (e.g., Coder, Reviewer, Security Expert) and coordinate them through parallel broadcasting or sequenced pipelines.

```python
from phoenix.framework import MultiAgentManager, MultiAgentConfig, AgentConfig

config = MultiAgentConfig(
    team_name="Engineering Task Force",
    agents=[
        AgentConfig(name="Giyu_Coder", profile="profiles/coder.json"),
        AgentConfig(name="Shinobu_Reviewer", profile="profiles/reviewer.json")
    ]
)

manager = MultiAgentManager(config)
report = await manager.run_pipeline(
    prompt="Implement a thread-safe cache system",
    agent_sequence=["Giyu_Coder", "Shinobu_Reviewer"]
)
```

### 🦾 Sensorium (Hardware SDK)
Connect your Phoenix Agents to the physical world using an async, zero-latency plugin architecture. Build Smart Home routines, Robotics controllers, or Drone surveillance systems!

```python
from phoenix.framework.sensorium.core.manager import DeviceManager
from phoenix.framework.sensorium.plugins.mock_plugin import MockSensorPlugin
from phoenix.framework.agent import tool

manager = DeviceManager()
await manager.add_device("living_room_temp", MockSensorPlugin())

@tool(name="get_temperature", description="Reads current room temp.")
async def read_temp():
    return await manager.get_device("living_room_temp").read()
```

---

## ⚠️ Local Inference Requirements

If you run Phoenix using **Local LLMs/VLMs** (via Ollama or native Transformers), ensure your machine meets the following specifications to prevent system instability:

- **RAM**: 8GB Minimum (16GB+ recommended).
- **GPU**: 4GB+ VRAM required for Vision/VLM models (utilizing built-in 4-bit quantization).
- **Disk Space**: 10GB+ free space for model weights.

> [!WARNING]
> High-resource models may cause system crashes on CPU-only devices. The SDK prioritizes stability and will pause to prompt for user confirmation in the terminal before booting large local providers.

---

## 📚 Comprehensive Documentation

Ready to dive deeper? Explore our dedicated guides to master the Phoenix ecosystem:

### 🧩 Core Architecture
- **[Main Framework Guide](docs/GUIDE.md)** ([Arabic Version](docs/GUIDE.ar.md))
- **[Data Pipelines & RAG](docs/PIPELINES.md)** ([Arabic Version](docs/PIPELINES.ar.md))
- **[Model Training & Finetuning](docs/TRAINING.md)** ([Arabic Version](docs/TRAINING.ar.md))

### 🤖 Autonomous Agents
- **[Agent Framework Architecture](docs/AGENT_GUIDE.md)**
- **[Multi-Agent Teams](docs/framework/multi_agent.md)**
- **[The Phoenix Tool Bank](docs/tool_bank_usage.md)**
- **[Custom Specialized Agents](docs/different_scenarios_usage.md)**

### 🔌 Integrations & Extensions
- **[Django Integration](docs/DJANGO_INTEGRATION.md)**
- **[FastAPI / Backend Integration](docs/API_INTEGRATION.md)**
- **[GUI App Integration](docs/GUI_INTEGRATION.md)**
- **[Sensorium Embodied AI (Hardware)](docs/sensorium.md)**
