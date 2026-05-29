# 🐦‍🔥 Phoenix AI (Advanced AI Infrastructure SDK)

<p align="center">
  <img src="https://raw.githubusercontent.com/blackeagle686/phoenix-ai/master/docs/statics/images/phx-light-circle.png" alt="Phoenix AI Logo" width="500">
</p>

A production-ready, modular backend infrastructure SDK designed for AI-powered Python backend services. 

Whether you are building with FastAPI, Django, or a custom event-driven service, **Phoenix AI** eliminates repetitive backend setup.

```bash
pip install phx-ashborn
# Or for advanced features:
pip install "phx-ashborn[agent]"
# Or for chat bot features:
pip install "phx-ashborn[chatbot]"
# Or for all features:
pip install "phx-ashborn[full]"
```

## 🐦‍🔥 Key Requirements & Core Features

1. **Dependency Injection**: Central standard registry. No manual instantiation inside business logic.
2. **Interface First**: Every module complies with an asynchronous base contract (`BaseCache`, `BaseLLM`, `BaseVectorDB`, etc.).
3. **Flexible Vector DB**: Native support for **ChromaDB** (Default/Persistent) and **Qdrant**.
4. **Embedded Insights**: Pre-configured with `sentence-transformers` (`all-MiniLM-L6-v2`) for local embedding generation.
5. **RAG Orchestration**: All-in-one `RAGPipeline` that handles document loading (.pdf, .docx, .xlsx), SQL databases, external APIs, and web scraping.
6. **Sensorium Hardware SDK**: (NEW) Native infrastructure for building embodied AI agents that interact with real hardware (Serial, MQTT, GPIO, Cameras).

## 📦 Installation
Choose the method that fits your workflow best.

### 1. Automated Installation (Recommended)
Get everything ready in one command (handles Python deps and Redis setup):
```bash
# For Linux/macOS/WSL
chmod +x install.sh
./install.sh

# For Windows
install.bat
```

### 2. Manual Installation
Alternatively, use the provided `Makefile` or `pip`:
```bash
# Full installation with all services (VDB, RAG, Memory, etc.)
make install-full

# Or basic installation
make install
```

### 3. Pip Installation (Official)
```bash
pip install phx-ashborn

# Or with full local model support
pip install "phx-ashborn[full]"
```
3. **Configure Environment Variables**:
   Copy the provided template and add your keys:
   ```bash
   cp .env.example .env
   ```
   Edit `.env` with your settings:
   ```env
   OPENAI_API_KEY="your_key"
   REDIS_URL="redis://localhost:6379/0" 
   # See .env.example for more advanced options
   ```

### 🛠️ System Dependencies
- **Redis Server**: Required for stateful memory and caching.
  - Ubuntu: `sudo apt install redis-server`
  - macOS: `brew install redis`

## 🐦‍🔥 Framework Mode: High-Level ChatBot

The Phoenix AI SDK now includes a high-level **Framework Layer** that allows you to build complex AI agents with Vision, Speech, RAG, and Memory in just **one line of code**.

```python
from phoenix import ChatBot

# Build the complete AI Agent with advanced RAG tuning
bot = (ChatBot(local=True, vlm=True)
       .with_rag(
           ["./docs", "./src"],
           chunk_size=500,
           reranking=True,        # Better accuracy
           fast_rag=True,         # Faster retrieval
           cag=True,              # Context-Augmented Generation
           hybrid_search=True     # Vector + Keyword search
       )
       .with_memory()                       # Enable session memory
       .with_security(mode="strict")        # Protection against Prompt Injection
       .with_system_prompt("Expert Dev")    # Guide bot behavior
       .build())

# Or switch to OpenAI with one line
# bot.with_openai(api_key="sk-...", base_url="https://api.openai.com")

# Multi-modal interaction
response = await bot.chat("What's in this image?", image_path="vision.jpg")
print(response) 
```
> [!TIP]
> Use `.set_session("user_123")` on the bot instance to switch between different users in production environments like FastAPI.

## 🐦‍🔥 Framework Mode: Autonomous Agent

The Phoenix AI SDK now supports creating a fully autonomous agent that can think, analyze, plan, execute tools, and reflect on its progress with a single line of code! 

> [!TIP]
> For a deep dive into the architecture and integration patterns, check out the **[Agent Framework Guide](https://github.com/blackeagle686/phoenix-ai/blob/master/docs/AGENT_GUIDE.md)**, **[Sensorium Hardware Guide](docs/sensorium.md)**, **[Multi-Agent Guide](docs/framework/multi_agent.md)**, **[Django Integration Guide](https://github.com/blackeagle686/phoenix-ai/blob/master/docs/DJANGO_INTEGRATION.md)**, **[GUI Integration Guide](https://github.com/blackeagle686/phoenix-ai/blob/master/docs/GUI_INTEGRATION.md)**, or the **[API Integration Guide](https://github.com/blackeagle686/phoenix-ai/blob/master/docs/API_INTEGRATION.md)**.

#### ⚡ High-Speed Cognitive Engine
*   **Parallel Awareness**: The `Thinker` and `Analyzer` run concurrently, allowing the agent to understand both your prompt and your project structure in a single cognitive step.
*   **Multi-Action Planning**: The agent can plan and execute multiple independent actions (tools) in parallel, cutting task completion time by up to 60%.
*   **Concurrent Memory**: Reflection, consolidation, and logging happen in the background, ensuring zero-latency transitions between agent steps.
*   **Hybrid Memory Layer**: Integrated `ShortTerm`, `LongTerm` (Vector), `Session`, and `Reflection` memories with parallel retrieval support.

```python
import asyncio
from phoenix import Agent

async def agent_demo():
    # Initialize the high-speed Agent
    agent = Agent() # Uses default LLM, Hybrid Memory, and Parallel Tools
    
    # Run a complex engineering task
    # The agent will automatically:
    # 1. Think: Deconstruct the prompt
    # 2. Analyze: Scan the repo structure and tech stack
    # 3. Plan: Create parallel steps for search and code analysis
    # 4. Act: Execute tools concurrently (e.g. searching while analyzing code)
    # 5. Reflect: Verify the fix and learn from the process
    
    prompt = "Find the redundant code in the memory module and optimize it using the new parallel patterns."
    result = await agent.run(prompt, mode="plan")
    
    print(f"Agent Engineering Report: {result}")
```

## 🐦‍🔥 Framework Mode: Multi-Agent Teams

The Phoenix AI SDK now supports **Multi-Agent Orchestration**. You can define teams of agents (e.g., Coder, Reviewer, Security Expert) and have them work together in parallel or through sequenced pipelines.

*   **Parallel Broadcasting**: Send a prompt to the entire team and gather concurrent responses.
*   **Sequenced Pipelines**: Chain agents together where the output of one agent becomes the input for the next (e.g., Code → Review → Secure).
*   **Targeted Execution**: Invoke specific agents by their role or name within the team.

```python
from phoenix.framework import MultiAgentManager, MultiAgentConfig, AgentConfig

# 1. Define a team with specific profiles
config = MultiAgentConfig(
    team_name="DevTeam",
    agents=[
        AgentConfig(name="Giyu", profile="profiles/coder.json"),
        AgentConfig(name="Shinobu", profile="profiles/reviewer.json")
    ]
)

# 2. Orchestrate a pipeline
manager = MultiAgentManager(config)
final_report = await manager.run_pipeline(
    prompt="Implement a thread-safe cache",
    agent_sequence=["Giyu", "Shinobu"]
)
```
## 🐦‍🔥 Framework Mode: Sensorium (Hardware SDK)

The Phoenix AI SDK now supports **Embodied AI**. You can connect your agents to real-world sensors, cameras, and actuators using a unified plugin-based architecture.

*   **Unified Device Communication**: Standardized interfaces for Serial (Arduino/ESP32), MQTT (IoT), and GPIO.
*   **Real-Time Streaming**: Optimized frame-dropping buffers for vision processing on the edge (Raspberry Pi / Jetson).
*   **Agent Integration**: Register hardware as tools that agents can autonomously use in their reasoning loops.

```python
from phoenix.framework.sensorium.core.manager import DeviceManager
from phoenix.framework.sensorium.plugins.mock_plugin import MockSensorPlugin

# 1. Initialize Hardware Manager
manager = DeviceManager()

# 2. Add and connect a sensor
await manager.add_device("main_thermometer", MockSensorPlugin())

# 3. Register as a tool for the Agent
@tool(name="get_temp", description="Reads current temperature")
async def get_temp():
    return await manager.get_device("main_thermometer").read()

agent.register_tool(get_temp)
await agent.run("Is it too hot in the living room?")
```
> [!NOTE]
> Sensorium is designed for performance-critical tasks. It uses Python `__slots__` and thread-safe protocols to ensure zero-latency interaction with hardware. For more details, see the **[Sensorium Hardware Guide](docs/sensorium.md)**.

### 🐦‍🔥 Custom Tools & Engineering Suite

The Agent comes pre-configured with a suite of engineering-grade tools:
*   **`python_analyzer`**: (High-Speed) AST-based indexing of classes and functions for precise code navigation.
*   **`file_update_multi`**: (Atomic) Applies multiple code changes across different parts of a file in one go.
*   **`python_repl`**: Executes Python logic in a sandbox.
*   **`web_search`**: Live internet access for news and documentation.
*   **`file_read / file_write / file_search`**: Advanced filesystem operations.

You can also easily create and inject your own custom tools using the `@tool` decorator:

```python
from phoenix.framework.agent import tool

# 1. Define your custom logic
@tool(name="custom_math", description="Calculates the square of a given number. Input: 'number' (int).")
def custom_math_tool(number: int):
    return f"The square of {number} is {number ** 2}"

# 2. Register it directly to the agent
agent.register_tool(custom_math_tool)

# 3. The agent can now autonomously use 'custom_math' in its planning!
await agent.run("What is the square of 12?")
```

### ⚡ Execution Modes (Auto-Routing)

The Agent features intelligent routing to save time and API costs on simple tasks. By default, it runs in `mode="auto"`.
- **`auto`**: The agent analyzes the prompt. If it's a simple question, it gives a direct answer. If it requires tools or multi-step logic, it spins up the planning loop.
- **`fast_ans`**: Forces the agent to skip planning and answer immediately using memory context.
- **`plan`**: Forces the agent into the rigorous `Think -> Plan -> Act -> Reflect` loop.

```python
# Forces a fast answer (Bypasses tool execution)
await agent.run("Hi, who are you?", mode="fast_ans")

# Forces complex planning
await agent.run("Search the web for the latest Python release...", mode="plan")
```

## 📖 Quickstart: RAG Pipeline

The `RAGPipeline` is the highest-level service for handling document-based knowledge.

```python
import asyncio
from phoenix import init_phoenix, startup_phoenix, get_rag_pipeline

async def rag_demo():
    # One-liner to initialize and get the pipeline
    rag = get_rag_pipeline()

    # 1. Ingest documents (Supports Docs + Source Code .py, .js, .go, .rs, etc.)
    await rag.ingest("./my_project")

    # 2. Ingest from GitHub Repository (Automated cloning & indexing)
    await rag.ingest_github("https://github.com/blackeagle686/phoenix-ai.git")

    # 3. Ingest from Web URL
    await rag.ingest_url("https://example.com/docs/api")

    # 4. Query with automatic Citations
    answer = await rag.query("How do I extend the cache layer?")
    print(f"AI Answer: {answer}")
```

### 🐦‍🔥 Source Attribution
The SDK now automatically instructs the LLM to cite its sources. When you query the RAG pipeline, the response will often include markers like `[Source: cloud.pdf]` or `[Source: https://example.com]`.

## ⚠️ Local Model Hardware Requirements
If you plan to use local inference (Ollama or Transformers), please ensure your system meets these specifications:
- **RAM**: 8GB Minimum (16GB+ recommended).
- **GPU**: 4GB+ VRAM required for VLM models (using 4-bit quantization).
- **Disk**: 10GB+ free space for model storage.

> [!WARNING]
> High-resource models may cause system instability on low-RAM or CPU-only devices. The SDK defaults to a safety-first approach and will prompt for confirmation before starting local providers.

## 🐦‍🔥 Dynamic Fallbacks & Native PyTorch

phoenix includes a robust "fail-loud and recover gracefully" orchestration architecture for AI providers:

### 1. Interactive Provider Fallbacks
If your primary provider (e.g. Local) fails to connect or crashes, the SDK's orchestration (`VLMPipeline` / `InsightEngine`) instantly intercepts the failure and prompts you to fallback to the secondary provider (e.g. OpenAI), bypassing pipeline crashes.

### 2. Native PyTorch Singleton Caching (`LocalVLM` & `LocalLLM`)
No `Ollama` server? No problem! The local providers automatically detect if Hugging Face `transformers` is installed and spin up models natively in your local GPU using an optimized Singleton cache.
> **Jupyter/Colab Tip**: If you face persistent `Ollama` warnings after installing `transformers`, run `LocalVLM._model_cache.clear()` or `LocalLLM._model_cache.clear()` in your notebook to wipe the previous state and force a PyTorch native reload.

### 3. Automatic 4-Bit Quantization
To prevent `CUDA Out of Memory` (OOM) errors on smaller GPUs (like Colab T4s), the SDK auto-detects `bitsandbytes` (`pip install bitsandbytes`) and instantly applies `load_in_4bit=True` to shrink massive models (like Qwen2-VL) into your VRAM.

### 4. Resilient RAG PDFs
The `RAGPipeline.ingest()` method supports PDFs robustly by sequentially testing for parsing libraries: `pypdf`, `pymupdf` (`fitz`), `pdfplumber`, and `PyPDF2`. Simply install whichever you prefer (`pip install pymupdf` is recommended for speed) and it works flawlessly!

## 🐦‍🔥 Advanced Usage: Insight Engine

The `InsightEngine` performs full context retrieval, query rewriting, and LLM generation efficiently.

```python
from phoenix import init_phoenix, get_insight_engine

async def insight_demo():
    # High-level retrieval engine
    insight = get_insight_engine()

    # This invokes: Clean Query -> Vector Search -> Rerank -> LLM Generation
    final_response = await insight.query("How do I extend the cache layer?")
    print(final_response)
```

## 🖼️ Quickstart: VLM (Vision)

The `VLMPipeline` orchestrates vision tasks with automatic caching and RAG.

```python
from phoenix import init_phoenix_full, get_vlm_pipeline

async def vision_demo():
    # Integrated Vision-Language Pipeline
    vlm = get_vlm_pipeline()

    # Integrated: Result Caching + RAG context injection
    answer = await vlm.ask("What is in this image?", "image.png", use_rag=True)
    print(answer)
```

## 📚 Comprehensive Documentation

The Phoenix SDK is heavily documented. Check out our dedicated guides to master different aspects of the framework:

### Core Framework
- **[Main Framework Guide](docs/GUIDE.md)** ([Arabic Version](docs/GUIDE.ar.md))
- **[Data Pipelines & RAG](docs/PIPELINES.md)** ([Arabic Version](docs/PIPELINES.ar.md))
- **[Model Training & Finetuning](docs/TRAINING.md)** ([Arabic Version](docs/TRAINING.ar.md))

### Autonomous Agents
- **[Agent Framework Architecture](docs/AGENT_GUIDE.md)**
- **[Multi-Agent Teams](docs/framework/multi_agent.md)**
- **[The Phoenix Tool Bank & Usage Examples](docs/tool_bank_usage.md)**
- **[Building Custom Specialized Agents](docs/different_scenarios_usage.md)**

### Integration & Hardware
- **[Django Integration Guide](docs/DJANGO_INTEGRATION.md)**
- **[FastAPI / API Integration Guide](docs/API_INTEGRATION.md)**
- **[GUI App Integration Guide](docs/GUI_INTEGRATION.md)**
- **[Sensorium Hardware SDK Guide](docs/sensorium.md)**
