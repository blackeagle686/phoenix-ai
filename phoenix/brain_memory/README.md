# BrainMemory Engine Architecture

This document outlines the architectural design of the **BrainMemory** system, an advanced memory engine developed in Rust and integrated with Python via PyO3. The primary objective of this engine is to transform static database-driven memory into a "living brain" that interacts in real-time, processes thoughts in the background, and distills wisdom over time.

---

## Core Components

1. **Neural Spiking Event Bus:**
   - Functions as an ultra-fast, lock-free conduit for ingesting events and perceptions instantaneously.
   - Utilizes `crossbeam` to instantiate a dedicated background thread that continuously listens for and routes memory spikes without blocking the main AI execution loop.

2. **Subconscious Dream Engine:**
   - An asynchronous processing environment built on the `Tokio` runtime.
   - Activates when the AI agent enters a "sleep" or idle state. It correlates past events, calculates semantic similarities, and extracts new patterns to synthesize high-level cognitive nodes (`Knowledge`).

3. **Fractal Compressor:**
   - An intelligent time-series decay algorithm.
   - Compresses aging events that have lost their granular relevance, transforming them into "Abstract Rules". This preserves the core lessons learned while significantly reducing data bloat.

---

## 1. High-Level Architecture

The following diagram illustrates the interaction between the Python environment (AI Agent) and the Rust backend (Memory Engine).

```mermaid
graph TD
    %% Define Colors and Styles
    classDef python fill:#3776AB,stroke:#fff,stroke-width:2px,color:#fff;
    classDef rustCore fill:#DEA584,stroke:#fff,stroke-width:2px,color:#000;
    classDef neural fill:#E74C3C,stroke:#fff,stroke-width:2px,color:#fff;
    classDef dream fill:#8E44AD,stroke:#fff,stroke-width:2px,color:#fff;
    classDef fractal fill:#2980B9,stroke:#fff,stroke-width:2px,color:#fff;
    classDef memoryDB fill:#2C3E50,stroke:#fff,stroke-width:2px,color:#fff;
    classDef api fill:#27AE60,stroke:#fff,stroke-width:2px,color:#fff;

    %% Python Layer
    subgraph "Python Layer (Agent)"
        Agent["AI Agent / LLM"]:::python
        PyO3["PyO3 Bridge"]:::api
    end

    %% Rust Layer - BrainMemory
    subgraph "Rust Layer (BrainMemory Engine)"
        BrainMemory["BrainMemory (Main Controller)"]:::rustCore
        
        NeuralBus["Neural Bus\n(Crossbeam Lock-Free)"]:::neural
        DreamEngine["Dream Engine\n(Tokio Async Thread)"]:::dream
        FractalComp["Fractal Compressor\n(Time Decay)"]:::fractal
        
        BrainMemory -->|Arc Reference| NeuralBus
        BrainMemory -->|Arc Reference| DreamEngine
        BrainMemory -->|Arc Reference| FractalComp
    end

    %% Storage Layer
    subgraph "Storage & Models"
        ShortTerm[("Short-term Memory\n(Events/Perceptions)")]:::memoryDB
        LongTerm[("Long-term Memory\n(Knowledge/Wisdom)")]:::memoryDB
    end

    %% Data Flow
    Agent <-->|Read / Write| PyO3
    PyO3 <-->|FFI Calls| BrainMemory
    
    NeuralBus -->|Spikes instantly| ShortTerm
    DreamEngine -->|Background Processing| ShortTerm
    DreamEngine -->|Consolidates into| LongTerm
    FractalComp -->|Compresses| ShortTerm
    FractalComp -->|Generates| LongTerm
```

---

## 2. Memory Lifecycle & Workflows

The following sequence details how an event transitions from immediate occurrence to consolidated wisdom over time.

```mermaid
sequenceDiagram
    autonumber
    
    actor Agent as Python AI Agent
    participant Bridge as BrainMemory (Rust)
    participant Bus as Neural Bus ⚡
    participant Dream as Dream Engine 💤
    participant Fractal as Fractal Compressor 🌀
    participant DB as Memory Storage 🗄️

    %% Real-time Interaction
    Agent->>Bridge: Agent observes stimuli (Perception)
    Bridge->>Bus: Fire Perception Spike!
    Bus-->>DB: Instantly route to Short-Term Storage
    
    Agent->>Bridge: Agent executes action (Event)
    Bridge->>Bus: Fire Event Spike!
    Bus-->>DB: Instantly route to Short-Term Storage
    
    %% Sleep Cycle Transition
    note over Agent, DB: ... Time Passes / Agent enters Idle State ...
    
    Agent->>Bridge: Invoke sleep() routine
    Bridge->>Dream: start_dreaming()
    activate Dream
    
    Dream->>DB: Fetch recent idle Events
    Dream->>Dream: Compute Cosine Similarity & Cluster
    Dream->>DB: Synthesize & persist new Epiphanies (Knowledge)
    
    %% Fractal Compression Over Time
    Dream->>Fractal: Evaluate legacy memory banks
    activate Fractal
    Fractal->>DB: Retrieve Events > 5 mins old
    Fractal->>Fractal: Apply Temporal Decay (importance reduction)
    Fractal->>DB: Strip redundant details, persist Abstract Rule
    deactivate Fractal
    
    Dream-->>DB: Prune deprecated raw events
    deactivate Dream
    
    %% Wake Up
    Agent->>Bridge: Invoke wake_up() routine
    Bridge->>Dream: stop_dreaming()
    note over Agent, DB: AI awakens with optimized cognitive state! 🚀
```

---

## Architectural Advantages

1. **Zero Bottlenecks (High Throughput):** Leveraging `Crossbeam`, the Python Agent is never blocked by memory I/O operations. Memory spikes are offloaded and persisted asynchronously in a dedicated background thread.
2. **Idle Time Utilization:** The AI conserves computational resources (Tokens/CPU) during active execution. It capitalizes on idle states to run the `Dream Engine` in the background, consolidating its experiences without impacting latency.
3. **No Memory Bloating:** Traditional AI agents suffer from context degradation as memory scales. The `Fractal Compressor` mitigates this by aggressively compressing context into immutable rules, optimizing storage footprint and ensuring retrieval latency remains sub-millisecond.
