# Building a Fully Customized Agent with the Phoenix Framework

The Phoenix Agent Framework is built for extreme extensibility. In this scenario, we will build a **"Stock Market Analyst Agent"**. 

Instead of using the default file I/O tools and default reasoning, we will provide the agent with:
1. A **Custom Tool** to fetch live stock prices.
2. A **Custom Thinker** that enforces financial compliance rules during its reasoning phase.
3. A **Custom Reflector** that evaluates whether the agent properly calculated the risk-to-reward ratio.
4. A **Custom Loop** to perform specialized background logging during execution.

---

## 1. Create a Custom Tool
Tools are the hands of the agent. By subclassing `BaseTool`, you can give the agent any capability, such as querying databases, hitting APIs, or controlling physical devices.

```python
from phoenix.framework.agent.tools.base import BaseTool, ToolResult
import httpx

class StockPriceTool(BaseTool):
    name = "fetch_stock_price"
    description = (
        "Fetches the real-time stock price for a given ticker symbol. "
        "Input: 'ticker' (str, e.g., 'AAPL')."
    )

    async def execute(self, ticker: str, **kwargs) -> ToolResult:
        try:
            # Simulate an API call to a financial service
            prices = {"AAPL": 150.25, "TSLA": 200.50, "MSFT": 310.10}
            price = prices.get(ticker.upper())
            
            if price:
                return ToolResult(success=True, output=f"The current price of {ticker.upper()} is ${price}")
            else:
                return ToolResult(success=False, error=f"Ticker {ticker} not found.")
        except Exception as e:
            return ToolResult(success=False, error=str(e))
```

## 2. Create a Custom Thinker
The `Thinker` is the brain responsible for analyzing tasks and deciding which tools to call. We can subclass the default `Thinker` (or `BaseThinker`) to override its behavior—in this case, injecting a custom system prompt that enforces strict financial compliance.

```python
from phoenix.framework.agent.cognition.thinker.thinker import Thinker
from phoenix.framework.agent.cognition.schema import TaskExecutionSchema
from typing import Any

class FinancialComplianceThinker(Thinker):
    
    async def solve_task(self, task: Any, context: str = "") -> TaskExecutionSchema:
        """Override the default reasoning step to inject financial rules."""
        
        tools_list = self._get_tools_list()
        task_desc = getattr(task, "description", str(task))
        
        system_prompt = (
            "You are a licensed Financial Analyst Agent. "
            "You must analyze the following task and determine the best approach. "
            "COMPLIANCE RULE: You must always explicitly state the 'Risk Warning' in your thought process.\n\n"
            f"Available Tools:\n{tools_list}"
        )
        
        full_prompt = f"{system_prompt}\n\nTask: {task_desc}\nContext: {context}"
        
        # Use the LLM to generate the strict TaskExecutionSchema
        return await self.llm.generate_structured(full_prompt, TaskExecutionSchema)
```

## 3. Create a Custom Reflector
The `Reflector` judges whether the Actor successfully completed the task. We can create a custom reflector that refuses to mark a task as "done" unless the agent explicitly calculated the risk ratio.

```python
from phoenix.framework.agent.cognition.reflector.base import BaseReflector
from phoenix.framework.agent.cognition.schema import ReflectorInputSchema, ReflectionSchema

class RiskAwareReflector(BaseReflector):
    
    async def reflect(self, input_data: ReflectorInputSchema) -> ReflectionSchema:
        # Check the actor's output to see if "risk" was mentioned
        target_content = str(input_data.target_content).lower()
        
        if "risk" not in target_content:
            return ReflectionSchema(
                status="failed",
                feedback="You failed to calculate or mention the risk ratio. Please retry and include a risk analysis.",
                rating=2,
                is_task_complete=False
            )
            
        # If risk is present, we pass it to the default LLM reflector logic, or just approve it
        return ReflectionSchema(
            status="success",
            feedback="Risk ratio calculated. The financial analysis is sound.",
            rating=9,
            is_task_complete=True
        )
```

## 4. Assemble the Custom Agent
Now we bring all of our custom components together using the `Agent` class. The framework acts as a dependency-injected orchestrator, seamlessly routing data between your custom classes.

```python
import asyncio
from phoenix.framework.agent.core.agent import Agent
from phoenix.services.llm.openai import OpenAILLM

async def run_custom_agent():
    # 1. Initialize the LLM
    llm = OpenAILLM(api_key="your-api-key", model="LongCat-2.0-Preview")
    await llm.init()

    # 2. Instantiate our custom tools and cognitive components
    custom_tools = [StockPriceTool()]
    custom_thinker = FinancialComplianceThinker(llm=llm)
    custom_reflector = RiskAwareReflector(thinker=custom_thinker)

    # 3. Build the Agent, injecting our custom parts
    # The Agent will automatically build the missing components (like the Planner and Actor)
    # and wire them up with our custom Thinker and Reflector.
    agent = Agent(
        llm=llm,
        tools=custom_tools,
        thinker=custom_thinker,
        reflector=custom_reflector
    )

    # 4. Run the Agent
    prompt = "I have $10,000 to invest. Find the price of AAPL and advise me if it's a good buy right now."
    print("Starting Customized Agent...")
    
    result = await agent.run(prompt)
    print("\n[FINAL RESULT]\n", result)

if __name__ == "__main__":
    asyncio.run(run_custom_agent())
```

### Advanced: Swapping Components at Runtime
The Phoenix framework even allows you to change the agent's behavior dynamically mid-execution. For example, if you wanted the agent to become more cautious after losing money, you could hot-swap the Thinker:

```python
# Swap the thinker to a more conservative one at runtime
agent.set_component("thinker", UltraConservativeThinker(llm=llm), rebuild_loop=True)
```

## 5. Custom Memory & Context Window Management
Memory limits are a huge bottleneck in long-running agent tasks. Phoenix handles memory via dependency injection in the `Agent` class. The agent expects a memory object that implements two core methods: `add_interaction` and `get_full_context`.

If you need a **Custom Context Window** that automatically summarizes old logs, prunes irrelevant data, or uses a Vector Database to inject semantic memories, you can easily build your own Memory Manager!

```python
class SemanticContextMemory:
    """A custom memory manager with a sliding context window and semantic retrieval."""
    
    def __init__(self, max_context_tokens: int = 4000):
        self.history = []
        self.max_tokens = max_context_tokens

    async def add_interaction(self, session_id: str, role: str, content: str):
        """Called by the Loop to record every LLM interaction, tool result, or system log."""
        self.history.append({"role": role, "content": content})
        
        # Implement a custom pruning strategy to prevent context window overflow
        if len(self.history) > 50:
            # Drop older interactions to preserve the context window
            self.history = self.history[-50:]

    async def get_full_context(self, session_id: str, query: str = "") -> str:
        """Called by the Thinker and Agent to build the prompt context window."""
        
        # 1. Base sliding window
        recent_history = "\\n".join([f"{msg['role']}: {msg['content']}" for msg in self.history[-10:]])
        
        # 2. Inject custom business logic or vector search
        relevant_docs = await my_vector_db.search(query)
        
        # 3. Build the highly customized Context Window string
        context_window = f"=== RELEVANT DOCS ===\\n{relevant_docs}\\n\\n=== RECENT HISTORY ===\\n{recent_history}"
        return context_window
```

To use it, just pass it to the `Agent` when initializing:

```python
custom_memory = SemanticContextMemory(max_context_tokens=8000)

agent = Agent(
    memory=custom_memory,
    tools=custom_tools,
    thinker=custom_thinker
)
```

Because of the framework's Duck Typing (`InteractiveMemoryAdapter`), the Agent will natively detect `add_interaction` and `get_full_context` and use your custom memory engine for the entire session lifecycle.

### Summary
By heavily relying on `Base*` abstractions (like `BaseTool`, `BaseThinker`), the Phoenix Framework allows you to:
- Inject entirely proprietary APIs and tools.
- Force the LLM to follow highly specific, domain-aware reasoning logic.
- Override the agent loop to implement custom logging, telemetry, or safety guardrails.
- Seamlessly connect custom Vector Databases or sliding-window logic into the central memory graph.
