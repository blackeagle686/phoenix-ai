# Phoenix Tool Bank Guide

The **Phoenix Tool Bank** (`phoenix.framework.agent.tools.bank`) is a modular, plug-and-play collection of pre-built capabilities that you can easily attach to any Phoenix Agent. Instead of writing custom API wrappers, you can instantly give your agents the ability to browse the web, interact with GitHub, or send emails.

---

## 🛠️ Tool Categories

The Tool Bank is divided into logical domains to keep your agents focused and your dependencies lightweight:

1. **`web`**: Information retrieval tools.
   - `WikipediaSearchTool`: Fetches summaries and articles from Wikipedia.
   - `ArxivSearchTool`: Queries academic papers and research from arXiv.
   - `WebScraperTool`: Extracts text content from standard web pages.

2. **`devops`**: Engineering and infrastructure management.
   - `GitHubTool`: Reads repository files, searches issues, and fetches commit histories without cloning locally.

3. **`productivity`**: Communication and daily operations.
   - `EmailTool`: Sends emails via standard SMTP servers.
   - `SlackTool`: Sends messages and alerts via Slack Webhooks.

---

## 🚀 Scenario 1: The Academic Research Assistant

**Goal:** Create an agent that can autonomously search for recent academic papers and summarize the background concepts using Wikipedia.

```python
import asyncio
from phoenix.main import init_phoenix
from phoenix.framework.agent import Agent
from phoenix.framework.agent.core.profile import AgentProfile, Identity, Role, Personality
from phoenix.framework.agent.tools.registry import ToolRegistry
from phoenix.framework.agent.tools.bank.web import WikipediaSearchTool, ArxivSearchTool

async def run_research_assistant():
    init_phoenix()

    # 1. Define the Agent Profile
    profile = AgentProfile(
        identity=Identity(name="ScholarBot", id="scholar-1"),
        role=Role(title="Research Assistant", mission="Find and summarize academic research."),
        personality=Personality(communication_tone="Academic", response_style="Detailed"),
        rules=["Always cite your sources using the Arxiv ID or Wikipedia URL."],
        tool_access=["wikipedia_search", "arxiv_search"]
    )

    # 2. Register Web Tools
    registry = ToolRegistry()
    registry.register(WikipediaSearchTool())
    registry.register(ArxivSearchTool())

    # 3. Instantiate the Agent
    agent = Agent(profile=profile, tools=registry)
    
    # 4. Execute the Task
    response = await agent.run(
        "Can you find recent papers on Quantum Error Correction and explain the basic concept using Wikipedia?"
    )
    print(response)

if __name__ == "__main__":
    asyncio.run(run_research_assistant())
```

---

## 👨‍💻 Scenario 2: The DevOps Code Reviewer

**Goal:** Create an agent that can monitor a specific GitHub repository, read its code, and analyze recent commits.

**Prerequisites:** Set your `GITHUB_TOKEN` in your `.env` file to access private repositories or increase API limits.

```python
import asyncio
from phoenix.main import init_phoenix
from phoenix.framework.agent import Agent
from phoenix.framework.agent.core.profile import AgentProfile, Identity, Role, Personality
from phoenix.framework.agent.tools.registry import ToolRegistry
from phoenix.framework.agent.tools.bank.devops import GitHubTool

async def run_devops_monitor():
    init_phoenix()

    profile = AgentProfile(
        identity=Identity(name="DevOpsMonitor", id="dev-1"),
        role=Role(title="Repository Manager", mission="Analyze code changes and repository health."),
        personality=Personality(communication_tone="Professional", response_style="Concise"),
        rules=["Never expose the GitHub token in your responses."],
        tool_access=["github"]
    )

    registry = ToolRegistry()
    registry.register(GitHubTool())

    agent = Agent(profile=profile, tools=registry)
    
    response = await agent.run(
        "Please fetch the last 3 commits from 'blackeagle686/phoenix-ai' and summarize what changed."
    )
    print(response)

if __name__ == "__main__":
    asyncio.run(run_devops_monitor())
```

---

## 📬 Scenario 3: The Productivity & Alert Bot

**Goal:** Create an agent that performs an action and notifies the user via Email or Slack upon completion.

**Prerequisites:** 
- Email: Set `SMTP_EMAIL` and `SMTP_PASSWORD` in your `.env`.
- Slack: Set `SLACK_WEBHOOK_URL` in your `.env`.

```python
import asyncio
from phoenix.main import init_phoenix
from phoenix.framework.agent import Agent
from phoenix.framework.agent.core.profile import AgentProfile, Identity, Role, Personality
from phoenix.framework.agent.tools.registry import ToolRegistry
from phoenix.framework.agent.tools.bank.productivity import EmailTool, SlackTool

async def run_productivity_bot():
    init_phoenix()

    profile = AgentProfile(
        identity=Identity(name="AlertBot", id="alert-1"),
        role=Role(title="Notification Assistant", mission="Dispatch alerts to the team."),
        personality=Personality(communication_tone="Urgent", response_style="Direct"),
        rules=["Ensure all alerts include a timestamp."],
        tool_access=["email", "slack"]
    )

    registry = ToolRegistry()
    registry.register(EmailTool())
    registry.register(SlackTool())

    agent = Agent(profile=profile, tools=registry)
    
    response = await agent.run(
        "Send an email to engineering@mycompany.com and a Slack message saying: "
        "'The database migration has completed successfully.'"
    )
    print(response)

if __name__ == "__main__":
    asyncio.run(run_productivity_bot())
```

---

## Best Practices
1. **Combine Registries:** You can mix and match tools from different bank categories (e.g., an agent with `ArxivSearchTool` and `SlackTool` to automatically Slack you new papers).
2. **Environment Variables:** Most third-party tools require authentication. Phoenix automatically loads your `.env` file upon `init_phoenix()`, allowing the tools to securely pick up credentials without hardcoding them in your logic.
3. **Graceful Failures:** If an API key is missing, the Tool Bank tools are designed to catch the error and politely inform the LLM, preventing the entire agent process from crashing.

