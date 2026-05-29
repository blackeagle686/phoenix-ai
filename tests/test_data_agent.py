import sqlite3
import asyncio
import os
from phoenix.main import init_phoenix
from phoenix.framework.agent import Agent
from phoenix.framework.agent.core.profile import AgentProfile, Identity, Role, Personality
from phoenix.framework.agent.tools.registry import ToolRegistry
from phoenix.framework.agent.tools.bank import SQLDatabaseTool
from phoenix.services.llm.openai import OpenAILLM

def setup_mock_db():
    if os.path.exists("test_company.db"):
        os.remove("test_company.db")
    
    conn = sqlite3.connect("test_company.db")
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE employees (id INTEGER PRIMARY KEY, name TEXT, department TEXT)")
    cursor.execute("CREATE TABLE sales (id INTEGER PRIMARY KEY, employee_id INTEGER, amount REAL)")
    
    cursor.execute("INSERT INTO employees (name, department) VALUES ('Alice', 'Sales'), ('Bob', 'Engineering'), ('Charlie', 'Sales')")
    cursor.execute("INSERT INTO sales (employee_id, amount) VALUES (1, 5000), (3, 7500), (1, 1200)")
    
    conn.commit()
    conn.close()

async def main():
    setup_mock_db()
    init_phoenix()
    
    profile = AgentProfile(
        identity=Identity(name="DataBot", id="db-1"),
        role=Role(title="Data Analyst", mission="Extract insights from the company database."),
        personality=Personality(communication_tone="Professional", response_style="Concise"),
        rules=["Always start by checking the database schema."],
        tool_access=["sql_database"]
    )
    
    registry = ToolRegistry()
    registry.register(SQLDatabaseTool(connection_uri="sqlite:///test_company.db", read_only=True))
    
    # Initialize the real LLM with Gemini
    llm = OpenAILLM(
        api_key=os.environ.get("GEMINI_API_KEY", ""),
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        model="gemini-2.5-flash"
    )
    await llm.init()
    
    # Basic Memory to bypass vector DB dependencies
    class BasicMemory:
        async def add_interaction(self, session_id, role, content, metadata=None): pass
        async def get_full_context(self, session_id, query=None): return ""
        
    agent = Agent(llm=llm, profile=profile, tools=registry, memory=BasicMemory())
    
    prompt = "Look into the database and tell me the name of the employee who made the highest single sale, and how much it was."
    print("[*] Agent is starting...")
    print(f"[*] Request: {prompt}\n")
    
    response = await agent.run(prompt, mode="plan")
    
    print("\nFinal Agent Response:")
    print("=====================")
    print(response)

if __name__ == "__main__":
    asyncio.run(main())
