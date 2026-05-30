#  Phoenix AI: Django Integration Guide

This guide provides a deep dive into integrating the Phoenix Autonomous Agent into a Django application. We will focus on using the **Singleton Pattern** to manage the agent instance efficiently, preventing redundant memory usage and ensuring consistent service availability across your application.

---

##  The Singleton Pattern for AI Agents

AI Agents are "heavy" objects. They initialize LLMs, load tool registries, and connect to memory systems (Redis/Vector DB). Re-initializing an agent on every HTTP request is slow and resource-intensive.

The **Singleton Pattern** ensures that only one instance of the Agent exists in your Django process.

### 1. Creating the Agent Service

Create a file named `services.py` in one of your Django apps (e.g., `ai_core/services.py`).

```python
import asyncio
import threading
from phoenix import init_phoenix, startup_phoenix
from phoenix import Agent

class PhoenixAgentService:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        """Thread-safe Singleton implementation."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(PhoenixAgentService, cls).__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
            
        self.agent = None
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self._thread.start()
        
        # Start initialization
        self.initialize()
        self._initialized = True

    def _run_event_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def initialize(self):
        """Asynchronous initialization ran in the dedicated thread."""
        future = asyncio.run_coroutine_threadsafe(self._async_init(), self._loop)
        return future.result() # Wait for completion

    async def _async_init(self):
        init_phoenix()
        await startup_phoenix()
        self.agent = Agent()

    def run_agent(self, prompt, session_id=None, mode="auto"):
        """Synchronous wrapper for views."""
        if not self.agent:
            raise RuntimeError("Agent not initialized")
            
        future = asyncio.run_coroutine_threadsafe(
            self.agent.run(prompt, session_id=session_id, mode=mode),
            self._loop
        )
        return future.result()

# Global access point
phoenix_service = PhoenixAgentService()
```

---

##  Usage in Django Views

### 1. Simple Function-Based View (FBV)

This is the easiest way to expose your agent via a REST API.

```python
# views.py
from django.http import JsonResponse
from .services import phoenix_service

def chat_view(request):
    user_prompt = request.GET.get('prompt')
    session_id = request.GET.get('session_id', 'default_user')
    
    if not user_prompt:
        return JsonResponse({"error": "No prompt provided"}, status=400)
    
    try:
        # Using our singleton service
        response = phoenix_service.run_agent(user_prompt, session_id=session_id)
        return JsonResponse({
            "status": "success",
            "reply": response,
            "session_id": session_id
        })
    except Exception as e:
        return JsonResponse({"status": "error", "message": str(e)}, status=500)
```

### 2. Async View (Native Django 3.1+)

If you are using an ASGI server (like Uvicorn or Daphne), you can use Django's native async views for better performance.

```python
# views.py
import json
from django.http import JsonResponse
from .services import phoenix_service

async def async_chat_view(request):
    if request.method == "POST":
        data = json.loads(request.body)
        prompt = data.get("prompt")
        session_id = data.get("session_id")
        
        # We can call the agent's run method directly if we have access to the loop
        # Or use our service's run_agent which handles the threading
        response = phoenix_service.run_agent(prompt, session_id=session_id)
        
        return JsonResponse({"reply": response})
```

---

## ⚡ Real-time Interaction with Django Channels

For a truly "alive" agent, you should use WebSockets. This allows the agent to send progress updates or partial thoughts as it works.

```python
# consumers.py
import json
from channels.generic.websocket import AsyncWebsocketConsumer
from .services import phoenix_service

class AgentConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.accept()
        await self.send(text_data=json.dumps({
            "status": "connected",
            "message": "Phoenix Agent is listening..."
        }))

    async def receive(self, text_data):
        data = json.loads(text_data)
        prompt = data.get("prompt")
        
        # Note: In a production scenario, you'd want to use 
        # database_sync_to_async or similar for non-async parts
        # but since our service handles its own loop, we can just call it
        response = phoenix_service.run_agent(prompt)
        
        await self.send(text_data=json.dumps({
            "type": "agent_response",
            "reply": response
        }))
```

---

## 🛠️ Configuration & Best Practices

### 1. Environment Variables
Don't hardcode keys. Use `python-dotenv` or Django settings.

```python
# settings.py
import os
from dotenv import load_dotenv

load_dotenv()

PHOENIX_API_KEY = os.getenv("OPENAI_API_KEY")
PHOENIX_VECTOR_DB = os.getenv("VECTOR_DB_TYPE", "chroma")
```

### 2. Pre-loading the Agent
To avoid the first-request delay, you can trigger the singleton in your `apps.py`.

```python
# apps.py
from django.apps import AppConfig

class AiCoreConfig(AppConfig):
    name = 'ai_core'

    def ready(self):
        # Importing here triggers the singleton initialization
        from .services import phoenix_service
        print("🐦🔥 Phoenix Agent Service Pre-loaded.")
```

---

##  Frequently Asked Questions

**Q: Can I have multiple agents?**
A: Yes. You can modify the Singleton to be a "Multiton" (Registry pattern) if you need different agents for different tasks (e.g., a "CoderAgent" and a "SupportAgent").

**Q: How do I handle file uploads to the agent?**
A: Use Django's `FileSystemStorage` to save the file, then pass the absolute path to the agent:
```python
path = fs.save(file.name, file)
full_path = fs.path(path)
response = phoenix_service.run_agent(f"Analyze this file: {full_path}")
```

**Q: Is the Singleton thread-safe?**
A: The implementation provided above uses a `threading.Lock` to ensure that only one instance is created even if multiple threads try to access it simultaneously during startup.

