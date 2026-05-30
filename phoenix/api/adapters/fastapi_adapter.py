
"""
FastAPI adapter: quickly mount Phoenix ChatBot and Agent into any FastAPI app.
"""
from phoenix.api.rest.chatbot_router import router as chatbot_router, init_router as init_chatbot
from phoenix.api.rest.agent_router import router as agent_router, init_router as init_agent

def mount_chatbot(app, bot_instance, prefix=""):
    init_chatbot(bot_instance)
    app.include_router(chatbot_router, prefix=prefix)

def mount_agent(app, agent_instance, prefix=""):
    init_agent(agent_instance)
    app.include_router(agent_router, prefix=prefix)

