
"""
Ready-to-use FastAPI router for Phoenix Agent.
"""
try:
    from fastapi import APIRouter, HTTPException
    from pydantic import BaseModel
except ImportError:
    raise ImportError("Install fastapi: pip install fastapi")

router = APIRouter(prefix="/agent", tags=["Agent"])

class AgentRequest(BaseModel):
    session_id: str = "default"
    task: str
    mode: str = "auto" # auto, plan, fast_ans

class AgentResponse(BaseModel):
    session_id: str
    result: str
    mode: str

_agent_instance = None

def init_router(agent_instance):
    global _agent_instance
    _agent_instance = agent_instance

@router.post("/run", response_model=AgentResponse)
async def run_agent(req: AgentRequest):
    if not _agent_instance:
        raise HTTPException(status_code=503, detail="Agent not initialized.")
    result = await _agent_instance.run(req.task, session_id=req.session_id, mode=req.mode)
    return AgentResponse(session_id=req.session_id, result=str(result), mode=req.mode)

