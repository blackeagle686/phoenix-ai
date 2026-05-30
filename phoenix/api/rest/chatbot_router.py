
"""
Ready-to-use FastAPI router for Phoenix ChatBot.
Mount this in any FastAPI app with: app.include_router(chatbot_router)
"""
from typing import Optional
try:
    from fastapi import APIRouter, HTTPException, File, UploadFile, Form
    from pydantic import BaseModel
except ImportError:
    raise ImportError("Install fastapi and pydantic: pip install fastapi pydantic")

router = APIRouter(prefix="/chat", tags=["ChatBot"])

class ChatResponse(BaseModel):
    session_id: str
    reply: str

_bot_instance = None

def init_router(bot_instance):
    """Call this with your ChatBotInstance before mounting."""
    global _bot_instance
    _bot_instance = bot_instance

@router.post("/", response_model=ChatResponse)
async def chat(
    message: str = Form(...),
    session_id: str = Form("default"),
    image: Optional[UploadFile] = File(None)
):
    if not _bot_instance:
        raise HTTPException(status_code=503, detail="ChatBot not initialized. Call init_router() first.")
    
    import tempfile
    import os
    import shutil
    
    temp_path = None
    if image:
        suffix = os.path.splitext(image.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(image.file, tmp)
            temp_path = tmp.name

    try:
        _bot_instance.set_session(session_id)
        reply = await _bot_instance.chat(text=message, image_path=temp_path)
        
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
            
        return ChatResponse(session_id=session_id, reply=reply)
    except Exception as e:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=str(e))

