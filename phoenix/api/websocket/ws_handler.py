
"""
WebSocket handler for streaming ChatBot responses.
"""
try:
    from fastapi import WebSocket, WebSocketDisconnect
except ImportError:
    raise ImportError("Install fastapi: pip install fastapi")

_bot_instance = None

def init_ws(bot_instance):
    global _bot_instance
    _bot_instance = bot_instance

async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            session_id = data.get("session_id", "default")
            message = data.get("message", "")
            if not _bot_instance:
                await websocket.send_json({"error": "Bot not initialized"})
                continue
            _bot_instance.set_session(session_id)
            reply = await _bot_instance.chat(text=message)
            await websocket.send_json({"session_id": session_id, "reply": reply})
    except WebSocketDisconnect:
        pass

