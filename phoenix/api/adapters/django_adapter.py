
"""
Django adapter: integrates Phoenix with Django views.
"""
import asyncio

_bot_instance = None

def init_django(bot_instance):
    global _bot_instance
    _bot_instance = bot_instance

def phoenix_view(request):
    """Django view that proxies requests to the Phoenix ChatBot."""
    try:
        from django.http import JsonResponse
        from django.views.decorators.csrf import csrf_exempt
        import json
    except ImportError:
        raise ImportError("Django is not installed.")

    if request.method == "POST":
        body = json.loads(request.body)
        message = body.get("message", "")
        session_id = body.get("session_id", "default")
        _bot_instance.set_session(session_id)
        loop = asyncio.new_event_loop()
        reply = loop.run_until_complete(_bot_instance.chat(text=message))
        loop.close()
        return JsonResponse({"reply": reply})
    return JsonResponse({"error": "Only POST allowed"}, status=405)

