from phoenix.core.base import BaseService

class BaseSTT(BaseService):
    async def transcribe(self, audio_path: str, lang: str = "en-US") -> str:
        """Transcribe speech to text from an audio file."""
        raise NotImplementedError

class BaseTTS(BaseService):
    async def synthesize(self, text: str, output_path: str, lang: str = "en") -> str:
        """Synthesize text to speech and save to output_path."""
        raise NotImplementedError


