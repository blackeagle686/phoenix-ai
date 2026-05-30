import os
from phoenix.services.audio.base import BaseSTT, BaseTTS
from phoenix.core.config import config
from openai import OpenAI

class OpenAISTT(BaseSTT):
    def __init__(self):
        self.api_key = config.OPENAI_API_KEY
        self.base_url = config.OPENAI_BASE_URL
        self.client = None

    async def init(self):
        if not self.api_key:
            print("Warning: OPENAI_API_KEY is missing. Operating in mock mode.")
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

    async def transcribe(self, audio_path: str, model: str = "whisper-1") -> str:
        if not self.client:
            raise RuntimeError("OpenAISTT is not initialized.")
        if not self.api_key:
            return f"[Mock OpenAISTT Transcription (No API Key) for: {audio_path}]"

        with open(audio_path, "rb") as audio_file:
            transcript = self.client.audio.transcriptions.create(
                model=model,
                file=audio_file
            )
        return transcript.text

class OpenAITTS(BaseTTS):
    def __init__(self):
        self.api_key = config.OPENAI_API_KEY
        self.base_url = config.OPENAI_BASE_URL
        self.client = None

    async def init(self):
        if not self.api_key:
            print("Warning: OPENAI_API_KEY is missing. Operating in mock mode.")
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

    async def synthesize(self, text: str, output_path: str, model: str = "tts-1", voice: str = "alloy") -> str:
        if not self.client:
            raise RuntimeError("OpenAITTS is not initialized.")
        if not self.api_key:
            with open(output_path, "wb") as f:
                f.write(b"Mock OpenAI audio data")
            return output_path

        response = self.client.audio.speech.create(
            model=model,
            voice=voice,
            input=text
        )
        with open(output_path, "wb") as f:
            for chunk in response.iter_bytes():
                f.write(chunk)
        return output_path

