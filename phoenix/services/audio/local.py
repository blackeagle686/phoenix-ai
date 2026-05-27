from phoenix.services.audio.base import BaseSTT, BaseTTS
import asyncio
import os

class LocalSTT(BaseSTT):
    def __init__(self):
        self.recognizer = None

    async def init(self):
        try:
            import speech_recognition as sr
        except ImportError:
            raise ImportError(
                "speech_recognition is not installed. "
                "Please install it using: pip install phx-ashborn[audio] or pip install SpeechRecognition"
            )
        self.recognizer = sr.Recognizer()

    async def transcribe(self, audio_path: str, lang: str = "en-US") -> str:
        if not self.recognizer:
            await self.init()
        
        try:
            import speech_recognition as sr
            with sr.AudioFile(audio_path) as source:
                audio_data = self.recognizer.record(source)
                # Using Google's free web search API for transcription
                text = self.recognizer.recognize_google(audio_data, language=lang)
                return text
        except Exception as e:
            return f"Error transcribing audio: {str(e)}"

class LocalTTS(BaseTTS):
    def __init__(self):
        self.initialized = False

    async def init(self):
        self.initialized = True

    async def synthesize(self, text: str, output_path: str, lang: str = "en") -> str:
        try:
            from gtts import gTTS
        except ImportError:
            raise ImportError(
                "gtts is not installed. "
                "Please install it using: pip install phx-ashborn[audio] or pip install gTTS"
            )
        try:
            # Generate speech using gTTS (Google Text-to-Speech)
            tts = gTTS(text=text, lang=lang)
            # gTTS save is blocking, so run in executor if needed, 
            # but for simplicity in SDK we'll just call it.
            tts.save(output_path)
            return output_path

        except Exception as e:
            raise RuntimeError(f"TTS Synthesis failed: {e}")
