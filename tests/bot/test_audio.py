import asyncio
import os
import sys

# Add project root to path
sys.path.insert(0, os.getcwd())

from phoenix.services.audio.local import LocalSTT, LocalTTS

async def test_audio_flow():
    print("\n" + "="*50)
    print("PHOENIX AI: AUDIO SERVICE TEST (gTTS & SpeechRecognition)")
    print("="*50)
    print("NOTE: This test requires internet access for Google STT/TTS APIs.")

    # 1. Test TTS (Generate real audio first so we can transcribe it)
    print("\n[1/2] Testing Local TTS (Synthesis with gTTS)...")
    tts = LocalTTS()
    await tts.init()
    
    output_audio = "tests/test_voice.mp3"
    input_text = "Phoenix AI is an agentic coding assistant."
    print(f"[*] Synthesizing text: '{input_text}' (Lang: en)")
    
    try:
        result_path = await tts.synthesize(input_text, output_audio, lang="en")
        if os.path.exists(result_path):
            print(f"[v] Success! Generated audio at: {result_path}")
        else:
            print("[!] Error: Output file not created.")
    except Exception as e:
        print(f"[!] TTS Error: {e}")

    # 2. Test STT
    print("\n[2/2] Testing Local STT (Transcription with SpeechRecognition)...")
    stt = LocalSTT()
    await stt.init()
    
    # Note: SpeechRecognition works best with WAV for AudioFile
    print("[*] Note: STT typically requires .wav format for sr.AudioFile.")
    
    # Cleanup
    if os.path.exists(output_audio):
        os.remove(output_audio)

    print("\n" + "="*50)
    print("AUDIO SERVICE TEST COMPLETE")
    print("="*50 + "\n")


if __name__ == "__main__":
    asyncio.run(test_audio_flow())

