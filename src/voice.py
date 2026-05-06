"""
AutoDrive Voice — Groq Whisper speech-to-text.
Accepts raw audio bytes, returns transcript string.
"""

from __future__ import annotations
import io
from .config import settings


def transcribe_audio(audio_bytes: bytes, filename: str = "audio.webm") -> str:
    """Send audio to Groq Whisper and return transcript."""
    if not settings.GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set — cannot transcribe audio.")

    from groq import Groq
    client = Groq(api_key=settings.GROQ_API_KEY)

    audio_file = (filename, io.BytesIO(audio_bytes), "audio/webm")
    result = client.audio.transcriptions.create(
        file=audio_file,
        model="whisper-large-v3",
        language="en",
        response_format="text",
    )
    return result.strip() if isinstance(result, str) else result.text.strip()
