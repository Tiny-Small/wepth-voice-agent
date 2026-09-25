"""OpenRouter adapter for Fish Audio's PCM text-to-speech model."""

from __future__ import annotations

import re
from collections.abc import AsyncIterator

import httpx

from .events import AudioFrame


FISH_AUDIO_MODEL = "fish-audio/s2.1-pro-free:free"
FISH_AUDIO_ENDPOINT = "https://openrouter.ai/api/v1/audio/speech"


class FishAudioSynthesizer:
    """Turn each committed phrase into streamed PCM frames for the audio output."""

    def __init__(self, *, client: httpx.AsyncClient, api_key: str,
                 endpoint: str = FISH_AUDIO_ENDPOINT,
                 model: str = FISH_AUDIO_MODEL, voice: str | None = None) -> None:
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is required for Fish Audio TTS")
        self.client, self.api_key = client, api_key
        self.endpoint, self.model, self.voice = endpoint, model, voice

    async def synthesize_stream(self, text: AsyncIterator[str]) -> AsyncIterator[AudioFrame]:
        phrase = "".join([part async for part in text]).strip()
        if not phrase:
            return
        payload = {"model": self.model, "input": phrase, "response_format": "pcm"}
        if self.voice:
            payload["voice"] = self.voice
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        async with self.client.stream(
                "POST", self.endpoint, headers=headers, json=payload,
                timeout=httpx.Timeout(connect=10, read=45, write=10, pool=10)) as response:
            if response.is_error:
                body = (await response.aread()).decode("utf-8", errors="replace").strip()
                detail = " ".join(body.split())[:1000]
                message = f"OpenRouter TTS returned HTTP {response.status_code}"
                if detail:
                    message = f"{message}: {detail}"
                raise httpx.HTTPStatusError(
                    message, request=response.request, response=response,
                )
            content_type = response.headers.get("content-type", "").lower()
            if not content_type.startswith("audio/pcm"):
                raise ValueError(f"Fish Audio returned {content_type or 'no content type'}; expected audio/pcm")
            sample_rate = _content_parameter(content_type, "rate", 44_100)
            channels = _content_parameter(content_type, "channels", 1)
            remainder = b""
            async for body in response.aiter_bytes(4096):
                pcm = remainder + body
                aligned_length = len(pcm) - (len(pcm) % (2 * channels))
                if aligned_length:
                    yield AudioFrame(pcm[:aligned_length], sample_rate, channels, 2)
                remainder = pcm[aligned_length:]
            if remainder:
                raise ValueError("Fish Audio returned an incomplete PCM sample")


def _content_parameter(content_type: str, name: str, default: int) -> int:
    match = re.search(rf"(?:^|;)\s*{name}\s*=\s*(\d+)", content_type)
    value = int(match.group(1)) if match else default
    if value <= 0:
        raise ValueError(f"Fish Audio returned an invalid PCM {name}")
    return value
