"""Provider-neutral streaming speech synthesis contract."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol

from .events import AudioFrame


class SpeechSynthesizer(Protocol):
    def synthesize_stream(self, text: AsyncIterator[str]) -> AsyncIterator[AudioFrame]: ...
    async def stop(self) -> None: ...


class RecordingSpeechSynthesizer:
    """Deterministic architecture probe used by tests and text-output demos."""

    def __init__(self, *, sample_rate: int = 24_000) -> None:
        self.sample_rate = sample_rate
        self.utterances: list[str] = []
        self.stopped = False

    async def synthesize_stream(self, text: AsyncIterator[str]) -> AsyncIterator[AudioFrame]:
        chunks = [chunk async for chunk in text]
        utterance = "".join(chunks).strip()
        if not utterance:
            return
        self.utterances.append(utterance)
        # This adapter proves stream plumbing; it intentionally emits silence rather
        # than masquerading as a standalone audible TTS provider.
        yield AudioFrame(b"\x00\x00" * 240, sample_rate=self.sample_rate)

    async def stop(self) -> None:
        self.stopped = True
