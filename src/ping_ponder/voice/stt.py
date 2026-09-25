"""Speech recognizer contract used by the voice runtime."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Protocol

from .events import AudioFrame, SpeechEvent


class SpeechRecognizer(Protocol):
    async def run(
        self,
        audio: AsyncIterator[AudioFrame],
        events: asyncio.Queue[SpeechEvent],
    ) -> None: ...

    async def stop(self) -> None: ...
