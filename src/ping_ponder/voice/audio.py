"""Replaceable local audio input/output boundaries."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import Protocol

from ping_ponder.queueing import DropOldestQueue

from .events import AudioFrame

logger = logging.getLogger(__name__)


class AudioInput(Protocol):
    async def start(self) -> None: ...
    def frames(self) -> AsyncIterator[AudioFrame]: ...
    async def stop(self) -> None: ...


class AudioOutput(Protocol):
    async def start(self) -> None: ...
    async def play(self, frames: AsyncIterator[AudioFrame]) -> None: ...
    async def stop_playback(self) -> None: ...
    async def stop(self) -> None: ...


class SoundDeviceAudioInput:
    """PCM16 microphone capture with bounded, drop-oldest buffering."""

    def __init__(self, *, sample_rate: int = 16_000, channels: int = 1,
                 block_samples: int = 800, device=None, queue_size: int = 8,
                 sounddevice_module=None) -> None:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if channels <= 0 or block_samples <= 0:
            raise ValueError("channels and block_samples must be positive")
        self.sample_rate = sample_rate
        self.channels = channels
        self.block_samples = block_samples
        self.device = device
        self._sounddevice = sounddevice_module
        self._queue: DropOldestQueue[AudioFrame | None] = DropOldestQueue(queue_size)
        self._stream = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._running = False

    async def start(self) -> None:
        if self._running:
            return
        sounddevice = self._sounddevice
        if sounddevice is None:
            try:
                import sounddevice
            except (ImportError, OSError) as error:
                raise RuntimeError(
                    "Local microphone input requires the 'voice' optional dependencies and PortAudio"
                ) from error
        self._loop = asyncio.get_running_loop()

        def callback(indata, frames, time_info, status):
            if status:
                logger.warning("voice_input_status %s", status)
            frame = AudioFrame(bytes(indata), self.sample_rate, self.channels, 2)
            loop = self._loop
            if loop is not None and not loop.is_closed():
                loop.call_soon_threadsafe(self._queue.put_nowait, frame)

        self._stream = sounddevice.RawInputStream(
            samplerate=self.sample_rate, channels=self.channels, dtype="int16",
            blocksize=self.block_samples, device=self.device, callback=callback,
        )
        self._stream.start()
        self._running = True

    async def frames(self) -> AsyncIterator[AudioFrame]:
        while self._running:
            frame = await self._queue.get()
            if frame is None:
                return
            yield frame

    async def stop(self) -> None:
        self._running = False
        self._queue.put_nowait(None)
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
        self._stream = None
        self._loop = None


class SoundDeviceAudioOutput:
    """PCM16 speaker playback whose current stream can be interrupted."""

    def __init__(self, *, sample_rate: int | None = 24_000, channels: int = 1,
                 device=None, sounddevice_module=None) -> None:
        if sample_rate is not None and sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if channels <= 0:
            raise ValueError("channels must be positive")
        self.sample_rate = sample_rate
        self.channels = channels
        self.device = device
        self._sounddevice = sounddevice_module
        self._sounddevice_module = None
        self._stream = None
        self._stream_rate: int | None = None
        self._epoch = 0

    async def start(self) -> None:
        if self._stream is not None:
            return
        sounddevice = self._sounddevice
        if sounddevice is None:
            try:
                import sounddevice
            except (ImportError, OSError) as error:
                raise RuntimeError(
                    "Local speaker output requires the 'voice' optional dependencies and PortAudio"
                ) from error
        self._sounddevice_module = sounddevice
        if self.sample_rate is not None:
            self._open_stream(self.sample_rate)

    def _open_stream(self, sample_rate: int) -> None:
        self._stream = self._sounddevice_module.RawOutputStream(
            samplerate=sample_rate, channels=self.channels, dtype="int16",
            device=self.device)
        self._stream.start()
        self._stream_rate = sample_rate

    def _ensure_stream(self, sample_rate: int) -> None:
        if self.sample_rate is not None and sample_rate != self.sample_rate:
            raise ValueError("synthesizer sample rate does not match output")
        if self._stream is not None and self._stream_rate == sample_rate:
            return
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self._open_stream(sample_rate)

    async def play(self, frames: AsyncIterator[AudioFrame]) -> None:
        epoch = self._epoch
        async for frame in frames:
            if epoch != self._epoch:
                return
            if (frame.channels, frame.sample_width) != (self.channels, 2):
                raise ValueError("synthesizer audio format does not match output")
            if self._sounddevice_module is None:
                raise RuntimeError("audio output is not started")
            self._ensure_stream(frame.sample_rate)
            await asyncio.to_thread(self._stream.write, frame.pcm)

    async def stop_playback(self) -> None:
        self._epoch += 1
        if self._stream is not None:
            self._stream.abort()
            self._stream.start()

    async def stop(self) -> None:
        self._epoch += 1
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
        self._stream = None
        self._stream_rate = None
        self._sounddevice_module = None
