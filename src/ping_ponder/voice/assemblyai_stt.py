"""AssemblyAI U3.5 Pro Realtime adapter."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import dataclass
from typing import Any

from .events import (
    AudioFrame,
    FinalTranscript,
    PartialTranscript,
    SpeechEvent,
    SpeechRecognitionError,
    SpeechStarted,
    SpeechStopped,
    TranscriptState,
)


@dataclass(frozen=True)
class AssemblyAIRecognizerSettings:
    sample_rate: int = 16_000
    speech_model: str = "universal-3-5-pro"
    format_turns: bool = True
    min_turn_silence: int | None = None
    max_turn_silence: int | None = None


ClientFactory = Callable[[AssemblyAIRecognizerSettings, Mapping[str, Callable]], Any]


class SpeechRecognitionRuntimeError(RuntimeError):
    """AssemblyAI terminated a live recognition session with an error."""


class AssemblyAIStreamingRecognizer:
    """Maps SDK callbacks to a queue without running downstream work inline."""

    def __init__(self, *, api_key: str,
                 settings: AssemblyAIRecognizerSettings | None = None,
                 event_queue: asyncio.Queue[SpeechEvent] | None = None,
                 client_factory: ClientFactory | None = None,
                 on_ready: Callable[[], None] | None = None) -> None:
        if not api_key:
            raise ValueError("AssemblyAI API key is required")
        self.api_key = api_key
        self.settings = settings or AssemblyAIRecognizerSettings()
        self._events = event_queue
        self._client_factory = client_factory
        self._on_ready = on_ready
        self._client = None
        self._parameters_factory: Callable[..., Any] | None = None
        self._revisions: dict[str, int] = {}
        self._provider_error: asyncio.Future[SpeechRecognitionRuntimeError] | None = None

    def _queue(self) -> asyncio.Queue[SpeechEvent]:
        if self._events is None:
            raise RuntimeError("recognizer event queue is not configured")
        return self._events

    @staticmethod
    def _turn_id(event: Any) -> str:
        order = getattr(event, "turn_order", None)
        return str(order if order is not None else "current")

    def handle_speech_started(self, event: Any) -> None:
        self._queue().put_nowait(SpeechStarted(self._turn_id(event)))

    def handle_turn(self, event: Any) -> None:
        text = str(getattr(event, "transcript", "") or "").strip()
        if not text:
            return
        turn_id = self._turn_id(event)
        revision = self._revisions.get(turn_id, 0) + 1
        self._revisions[turn_id] = revision
        final = bool(getattr(event, "end_of_turn", False))
        state = TranscriptState(turn_id, revision, text, final)
        self._queue().put_nowait(
            FinalTranscript(state) if final else PartialTranscript(state))
        if final:
            self._queue().put_nowait(SpeechStopped(turn_id))

    def handle_error(self, error: Any) -> None:
        message = str(getattr(error, "message", None) or error)
        code = getattr(error, "code", None)
        self._queue().put_nowait(SpeechRecognitionError(message=message, code=code))
        failure = SpeechRecognitionRuntimeError(message)
        future = self._provider_error
        if future is not None and not future.done():
            future.set_result(failure)

    async def _on_speech_started(self, _client: Any, event: Any) -> None:
        self.handle_speech_started(event)

    async def _on_turn(self, _client: Any, event: Any) -> None:
        self.handle_turn(event)

    async def _on_error(self, _client: Any, error: Any) -> None:
        self.handle_error(error)

    async def run(self, audio: AsyncIterator[AudioFrame],
                  events: asyncio.Queue[SpeechEvent]) -> None:
        self._events = events
        handlers = {
            "speech_started": self._on_speech_started,
            "turn": self._on_turn,
            "error": self._on_error,
        }
        factory = self._client_factory or self._build_sdk_client
        client = factory(self.settings, handlers)
        self._client = client
        self._provider_error = asyncio.get_running_loop().create_future()
        parameters = {
            "sample_rate": self.settings.sample_rate,
            "speech_model": self.settings.speech_model,
            "format_turns": self.settings.format_turns,
        }
        if self.settings.min_turn_silence is not None:
            parameters["min_turn_silence"] = self.settings.min_turn_silence
        if self.settings.max_turn_silence is not None:
            parameters["max_turn_silence"] = self.settings.max_turn_silence

        async def pcm_chunks():
            async for frame in audio:
                if frame.sample_width != 2 or frame.channels != 1:
                    raise ValueError("AssemblyAI streaming expects mono PCM16 audio")
                if frame.sample_rate != self.settings.sample_rate:
                    raise ValueError("audio sample rate does not match recognizer settings")
                yield frame.pcm

        connection_parameters = (
            self._parameters_factory(**parameters)
            if self._parameters_factory is not None else parameters
        )
        await client.connect(connection_parameters)
        if self._on_ready is not None:
            self._on_ready()
        stream_task = asyncio.create_task(client.stream(pcm_chunks()))
        error_task = asyncio.ensure_future(asyncio.shield(self._provider_error))
        try:
            done, _ = await asyncio.wait(
                {stream_task, error_task}, return_when=asyncio.FIRST_COMPLETED)
            if error_task in done:
                stream_task.cancel()
                await asyncio.gather(stream_task, return_exceptions=True)
                raise error_task.result()
            error_task.cancel()
            await stream_task
        finally:
            error_task.cancel()
            await client.disconnect(terminate=True)
            self._client = None
            self._provider_error = None

    def _build_sdk_client(self, settings: AssemblyAIRecognizerSettings,
                          handlers: Mapping[str, Callable]):
        try:
            from assemblyai.streaming.v3 import (
                AsyncRealTimeTranscriber,
                RealTimeEvents,
                RealTimeParameters,
                RealTimeTranscriberOptions,
            )
        except ImportError as error:
            raise RuntimeError("Install the 'voice' optional dependencies for AssemblyAI STT") from error
        client = AsyncRealTimeTranscriber(
            RealTimeTranscriberOptions(api_key=self.api_key))
        self._parameters_factory = RealTimeParameters
        client.on(RealTimeEvents.SpeechStarted, handlers["speech_started"])
        client.on(RealTimeEvents.Turn, handlers["turn"])
        client.on(RealTimeEvents.Error, handlers["error"])
        return client

    async def stop(self) -> None:
        client = self._client
        if client is None:
            return
        result = client.disconnect(terminate=True)
        if inspect.isawaitable(result):
            await result
        self._client = None
