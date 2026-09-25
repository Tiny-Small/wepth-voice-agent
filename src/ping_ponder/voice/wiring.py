"""Lifecycle wiring for the local concurrent voice runtime."""

from __future__ import annotations

import asyncio
from typing import Callable

from ping_ponder.agentic.spine import VoiceActionSpine
from ping_ponder.agentic.wiring import build_default_spine

from .audio import AudioInput, AudioOutput
from .coordinator import CoordinatorSettings, TurnCoordinator
from .dialogue import DialogueModel, ResponsePolicy
from .events import SpeechEvent
from .fast_voice import FastVoiceModel, ParallelVoiceCoordinator, ParallelVoiceSettings
from .stt import SpeechRecognizer
from .tts import SpeechSynthesizer


class VoiceRuntime:
    """Owns transport, recognizer, coordinator, and orderly shutdown."""

    def __init__(self, *, coordinator: TurnCoordinator,
                 audio_input: AudioInput | None = None,
                 audio_output: AudioOutput | None = None,
                 recognizer: SpeechRecognizer | None = None,
                 synthesizer: SpeechSynthesizer | None = None) -> None:
        self.coordinator = coordinator
        self.audio_input = audio_input
        self.audio_output = audio_output
        self.recognizer = recognizer
        self.synthesizer = synthesizer
        self._recognizer_task: asyncio.Task | None = None
        self._started = False

    async def start(self) -> None:
        if self._started:
            return
        try:
            if self.audio_output is not None and hasattr(self.audio_output, "start"):
                await self.audio_output.start()
            if self.audio_input is not None:
                await self.audio_input.start()
            await self.coordinator.start()
            if self.recognizer is not None:
                if self.audio_input is None:
                    raise RuntimeError("a recognizer requires an audio input")
                self._recognizer_task = asyncio.create_task(
                    self.recognizer.run(self.audio_input.frames(), self.coordinator.events),
                    name="voice-speech-recognizer",
                )
            self._started = True
        except BaseException:
            await self.stop()
            raise

    async def run(self) -> None:
        await self.start()
        if self._recognizer_task is None:
            raise RuntimeError("run() requires a configured recognizer")
        try:
            await self._recognizer_task
        finally:
            await self.stop()

    def dispatch(self, event: SpeechEvent) -> None:
        self.coordinator.dispatch(event)

    async def wait_idle(self) -> None:
        await self.coordinator.wait_idle()

    async def stop(self) -> None:
        errors: list[BaseException] = []

        async def attempt(operation) -> None:
            try:
                await operation()
            except BaseException as error:  # finish the remaining cleanup first
                errors.append(error)

        if self.recognizer is not None:
            await attempt(self.recognizer.stop)
        if self._recognizer_task is not None:
            self._recognizer_task.cancel()
            await asyncio.gather(self._recognizer_task, return_exceptions=True)
            self._recognizer_task = None
        await attempt(self.coordinator.stop)
        if self.synthesizer is not None:
            await attempt(self.synthesizer.stop)
        if self.audio_input is not None:
            await attempt(self.audio_input.stop)
        if self.audio_output is not None and hasattr(self.audio_output, "stop"):
            await attempt(self.audio_output.stop)
        self._started = False
        if errors:
            raise ExceptionGroup("voice runtime cleanup failed", errors)


def build_voice_runtime(*, spine: VoiceActionSpine | None = None,
                        audio_input: AudioInput | None = None,
                        audio_output: AudioOutput | None = None,
                        recognizer: SpeechRecognizer | None = None,
                        response_policy: ResponsePolicy | None = None,
                        dialogue_model: DialogueModel | None = None,
                        synthesizer: SpeechSynthesizer | None = None,
                        event_observer: Callable[[SpeechEvent], None] | None = None,
                        settings: CoordinatorSettings | None = None,
                        fast_voice: FastVoiceModel | None = None,
                        parallel_settings: ParallelVoiceSettings | None = None) -> VoiceRuntime:
    """Assemble replaceable boundaries around the existing control spine."""
    control = spine or build_default_spine()
    if fast_voice is not None:
        coordinator = ParallelVoiceCoordinator(
            spine=control, fast_voice=fast_voice, synthesizer=synthesizer,
            audio_output=audio_output, event_observer=event_observer,
            settings=parallel_settings,
        )
    else:
        coordinator = TurnCoordinator(
            spine=control,
            response_policy=response_policy,
            dialogue_model=dialogue_model,
            synthesizer=synthesizer,
            audio_output=audio_output,
            event_observer=event_observer,
            settings=settings,
        )
    return VoiceRuntime(
        coordinator=coordinator,
        audio_input=audio_input,
        audio_output=audio_output,
        recognizer=recognizer,
        synthesizer=synthesizer,
    )
