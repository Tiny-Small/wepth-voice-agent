import asyncio
import logging
from types import SimpleNamespace

import pytest

from ping_ponder.voice.events import AudioFrame, FinalTranscript, SpeechStarted, TranscriptState
from ping_ponder.voice.fast_voice import (
    ParallelVoiceCoordinator, ParallelVoiceSettings, Route, SpeechCommitSettings,
    speakable_chunks,
)
from ping_ponder.progress import publish_progress
from ping_ponder.voice.tts import RecordingSpeechSynthesizer


def final(turn, text):
    return FinalTranscript(TranscriptState(turn, 1, text, True))


class Model:
    def __init__(self, *, chat="Hello there.", ack="Sure, let me check."):
        self.chat, self.ack = chat, ack
        self.calls = []

    async def stream(self, text, route):
        self.calls.append((text, route))
        value = self.chat if route is Route.CHAT else self.ack
        for part in (value[:5], value[5:]):
            await asyncio.sleep(0)
            yield part


class Spine:
    active_local = None

    def __init__(self, capability=None, *, outcome="SATISFIED", delay=None):
        self.capability = capability
        self.outcome = outcome
        self.delay = delay
        self.reconciled = 0
        self.started = asyncio.Event()

    async def evaluate_semantics(self, text, *, active_local=None, final=False):
        return SimpleNamespace(capability=self.capability, local_goal_type="PLAY",
                               build=SimpleNamespace(missing_slots=()))

    async def reconcile_final(self, evaluation):
        self.reconciled += 1
        self.started.set()
        publish_progress("STEP_STARTED", self.capability or "tool", operator="Open")
        if self.delay:
            await self.delay.wait()
        publish_progress("STEP_COMPLETED", self.capability or "tool", operator="Open")
        return SimpleNamespace(report=SimpleNamespace(outcome=self.outcome, reason=None))


@pytest.mark.asyncio
async def test_chat_fast_voice_answers_without_executing_tools():
    spine = Spine()
    coordinator = ParallelVoiceCoordinator(spine=spine, fast_voice=Model())
    await coordinator.start()
    coordinator.dispatch(final("chat", "Hello"))
    await coordinator.wait_idle()
    assert coordinator.spoken == ["Hello there."]
    assert spine.reconciled == 0
    assert coordinator.telemetry["chat"].route is Route.CHAT
    await coordinator.stop()


@pytest.mark.asyncio
async def test_action_model_cannot_make_premature_success_claim():
    release = asyncio.Event()
    spine = Spine("Spotify", delay=release)
    model = Model(ack="I've opened Spotify.")
    coordinator = ParallelVoiceCoordinator(
        spine=spine, fast_voice=model,
        settings=ParallelVoiceSettings(progress_silence_s=0.01))
    await coordinator.start()
    coordinator.dispatch(final("action", "Play jazz"))
    await spine.started.wait()
    while not coordinator.spoken:
        await asyncio.sleep(0)
    assert coordinator.spoken[0] == "Sure, let me check."
    while coordinator.telemetry["action"].progress_updates_spoken == 0:
        await asyncio.sleep(0)
    release.set()
    await coordinator.wait_idle()
    assert coordinator.spoken[-1] == "Done."
    assert spine.reconciled == 1
    await coordinator.stop()


@pytest.mark.asyncio
async def test_slow_action_uses_actual_progress_once_after_threshold():
    release = asyncio.Event()
    spine = Spine("Browser", delay=release)
    coordinator = ParallelVoiceCoordinator(
        spine=spine, fast_voice=Model(),
        settings=ParallelVoiceSettings(progress_silence_s=0.01))
    await coordinator.start()
    coordinator.dispatch(final("slow", "Open a site"))
    await spine.started.wait()
    await asyncio.sleep(0.03)
    assert coordinator.telemetry["slow"].progress_updates_spoken == 1
    assert any(event.type == "STEP_STARTED" for event in coordinator.telemetry["slow"].progress_events)
    release.set()
    await coordinator.wait_idle()
    await coordinator.stop()


@pytest.mark.asyncio
async def test_fast_action_has_no_progress_chatter():
    coordinator = ParallelVoiceCoordinator(
        spine=Spine("Spotify"), fast_voice=Model(),
        settings=ParallelVoiceSettings(progress_silence_s=0.1))
    await coordinator.start()
    coordinator.dispatch(final("fast", "Play jazz"))
    await coordinator.wait_idle()
    assert coordinator.telemetry["fast"].progress_updates_spoken == 0
    await coordinator.stop()


@pytest.mark.asyncio
async def test_interruption_invalidates_stale_fast_output():
    class GatedModel(Model):
        def __init__(self):
            super().__init__()
            self.release = asyncio.Event()

        async def stream(self, text, route):
            await self.release.wait()
            yield f"Response to {text}."

    model = GatedModel()
    coordinator = ParallelVoiceCoordinator(spine=Spine(), fast_voice=model)
    await coordinator.start()
    coordinator.dispatch(final("old", "Hello"))
    await coordinator.events.join()
    coordinator.dispatch(SpeechStarted("new"))
    coordinator.dispatch(final("new", "Hello"))
    model.release.set()
    await coordinator.wait_idle()
    assert coordinator.spoken == ["Response to Hello."]
    assert coordinator.telemetry["old"].stale_output_cancelled
    await coordinator.stop()


@pytest.mark.asyncio
async def test_unknown_external_question_stays_uncertain():
    coordinator = ParallelVoiceCoordinator(spine=Spine(), fast_voice=Model(chat="A made-up fact."))
    await coordinator.start()
    coordinator.dispatch(final("unknown", "What is the current weather?"))
    await coordinator.wait_idle()
    assert "A made-up fact." not in coordinator.spoken
    assert coordinator.telemetry["unknown"].route is Route.UNCERTAIN
    await coordinator.stop()


@pytest.mark.asyncio
async def test_clause_is_committed_before_model_stream_finishes():
    release = asyncio.Event()

    async def tokens():
        yield "Here is the first sentence. "
        await release.wait()
        yield "And the second."

    iterator = speakable_chunks(tokens(), SpeechCommitSettings(min_characters=10))
    first = await asyncio.wait_for(anext(iterator), 0.1)
    assert first == "Here is the first sentence."
    release.set()
    assert [chunk async for chunk in iterator] == ["And the second."]


@pytest.mark.asyncio
async def test_failed_action_gets_failure_completion():
    coordinator = ParallelVoiceCoordinator(spine=Spine("Spotify", outcome="FAILED"),
                                           fast_voice=Model())
    await coordinator.start()
    coordinator.dispatch(final("fail", "Play jazz"))
    await coordinator.wait_idle()
    assert "Done." not in coordinator.spoken
    assert coordinator.spoken[-1] == "I couldn't complete that in Spotify."
    await coordinator.stop()


class GatedProgressSynthesizer:
    def __init__(self, *, ignore_cancel=False):
        self.progress_submitted = asyncio.Event()
        self.progress_release = asyncio.Event()
        self.progress_cancelled = False
        self.ignore_cancel = ignore_cancel

    async def synthesize_stream(self, text):
        phrase = "".join([chunk async for chunk in text])
        if "still working" in phrase:
            self.progress_submitted.set()
            try:
                await self.progress_release.wait()
            except asyncio.CancelledError:
                self.progress_cancelled = True
                if not self.ignore_cancel:
                    raise
                await self.progress_release.wait()
            yield AudioFrame(b"progress", sample_rate=24_000)
        else:
            yield AudioFrame(b"other", sample_rate=24_000)

    async def stop(self):
        pass


class ImmediateProgressSynthesizer:
    async def synthesize_stream(self, text):
        phrase = "".join([chunk async for chunk in text])
        payload = b"progress" if "still working" in phrase else b"other"
        yield AudioFrame(payload, sample_rate=24_000)

    async def stop(self):
        pass


class RecordingOutput:
    def __init__(self, *, hold_progress=False):
        self.played = []
        self.progress_started = asyncio.Event()
        self.release_progress = asyncio.Event()
        self.hold_progress = hold_progress
        self.stopped = 0

    async def play(self, frames):
        async for frame in frames:
            if frame.pcm == b"progress":
                self.progress_started.set()
                if self.hold_progress:
                    await self.release_progress.wait()
            self.played.append(frame.pcm)

    async def stop_playback(self):
        self.stopped += 1
        self.release_progress.set()


@pytest.mark.asyncio
async def test_progress_audio_returning_after_control_completion_is_discarded(caplog):
    release_action = asyncio.Event()
    spine = Spine("Browser", delay=release_action)
    synthesizer = GatedProgressSynthesizer(ignore_cancel=True)
    output = RecordingOutput()
    coordinator = ParallelVoiceCoordinator(
        spine=spine, fast_voice=Model(), synthesizer=synthesizer, audio_output=output,
        settings=ParallelVoiceSettings(progress_silence_s=0.01))
    await coordinator.start()
    caplog.set_level(logging.INFO, logger="ping_ponder.voice.fast_voice")
    coordinator.dispatch(final("late-progress", "Find a repo"))
    await spine.started.wait()
    await asyncio.wait_for(synthesizer.progress_submitted.wait(), 0.2)

    release_action.set()
    while "jev_tool_complete" not in coordinator.telemetry["late-progress"].timestamps:
        await asyncio.sleep(0)
    synthesizer.progress_release.set()
    await coordinator.wait_idle()

    assert b"progress" not in output.played
    assert "I'm still working on that step." not in coordinator.spoken
    assert synthesizer.progress_cancelled
    assert any('"event": "progress_suppressed_before_playback"' in record.message
               for record in caplog.records)
    await coordinator.stop()


@pytest.mark.asyncio
async def test_fast_success_suppresses_generic_done_after_ack(caplog):
    release_action = asyncio.Event()
    spine = Spine("Browser", delay=release_action)
    coordinator = ParallelVoiceCoordinator(
        spine=spine, fast_voice=Model(),
        settings=ParallelVoiceSettings(progress_silence_s=0.1))
    await coordinator.start()
    caplog.set_level(logging.INFO, logger="ping_ponder.voice.fast_voice")
    coordinator.dispatch(final("fast-success", "Open GitHub"))
    await spine.started.wait()
    while not coordinator.spoken:
        await asyncio.sleep(0)
    release_action.set()
    await coordinator.wait_idle()
    assert coordinator.spoken == ["Sure, let me check."]
    assert any('"event": "completion_suppressed_as_redundant"' in record.message
               for record in caplog.records)
    await coordinator.stop()


@pytest.mark.asyncio
async def test_progress_that_started_playing_is_not_cancelled_when_action_completes():
    release_action = asyncio.Event()
    spine = Spine("Browser", delay=release_action)
    output = RecordingOutput(hold_progress=True)
    coordinator = ParallelVoiceCoordinator(
        spine=spine, fast_voice=Model(), synthesizer=ImmediateProgressSynthesizer(),
        audio_output=output,
        settings=ParallelVoiceSettings(progress_silence_s=0.01))
    await coordinator.start()
    coordinator.dispatch(final("progress-started", "Find a repo"))
    await spine.started.wait()
    await asyncio.wait_for(output.progress_started.wait(), 0.2)

    release_action.set()
    while "jev_tool_complete" not in coordinator.telemetry["progress-started"].timestamps:
        await asyncio.sleep(0)
    output.release_progress.set()
    await coordinator.wait_idle()
    assert b"progress" in output.played
    await coordinator.stop()


@pytest.mark.asyncio
async def test_interruption_cancels_progress_tts_and_stops_playback():
    release_action = asyncio.Event()
    spine = Spine("Browser", delay=release_action)
    synthesizer = GatedProgressSynthesizer()
    output = RecordingOutput()
    coordinator = ParallelVoiceCoordinator(
        spine=spine, fast_voice=Model(), synthesizer=synthesizer, audio_output=output,
        settings=ParallelVoiceSettings(progress_silence_s=0.01))
    await coordinator.start()
    coordinator.dispatch(final("old-progress", "Find a repo"))
    await spine.started.wait()
    await asyncio.wait_for(synthesizer.progress_submitted.wait(), 0.2)

    coordinator.dispatch(SpeechStarted("new-turn"))
    await coordinator.events.join()
    synthesizer.progress_release.set()
    release_action.set()
    await coordinator.wait_idle()
    assert synthesizer.progress_cancelled
    assert output.stopped >= 1
    assert b"progress" not in output.played
    await coordinator.stop()


@pytest.mark.asyncio
async def test_new_revision_of_same_turn_invalidates_old_progress():
    release_action = asyncio.Event()
    spine = Spine("Browser", delay=release_action)
    synthesizer = GatedProgressSynthesizer()
    output = RecordingOutput()
    coordinator = ParallelVoiceCoordinator(
        spine=spine, fast_voice=Model(), synthesizer=synthesizer, audio_output=output,
        settings=ParallelVoiceSettings(progress_silence_s=0.01))
    await coordinator.start()
    coordinator.dispatch(final("revisioned", "Find the first repo"))
    await spine.started.wait()
    await asyncio.wait_for(synthesizer.progress_submitted.wait(), 0.2)

    coordinator.dispatch(FinalTranscript(
        TranscriptState("revisioned", 2, "Find the corrected repo", True)))
    await coordinator.events.join()
    release_action.set()
    synthesizer.progress_release.set()
    await coordinator.wait_idle()

    assert coordinator.telemetry["revisioned"].revision == 2
    assert output.stopped >= 1
    assert b"progress" not in output.played
    await coordinator.stop()


@pytest.mark.asyncio
async def test_near_simultaneous_fast_and_control_completion_never_overlaps_playback():
    class Output:
        def __init__(self):
            self.active = 0
            self.maximum = 0

        async def play(self, frames):
            self.active += 1
            self.maximum = max(self.maximum, self.active)
            await asyncio.sleep(0.001)
            async for _ in frames:
                pass
            self.active -= 1

        async def stop_playback(self):
            pass

    output = Output()
    coordinator = ParallelVoiceCoordinator(
        spine=Spine("Spotify"), fast_voice=Model(),
        synthesizer=RecordingSpeechSynthesizer(), audio_output=output)
    await coordinator.start()
    coordinator.dispatch(final("race", "Play jazz"))
    await coordinator.wait_idle()
    assert output.maximum <= 1
    await coordinator.stop()
