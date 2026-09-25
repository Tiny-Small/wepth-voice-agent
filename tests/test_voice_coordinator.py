import asyncio
from dataclasses import replace
from types import SimpleNamespace

import pytest

from ping_ponder.agentic.goal_builder import GoalBuild
from ping_ponder.agentic.goals import SemanticGoal
from ping_ponder.agentic.spine import SemanticEvaluation, normalize_transcript
from ping_ponder.agentic.wiring import build_default_spine
from ping_ponder.voice.coordinator import TurnCoordinator
from ping_ponder.voice.dialogue import ExecutionStatus
from ping_ponder.voice.events import (
    FinalTranscript,
    PartialTranscript,
    SpeechStarted,
    TranscriptState,
)
from ping_ponder.voice.tts import RecordingSpeechSynthesizer


def partial(turn_id, revision, text):
    return PartialTranscript(TranscriptState(turn_id, revision, text, False))


def final(turn_id, revision, text):
    return FinalTranscript(TranscriptState(turn_id, revision, text, True))


def evaluation(text):
    goal = SemanticGoal("Spotify", "PLAY", {"query": text.split()[-1].casefold()})
    return SemanticEvaluation(
        text=text,
        normalized_text=normalize_transcript(text),
        active_local_before=None,
        global_capability="Spotify",
        capability="Spotify",
        local_goal_type="PLAY",
        build=GoalBuild(goal),
    )


class RecordingSpine:
    def __init__(self):
        self.calls = []
        self.applied = []
        self.reconciled = []
        self.active_local = None

    async def evaluate_semantics(self, text, *, active_local=None, final=False):
        self.calls.append((text, final))
        return evaluation(text)

    async def apply_partial(self, result):
        self.applied.append(result.text)
        return SimpleNamespace(speculative_report=None)

    async def reconcile_final(self, result):
        self.reconciled.append(result.text)
        return SimpleNamespace(
            report=SimpleNamespace(outcome="SATISFIED", reason=None),
            goal=result.build.goal,
        )


@pytest.mark.asyncio
async def test_stale_partial_cannot_publish_or_execute_when_cancellation_is_ignored():
    first_started = asyncio.Event()
    release_first = asyncio.Event()

    class CancellationResistantSpine(RecordingSpine):
        async def evaluate_semantics(self, text, *, active_local=None, final=False):
            self.calls.append((text, final))
            if text == "Play":
                first_started.set()
                try:
                    await release_first.wait()
                except asyncio.CancelledError:
                    await release_first.wait()
            return evaluation(text)

    spine = CancellationResistantSpine()
    coordinator = TurnCoordinator(spine=spine)
    await coordinator.start()

    coordinator.dispatch(partial("t1", 1, "Play"))
    await first_started.wait()
    coordinator.dispatch(partial("t1", 2, "Play some jazz"))
    await asyncio.sleep(0)
    release_first.set()
    await coordinator.wait_idle()

    assert spine.applied == ["Play some jazz"]
    assert coordinator.context.goal == "Spotify.PLAY(query='jazz')"
    assert coordinator.metrics.partials_discarded == 1
    await coordinator.stop()


@pytest.mark.asyncio
async def test_transcript_observer_receives_events_from_coordinator_not_stt_callback():
    observed = []
    coordinator = TurnCoordinator(spine=RecordingSpine(), event_observer=observed.append)
    await coordinator.start()

    first = partial("t1", 1, "Play some")
    last = final("t1", 2, "Play some jazz")
    coordinator.dispatch(first)
    coordinator.dispatch(last)
    await coordinator.wait_idle()

    assert observed == [first, last]
    await coordinator.stop()


@pytest.mark.asyncio
async def test_matching_final_reuses_completed_partial_semantics():
    spine = RecordingSpine()
    coordinator = TurnCoordinator(spine=spine)
    await coordinator.start()

    coordinator.dispatch(partial("t1", 1, "Play some jazz"))
    await coordinator.wait_idle()
    coordinator.dispatch(final("t1", 2, "  PLAY   SOME JAZZ "))
    await coordinator.wait_idle()

    assert len(spine.calls) == 1
    assert spine.reconciled == ["Play some jazz"]
    assert coordinator.metrics.partials_reused == 1
    await coordinator.stop()


@pytest.mark.asyncio
async def test_matching_final_reuses_inflight_partial_semantic_task():
    started = asyncio.Event()
    release = asyncio.Event()

    class GatedSpine(RecordingSpine):
        async def evaluate_semantics(self, text, *, active_local=None, final=False):
            self.calls.append((text, final))
            started.set()
            await release.wait()
            return evaluation(text)

    spine = GatedSpine()
    coordinator = TurnCoordinator(spine=spine)
    await coordinator.start()

    coordinator.dispatch(partial("t1", 1, "Play some jazz"))
    await started.wait()
    coordinator.dispatch(final("t1", 2, "play some jazz"))
    await asyncio.sleep(0)
    release.set()
    await coordinator.wait_idle()

    assert len(spine.calls) == 1
    assert spine.reconciled == ["Play some jazz"]
    await coordinator.stop()


@pytest.mark.asyncio
async def test_punctuation_change_reruns_authoritative_semantics():
    spine = RecordingSpine()
    coordinator = TurnCoordinator(spine=spine)
    await coordinator.start()

    coordinator.dispatch(partial("t1", 1, "Play some jazz"))
    await coordinator.wait_idle()
    coordinator.dispatch(final("t1", 2, "Play some jazz."))
    await coordinator.wait_idle()

    assert spine.calls == [("Play some jazz", False), ("Play some jazz.", True)]
    assert coordinator.context.execution_status is ExecutionStatus.SUCCEEDED
    await coordinator.stop()


@pytest.mark.asyncio
async def test_speech_started_stops_output_without_cancelling_control():
    control_release = asyncio.Event()

    class SlowFinalSpine(RecordingSpine):
        async def reconcile_final(self, result):
            await control_release.wait()
            return await super().reconcile_final(result)

    class RecordingOutput:
        def __init__(self):
            self.stops = 0

        async def stop_playback(self):
            self.stops += 1

    spine = SlowFinalSpine()
    output = RecordingOutput()
    coordinator = TurnCoordinator(spine=spine, audio_output=output)
    await coordinator.start()
    coordinator.dispatch(final("t1", 1, "Play some jazz"))
    await asyncio.sleep(0)

    coordinator.dispatch(SpeechStarted("t2"))
    await coordinator.events.join()
    await asyncio.sleep(0)

    assert output.stops == 1
    assert coordinator.final_control_running
    control_release.set()
    await coordinator.wait_idle()
    assert spine.reconciled == ["Play some jazz"]
    await coordinator.stop()


@pytest.mark.asyncio
async def test_acknowledgement_tts_overlaps_final_control():
    control_started = asyncio.Event()
    control_release = asyncio.Event()
    playback_started = asyncio.Event()

    class SlowFinalSpine(RecordingSpine):
        async def reconcile_final(self, result):
            control_started.set()
            await control_release.wait()
            return await super().reconcile_final(result)

    class RecordingOutput:
        async def play(self, frames):
            playback_started.set()
            async for _ in frames:
                pass

        async def stop_playback(self):
            pass

    spine = SlowFinalSpine()
    synthesizer = RecordingSpeechSynthesizer()
    coordinator = TurnCoordinator(
        spine=spine, synthesizer=synthesizer, audio_output=RecordingOutput(),
    )
    await coordinator.start()
    coordinator.dispatch(partial("t1", 1, "Play some jazz"))
    await coordinator.wait_idle()
    coordinator.dispatch(final("t1", 2, "Play some jazz"))

    await control_started.wait()
    await playback_started.wait()

    assert coordinator.final_control_running
    assert synthesizer.utterances == ["Sure"]
    control_release.set()
    await coordinator.wait_idle()
    await coordinator.stop()


@pytest.mark.asyncio
async def test_acknowledged_failure_emits_one_truthful_correction_even_if_failure_is_fast():
    class FailingSpine(RecordingSpine):
        async def reconcile_final(self, result):
            self.reconciled.append(result.text)
            return SimpleNamespace(
                report=SimpleNamespace(
                    outcome="FAILED", reason="OpenSpotify failed: unavailable",
                ),
                goal=result.build.goal,
            )

    spine = FailingSpine()
    coordinator = TurnCoordinator(spine=spine)
    await coordinator.start()
    coordinator.dispatch(final("t1", 1, "Play some jazz"))
    await coordinator.wait_idle()

    assert coordinator.spoken == ["Sure", "I couldn't open Spotify."]
    assert coordinator.context.execution_status is ExecutionStatus.FAILED
    await coordinator.stop()


@pytest.mark.asyncio
async def test_incomplete_browser_search_speaks_clarification_without_calling_dialogue_model():
    class IncompleteSearchSpine:
        active_local = None

        def __init__(self):
            self.reconciled = []
            self.goal = SemanticGoal("Browser", "SEARCH")

        async def evaluate_semantics(self, text, *, active_local=None, final=False):
            return SemanticEvaluation(
                text=text,
                normalized_text=normalize_transcript(text),
                active_local_before=None,
                global_capability="Browser",
                capability="Browser",
                local_goal_type="SEARCH",
                build=GoalBuild(self.goal, missing_slots=("query",)),
            )

        async def reconcile_final(self, result):
            self.reconciled.append(result.build.goal)
            return SimpleNamespace(report=None, goal=None)

    class UnexpectedDialogueModel:
        def __init__(self):
            self.calls = 0

        async def stream_response(self, user_text, context):
            self.calls += 1
            yield "I guessed what you meant."

    spine = IncompleteSearchSpine()
    model = UnexpectedDialogueModel()
    synthesizer = RecordingSpeechSynthesizer()
    coordinator = TurnCoordinator(
        spine=spine, dialogue_model=model, synthesizer=synthesizer,
    )
    await coordinator.start()
    coordinator.dispatch(final("t-search", 1, "Search YouTube"))
    await coordinator.wait_idle()

    assert coordinator.spoken == [
        "Would you like me to open YouTube, or search for something on YouTube?"
    ]
    assert synthesizer.utterances == coordinator.spoken
    assert model.calls == 0
    assert spine.reconciled == [spine.goal]
    await coordinator.stop()


@pytest.mark.asyncio
async def test_final_complete_search_suppresses_clarification_from_incomplete_partial():
    class VaryingSearchSpine(RecordingSpine):
        async def evaluate_semantics(self, text, *, active_local=None, final=False):
            if final:
                goal = SemanticGoal("Browser", "SEARCH", {"query": "YouTube"})
                return SemanticEvaluation(
                    text=text,
                    normalized_text=normalize_transcript(text),
                    active_local_before=None,
                    global_capability="Browser",
                    capability="Browser",
                    local_goal_type="SEARCH",
                    build=GoalBuild(goal),
                )
            goal = SemanticGoal("Browser", "SEARCH")
            return SemanticEvaluation(
                text=text,
                normalized_text=normalize_transcript(text),
                active_local_before=None,
                global_capability="Browser",
                capability="Browser",
                local_goal_type="SEARCH",
                build=GoalBuild(goal, missing_slots=("query",)),
            )

        async def reconcile_final(self, result):
            self.reconciled.append(result.text)
            return SimpleNamespace(
                report=SimpleNamespace(outcome="SATISFIED", reason=None),
                goal=result.build.goal,
            )

    spine = VaryingSearchSpine()
    synthesizer = RecordingSpeechSynthesizer()
    coordinator = TurnCoordinator(spine=spine, synthesizer=synthesizer)
    await coordinator.start()
    coordinator.dispatch(partial("t-search-final", 1, "Search"))
    await coordinator.wait_idle()
    assert coordinator.context.missing_slots == ("query",)

    coordinator.dispatch(final("t-search-final", 2, "Search YouTube"))
    await coordinator.wait_idle()

    assert len(coordinator.spoken) == 1
    assert coordinator.spoken[0] in {"Sure", "Okay", "Got it"}
    assert coordinator.context.missing_slots == ()
    assert spine.reconciled == ["Search YouTube"]
    await coordinator.stop()


@pytest.mark.asyncio
async def test_next_turn_does_not_cancel_semantics_promoted_by_previous_final():
    first_started = asyncio.Event()
    release_first = asyncio.Event()

    class GatedSpine(RecordingSpine):
        async def evaluate_semantics(self, text, *, active_local=None, final=False):
            self.calls.append((text, final))
            if text == "Play some jazz":
                first_started.set()
                await release_first.wait()
            return evaluation(text)

    spine = GatedSpine()
    coordinator = TurnCoordinator(spine=spine)
    await coordinator.start()
    coordinator.dispatch(partial("t1", 1, "Play some jazz"))
    await first_started.wait()
    coordinator.dispatch(final("t1", 2, "Play some jazz"))
    coordinator.dispatch(partial("t2", 1, "Play some blues"))
    await coordinator.events.join()
    release_first.set()
    await coordinator.wait_idle()

    assert "Play some jazz" in spine.reconciled
    assert ("Play some blues", False) in spine.calls
    await coordinator.stop()


@pytest.mark.asyncio
async def test_older_partial_revision_and_duplicate_final_are_ignored():
    spine = RecordingSpine()
    coordinator = TurnCoordinator(spine=spine)
    await coordinator.start()
    coordinator.dispatch(partial("t1", 2, "Play some jazz"))
    await coordinator.wait_idle()
    coordinator.dispatch(partial("t1", 1, "Play some rock"))
    coordinator.dispatch(final("t1", 3, "Play some jazz"))
    coordinator.dispatch(final("t1", 3, "Play some jazz"))
    await coordinator.wait_idle()

    assert spine.calls == [("Play some jazz", False)]
    assert spine.applied == ["Play some jazz"]
    assert spine.reconciled == ["Play some jazz"]
    await coordinator.stop()


@pytest.mark.asyncio
async def test_incomplete_goal_asks_for_missing_play_query():
    coordinator = TurnCoordinator(spine=build_default_spine())
    await coordinator.start()
    coordinator.dispatch(final("t1", 1, "Play"))
    await coordinator.wait_idle()

    assert coordinator.context.goal is None
    assert coordinator.spoken == ["What would you like me to play?"]
    await coordinator.stop()


@pytest.mark.asyncio
async def test_nonmatching_incomplete_final_clarifies_instead_of_acknowledging_partial_goal():
    release_final = asyncio.Event()

    class ChangedFinalSpine(RecordingSpine):
        async def evaluate_semantics(self, text, *, active_local=None, final=False):
            self.calls.append((text, final))
            if final:
                await release_final.wait()
                result = evaluation(text)
                return replace(result, build=GoalBuild(
                    result.build.goal, missing_slots=("query",),
                ))
            return evaluation(text)

    spine = ChangedFinalSpine()
    coordinator = TurnCoordinator(spine=spine)
    await coordinator.start()
    coordinator.dispatch(partial("t1", 1, "Play some jazz"))
    await coordinator.wait_idle()
    coordinator.dispatch(final("t1", 2, "Play"))
    await coordinator.events.join()
    await asyncio.sleep(0)

    assert coordinator.spoken == []
    release_final.set()
    await coordinator.wait_idle()
    assert coordinator.spoken == ["What would you like me to play?"]
    await coordinator.stop()


@pytest.mark.asyncio
async def test_exception_after_ack_emits_truthful_correction():
    release_failure = asyncio.Event()

    class RaisingSpine(RecordingSpine):
        async def reconcile_final(self, result):
            await release_failure.wait()
            raise RuntimeError("OpenSpotify failed")

    spine = RaisingSpine()
    coordinator = TurnCoordinator(spine=spine)
    await coordinator.start()
    coordinator.dispatch(final("t1", 1, "Play some jazz"))
    while coordinator.spoken != ["Sure"]:
        await asyncio.sleep(0)
    release_failure.set()
    await coordinator.wait_idle()

    assert coordinator.spoken == ["Sure", "I couldn't open Spotify."]
    await coordinator.stop()


@pytest.mark.asyncio
async def test_failure_correction_waits_for_ack_playback_to_finish():
    ack_release = asyncio.Event()
    first_playing = asyncio.Event()

    class FailingSpine(RecordingSpine):
        async def reconcile_final(self, result):
            return SimpleNamespace(
                report=SimpleNamespace(outcome="FAILED", reason="OpenSpotify failed"),
                goal=result.build.goal,
            )

    class GatedOutput:
        def __init__(self):
            self.play_calls = 0

        async def play(self, frames):
            self.play_calls += 1
            if self.play_calls == 1:
                first_playing.set()
                await ack_release.wait()
            async for _ in frames:
                pass

        async def stop_playback(self):
            pass

    output = GatedOutput()
    coordinator = TurnCoordinator(
        spine=FailingSpine(),
        synthesizer=RecordingSpeechSynthesizer(),
        audio_output=output,
    )
    await coordinator.start()
    coordinator.dispatch(final("t1", 1, "Play some jazz"))
    await first_playing.wait()
    await asyncio.sleep(0)

    assert output.play_calls == 1
    ack_release.set()
    await coordinator.wait_idle()
    assert output.play_calls == 2
    await coordinator.stop()
