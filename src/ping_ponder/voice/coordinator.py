"""Asynchronous turn coordination for independent control and dialogue planes."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, replace
from typing import Any, Callable

from ping_ponder.agentic.spine import SemanticEvaluation, VoiceActionSpine, normalize_transcript
from ping_ponder.observability import emit

from .audio import AudioOutput
from .dialogue import (
    ControlContext,
    DialogueModel,
    ExecutionStatus,
    ResponseMode,
    ResponsePolicy,
    RuleBasedResponsePolicy,
    acknowledgement,
    clarification_response,
    corrective_response,
    guarded_dialogue,
)
from .events import (
    FinalTranscript,
    PartialTranscript,
    SpeechEvent,
    SpeechRecognitionError,
    SpeechStarted,
)
from .tts import SpeechSynthesizer

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CoordinatorSettings:
    partial_debounce_ms: int = 0

    def __post_init__(self) -> None:
        if self.partial_debounce_ms < 0:
            raise ValueError("partial_debounce_ms cannot be negative")


@dataclass
class CoordinatorMetrics:
    partials_received: int = 0
    partials_cancelled: int = 0
    partials_started: int = 0
    partials_completed: int = 0
    partials_reused: int = 0
    partials_discarded: int = 0


@dataclass
class _PartialWork:
    state: Any
    semantic_task: asyncio.Task[SemanticEvaluation]
    wrapper_task: asyncio.Task | None = None


class TurnCoordinator:
    """Owns revision gates while keeping STT callbacks enqueue-only."""

    def __init__(self, *, spine: VoiceActionSpine,
                 response_policy: ResponsePolicy | None = None,
                 dialogue_model: DialogueModel | None = None,
                 synthesizer: SpeechSynthesizer | None = None,
                 audio_output: AudioOutput | None = None,
                 event_observer: Callable[[SpeechEvent], None] | None = None,
                 settings: CoordinatorSettings | None = None) -> None:
        self.spine = spine
        self.response_policy = response_policy or RuleBasedResponsePolicy()
        self.dialogue_model = dialogue_model
        self.synthesizer = synthesizer
        self.audio_output = audio_output
        self.event_observer = event_observer
        self.settings = settings or CoordinatorSettings()
        self.events: asyncio.Queue[SpeechEvent] = asyncio.Queue()
        self.metrics = CoordinatorMetrics()
        self.context = ControlContext(turn_id="")
        self.spoken: list[str] = []
        self.last_error: str | None = None

        self._consumer: asyncio.Task | None = None
        self._tasks: set[asyncio.Task] = set()
        self._dialogue_tasks: set[asyncio.Task] = set()
        self._final_control_tasks: set[asyncio.Task] = set()
        self._latest_partial: _PartialWork | None = None
        self._current_state = None
        self._speech_epoch = 0
        self._acknowledged: dict[str, int] = {}
        self._corrected: set[str] = set()
        self._accepted_revisions: dict[str, int] = {}
        self._finalized_turns: set[str] = set()
        self._speech_lock = asyncio.Lock()

    @property
    def final_control_running(self) -> bool:
        return any(not task.done() for task in self._final_control_tasks)

    async def start(self) -> None:
        if self._consumer is None or self._consumer.done():
            self._consumer = asyncio.create_task(
                self.run(), name="voice-turn-coordinator")

    def dispatch(self, event: SpeechEvent) -> None:
        """Enqueue and return; safe to call directly from an STT callback."""
        self.events.put_nowait(event)

    async def run(self) -> None:
        while True:
            event = await self.events.get()
            try:
                self._handle(event)
            finally:
                self.events.task_done()

    def _track(self, task: asyncio.Task, *, dialogue: bool = False,
               final_control: bool = False) -> asyncio.Task:
        self._tasks.add(task)
        if dialogue:
            self._dialogue_tasks.add(task)
        if final_control:
            self._final_control_tasks.add(task)

        def completed(done: asyncio.Task) -> None:
            self._tasks.discard(done)
            self._dialogue_tasks.discard(done)
            self._final_control_tasks.discard(done)
            if not done.cancelled():
                try:
                    error = done.exception()
                    if error is not None:
                        self.last_error = f"{type(error).__name__}: {error}"
                        emit(logger, "voice_task_failed", task=done.get_name(),
                             error_type=type(error).__name__, error=str(error))
                except Exception as error:  # pragma: no cover - defensive callback
                    self.last_error = f"{type(error).__name__}: {error}"

        task.add_done_callback(completed)
        return task

    def _handle(self, event: SpeechEvent) -> None:
        if self.event_observer is not None:
            try:
                self.event_observer(event)
            except Exception as error:
                # Display/telemetry is observational and cannot stop either plane.
                self.last_error = f"event observer {type(error).__name__}: {error}"
        if isinstance(event, SpeechStarted):
            self._speech_epoch += 1
            for task in tuple(self._dialogue_tasks):
                task.cancel()
            if self.audio_output is not None:
                self._track(asyncio.create_task(
                    self.audio_output.stop_playback(), name="voice-barge-in"))
            return
        if isinstance(event, PartialTranscript):
            self._handle_partial(event)
            return
        if isinstance(event, FinalTranscript):
            self._handle_final(event)
            return
        if isinstance(event, SpeechRecognitionError):
            self.last_error = event.message
            emit(logger, "speech_recognition_failed", code=event.code,
                 error=event.message)

    def _cancel_partial(self, work: _PartialWork | None) -> None:
        if work is None:
            return
        cancelled = False
        for task in (work.wrapper_task, work.semantic_task):
            if task is not None and not task.done():
                task.cancel()
                cancelled = True
        if cancelled:
            self.metrics.partials_cancelled += 1
            self.metrics.partials_discarded += 1
            emit(logger, "voice_partial_cancelled", turn_id=work.state.turn_id,
                 revision=work.state.revision)

    def _handle_partial(self, event: PartialTranscript) -> None:
        state = event.state
        self.metrics.partials_received += 1
        accepted = self._accepted_revisions.get(state.turn_id, 0)
        if state.turn_id in self._finalized_turns or state.revision <= accepted:
            self.metrics.partials_discarded += 1
            emit(logger, "voice_partial_discarded", turn_id=state.turn_id,
                 revision=state.revision, accepted_revision=accepted,
                 reason="turn_finalized_or_revision_stale")
            return
        self._accepted_revisions[state.turn_id] = state.revision
        emit(logger, "voice_partial_received", turn_id=state.turn_id,
             revision=state.revision, characters=len(state.text))
        self._cancel_partial(self._latest_partial)
        self._current_state = state
        semantic_task = self._track(asyncio.create_task(
            self._evaluate_partial(state),
            name=f"partial-semantics:{state.turn_id}:{state.revision}"))
        work = _PartialWork(state=state, semantic_task=semantic_task)
        self._latest_partial = work
        work.wrapper_task = self._track(asyncio.create_task(
            self._apply_current_partial(work),
            name=f"partial-application:{state.turn_id}:{state.revision}"))

    async def _evaluate_partial(self, state) -> SemanticEvaluation:
        self.metrics.partials_started += 1
        if self.settings.partial_debounce_ms:
            await asyncio.sleep(self.settings.partial_debounce_ms / 1000)
        result = await self.spine.evaluate_semantics(
            state.text, active_local=getattr(self.spine, "active_local", None), final=False)
        self.metrics.partials_completed += 1
        emit(logger, "voice_partial_semantics_complete", turn_id=state.turn_id,
             revision=state.revision, capability=result.capability,
             goal_type=result.local_goal_type,
             goal_complete=bool(result.build and result.build.complete),
             missing_slots=list(result.build.missing_slots) if result.build else [],
             latency_seconds=result.effective_semantic_latency)
        return result

    def _is_current_partial(self, state) -> bool:
        current = self._current_state
        return (
            current is not None
            and not current.final
            and (current.turn_id, current.revision) == (state.turn_id, state.revision)
        )

    async def _apply_current_partial(self, work: _PartialWork) -> None:
        try:
            result = await asyncio.shield(work.semantic_task)
        except asyncio.CancelledError:
            return
        if not self._is_current_partial(work.state):
            emit(logger, "voice_partial_result_discarded", turn_id=work.state.turn_id,
                 revision=work.state.revision, reason="revision_no_longer_current")
            return
        await self.spine.apply_partial(result)
        emit(logger, "voice_partial_applied", turn_id=work.state.turn_id,
             revision=work.state.revision, capability=result.capability,
             goal_type=result.local_goal_type,
             goal_complete=bool(result.build and result.build.complete))
        if not self._is_current_partial(work.state):
            emit(logger, "voice_partial_application_stale", turn_id=work.state.turn_id,
                 revision=work.state.revision)
            return
        self.context = self._context_from_evaluation(
            work.state.turn_id, result, ExecutionStatus.RUNNING)

    def _handle_final(self, event: FinalTranscript) -> None:
        state = event.state
        accepted = self._accepted_revisions.get(state.turn_id, 0)
        if state.turn_id in self._finalized_turns or state.revision <= accepted:
            return
        self._accepted_revisions[state.turn_id] = state.revision
        self._finalized_turns.add(state.turn_id)
        latest = self._latest_partial
        matching = (
            latest is not None
            and latest.state.turn_id == state.turn_id
            and normalize_transcript(state.text) == normalize_transcript(latest.state.text)
        )
        if matching:
            semantic_task = latest.semantic_task
            self.metrics.partials_reused += 1
            emit(logger, "voice_final_reused_partial", turn_id=state.turn_id,
                 final_revision=state.revision, partial_revision=latest.state.revision,
                 semantic_task_done=semantic_task.done())
            # Final control now owns this task. A later turn must not cancel it as
            # obsolete partial work.
            self._latest_partial = None
        else:
            emit(logger, "voice_final_re_evaluate", turn_id=state.turn_id,
                 revision=state.revision, reason="final_text_differs_from_latest_partial")
            self._cancel_partial(latest)
            self._latest_partial = None
            # The authoritative text changed, so the previous partial goal is not
            # valid response-policy context for this final turn.
            self.context = ControlContext(turn_id=state.turn_id)
            semantic_task = self._track(asyncio.create_task(
                self.spine.evaluate_semantics(
                    state.text,
                    active_local=getattr(self.spine, "active_local", None),
                    final=True,
                ),
                name=f"final-semantics:{state.turn_id}:{state.revision}"))
        self._current_state = state
        goal_ready = asyncio.get_running_loop().create_future()
        epoch = self._speech_epoch
        self._track(asyncio.create_task(
            self._run_final_control(state, semantic_task, goal_ready, epoch),
            name=f"final-control:{state.turn_id}:{state.revision}"),
            final_control=True)
        self._track(asyncio.create_task(
            self._run_dialogue(state, goal_ready, epoch),
            name=f"final-dialogue:{state.turn_id}:{state.revision}"),
            dialogue=True)

    async def _run_final_control(self, state, semantic_task, goal_ready, epoch) -> None:
        turn_context = ControlContext(turn_id=state.turn_id)
        try:
            evaluation = await asyncio.shield(semantic_task)
            emit(logger, "voice_final_semantics_complete", turn_id=state.turn_id,
                 capability=evaluation.capability, goal_type=evaluation.local_goal_type,
                 goal_complete=bool(evaluation.build and evaluation.build.complete),
                 missing_slots=list(evaluation.build.missing_slots) if evaluation.build else [],
                 latency_seconds=evaluation.effective_semantic_latency)
            running = self._context_from_evaluation(
                state.turn_id, evaluation,
                ExecutionStatus.RUNNING
                if evaluation.build is not None and evaluation.build.complete
                else ExecutionStatus.NOT_REQUESTED)
            turn_context = running
            self.context = running
            if not goal_ready.done():
                goal_ready.set_result(running)
            outcome = await self.spine.reconcile_final(evaluation)
            report = outcome.report
            status = self._status_for_report(report)
            reason = getattr(report, "reason", None) if report is not None else None
            finished = replace(running, execution_status=status, reason=reason)
            self.context = finished
            emit(logger, "voice_final_control_complete", turn_id=state.turn_id,
                 capability=finished.active_capability, goal=finished.goal,
                 status=finished.execution_status.value, reason=finished.reason)
            self._schedule_correction(state.turn_id, epoch, finished)
        except asyncio.CancelledError:
            if not goal_ready.done():
                goal_ready.cancel()
            raise
        except Exception as error:
            failed = replace(
                turn_context,
                execution_status=ExecutionStatus.FAILED,
                reason=f"{type(error).__name__}: {error}",
            )
            self.context = failed
            if not goal_ready.done():
                goal_ready.set_result(failed)
            self.last_error = failed.reason
            emit(logger, "voice_final_control_failed", turn_id=state.turn_id,
                 error_type=type(error).__name__, error=str(error))
            self._schedule_correction(state.turn_id, epoch, failed)

    @staticmethod
    def _status_for_report(report) -> ExecutionStatus:
        if report is None:
            return ExecutionStatus.NOT_REQUESTED
        outcome = str(getattr(report, "outcome", ""))
        if outcome == "SATISFIED":
            return ExecutionStatus.SUCCEEDED
        if outcome == "INFEASIBLE":
            return ExecutionStatus.INFEASIBLE
        if outcome == "BLOCKED":
            return ExecutionStatus.BLOCKED
        return ExecutionStatus.FAILED

    @staticmethod
    def _context_from_evaluation(turn_id: str, evaluation: SemanticEvaluation,
                                 status: ExecutionStatus) -> ControlContext:
        goal = (
            evaluation.build.goal.describe()
            if evaluation.build is not None and evaluation.build.complete else None
        )
        return ControlContext(
            turn_id=turn_id,
            active_capability=evaluation.capability,
            goal=goal,
            execution_status=status,
            goal_type=evaluation.local_goal_type,
            missing_slots=(
                evaluation.build.missing_slots if evaluation.build is not None else ()
            ),
        )

    async def _run_dialogue(self, state, goal_ready, epoch) -> None:
        context = self.context if self.context.turn_id == state.turn_id else ControlContext(state.turn_id)
        mode = self.response_policy.decide(state.text, context)
        if mode in {ResponseMode.SILENT, ResponseMode.CLARIFICATION}:
            # Clarification must be based on authoritative final semantics, not
            # an incomplete/stale partial hypothesis for the same turn.
            context = await goal_ready
            mode = self.response_policy.decide(state.text, context)
        if epoch != self._speech_epoch:
            return
        if mode is ResponseMode.CLARIFICATION:
            await self._speak(clarification_response(state.text, context))
        elif mode is ResponseMode.ACK_ONLY:
            self._acknowledged[state.turn_id] = epoch
            await self._speak(acknowledgement(state.turn_id))
            if self.context.turn_id == state.turn_id:
                self._schedule_correction(state.turn_id, epoch, self.context)
        elif mode is ResponseMode.LLM_RESPONSE and self.dialogue_model is not None:
            await self._speak_tokens(guarded_dialogue(
                self.dialogue_model, state.text, context))

    def _schedule_correction(self, turn_id: str, epoch: int,
                             context: ControlContext) -> None:
        failed = context.execution_status in {
            ExecutionStatus.FAILED,
            ExecutionStatus.BLOCKED,
            ExecutionStatus.INFEASIBLE,
        }
        if (not failed or self._acknowledged.get(turn_id) != epoch
                or epoch != self._speech_epoch or turn_id in self._corrected):
            return
        self._corrected.add(turn_id)
        self._track(asyncio.create_task(
            self._speak(corrective_response(context)),
            name=f"failure-correction:{turn_id}"), dialogue=True)

    async def _speak(self, text: str) -> None:
        async def tokens():
            yield text
        await self._speak_tokens(tokens())

    async def _speak_tokens(self, tokens) -> None:
        chunks = [chunk async for chunk in tokens]
        text = "".join(chunks).strip()
        if not text:
            return
        async with self._speech_lock:
            self.spoken.append(text)

            async def replay():
                for chunk in chunks:
                    yield chunk

            if self.synthesizer is None:
                return
            frames = self.synthesizer.synthesize_stream(replay())
            if self.audio_output is None:
                async for _ in frames:
                    pass
            else:
                await self.audio_output.play(frames)

    async def wait_idle(self) -> None:
        await self.events.join()
        while True:
            pending = [task for task in self._tasks if not task.done()]
            if not pending:
                await asyncio.sleep(0)
                if not any(not task.done() for task in self._tasks):
                    return
                continue
            await asyncio.gather(*pending, return_exceptions=True)

    async def stop(self) -> None:
        if self._consumer is not None:
            self._consumer.cancel()
        for task in tuple(self._tasks):
            task.cancel()
        await asyncio.gather(
            *([self._consumer] if self._consumer is not None else []),
            *tuple(self._tasks), return_exceptions=True)
        self._consumer = None
