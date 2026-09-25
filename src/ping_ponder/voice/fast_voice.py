"""Safe parallel speech path: Llama supplies phrasing, Jev supplies truth."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Callable, Protocol

import httpx

from ping_ponder.agentic.spine import VoiceActionSpine
from ping_ponder.observability import emit
from ping_ponder.progress import ProgressEvent, progress_sink

from .audio import AudioOutput
from .dialogue import ControlContext, ExecutionStatus, clarification_response, corrective_response
from .events import FinalTranscript, SpeechEvent, SpeechStarted
from .tts import SpeechSynthesizer

logger = logging.getLogger(__name__)
FAST_VOICE_MODEL = "meta-llama/llama-3.1-8b-instruct"
FAST_VOICE_PROMPT = """You are Fast Voice, the speaking surface for Jev. Jev alone decides
intent, tools, actions, external facts, and completion. Never claim an action was done
or that external information was found. Never issue tool calls. For CHAT, answer only
ordinary conversation that needs no tools or verification. For ACK, give one brief,
natural acknowledgement such as 'Sure, let me check.' Do not add factual content."""
SAFE_ACKS = frozenset({
    "Sure, let me check.", "Yep, give me a moment.",
    "Okay, let me look that up.", "Sure, give me a moment.",
})


class Route(StrEnum):
    CHAT = "CHAT"
    ACTION = "ACTION"
    UNCERTAIN = "UNCERTAIN"


class SpeechState(StrEnum):
    LISTENING = "LISTENING"
    SPECULATING = "SPECULATING"
    FAST_SPEAKING = "FAST_SPEAKING"
    ACTION_RUNNING = "ACTION_RUNNING"
    FINAL_SPEAKING = "FINAL_SPEAKING"
    INTERRUPTED = "INTERRUPTED"


class SpeechKind(StrEnum):
    FAST = "FAST"
    ACK = "ACK"
    PROGRESS = "PROGRESS"
    FINAL = "FINAL"


@dataclass
class ControlTaskState:
    task_id: str
    turn_id: str
    revision: int
    epoch: int
    started_at: float
    metric: VoiceTurnTelemetry
    running: bool = True
    completed_at: float | None = None
    progress_stage: str = "scheduled"
    progress_playback_started: bool = False
    current_step: str | None = None


def provisional_route(text: str) -> Route:
    """Only a tiny whitelist can speculatively receive conversational content."""
    normalized = " ".join(text.casefold().strip().split()).rstrip(".!?")
    if normalized in {"hi", "hello", "hey", "how are you", "tell me a joke",
                      "thank you", "thanks", "good morning", "good evening"}:
        return Route.CHAT
    if re.match(r"^(open|play|search|find|send|book|navigate|go to|click|browse)\b", normalized):
        return Route.ACTION
    return Route.UNCERTAIN


class FastVoiceModel(Protocol):
    def stream(self, text: str, route: Route) -> AsyncIterator[str]: ...


class OpenAICompatibleFastVoice:
    """Separate request/context on the same OpenAI compatible Llama endpoint."""

    def __init__(self, *, endpoint: str, client: httpx.AsyncClient,
                 model: str = FAST_VOICE_MODEL, api_key: str | None = None) -> None:
        self.endpoint, self.client, self.model, self.api_key = endpoint, client, model, api_key

    async def stream(self, text: str, route: Route) -> AsyncIterator[str]:
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        payload = {"model": self.model, "stream": True, "messages": [
            {"role": "system", "content": FAST_VOICE_PROMPT},
            {"role": "system", "content": f"Mode: {'CHAT' if route is Route.CHAT else 'ACK'}."},
            {"role": "user", "content": text},
        ]}
        async with self.client.stream("POST", self.endpoint, headers=headers, json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line.removeprefix("data:").strip()
                if data == "[DONE]":
                    return
                content = json.loads(data).get("choices", [{}])[0].get("delta", {}).get("content")
                if content:
                    yield str(content)


@dataclass(frozen=True)
class SpeechCommitSettings:
    min_characters: int = 24
    max_delay_s: float = 0.35


async def speakable_chunks(tokens: AsyncIterator[str],
                           settings: SpeechCommitSettings) -> AsyncIterator[str]:
    """Commit clauses; never pass individual model tokens to TTS."""
    buffer = ""
    last_commit = time.monotonic()
    queue: asyncio.Queue[str | None] = asyncio.Queue()

    async def pump() -> None:
        try:
            async for token in tokens:
                queue.put_nowait(token)
        finally:
            queue.put_nowait(None)

    producer = asyncio.create_task(pump())
    try:
        while True:
            remaining = max(0.0, settings.max_delay_s - (time.monotonic() - last_commit))
            try:
                token = await asyncio.wait_for(queue.get(), timeout=remaining) if buffer else await queue.get()
            except asyncio.TimeoutError:
                token = ""
            if token is None:
                break
            buffer += token
            boundary = re.search(r"[.!?;,](?:\s|$)", buffer)
            if boundary and boundary.end() >= settings.min_characters:
                chunk, buffer = buffer[:boundary.end()], buffer[boundary.end():]
                yield chunk.strip()
                last_commit = time.monotonic()
            elif buffer and time.monotonic() - last_commit >= settings.max_delay_s:
                split = buffer.rfind(" ")
                if split >= settings.min_characters:
                    yield buffer[:split].strip()
                    buffer = buffer[split + 1:]
                    last_commit = time.monotonic()
                else:
                    last_commit = time.monotonic()
        await producer
    finally:
        producer.cancel()
        await asyncio.gather(producer, return_exceptions=True)
    if buffer.strip():
        yield buffer.strip()


@dataclass
class VoiceTurnTelemetry:
    turn_id: str
    turn_end: float
    revision: int = 1
    route: Route = Route.UNCERTAIN
    timestamps: dict[str, float] = field(default_factory=dict)
    progress_events: list[ProgressEvent] = field(default_factory=list)
    frontchannel_used: bool = False
    progress_speech_used: bool = False
    progress_updates_spoken: int = 0
    stale_output_cancelled: bool = False
    contradicted_by_jev: bool = False
    acknowledgement_submitted: bool = False

    def mark(self, name: str) -> None:
        if name not in self.timestamps:
            elapsed = time.monotonic() - self.turn_end
            self.timestamps[name] = elapsed
            emit(logger, "voice_timing", turn_id=self.turn_id,
                 revision=self.revision,
                 metric=name, elapsed_seconds=elapsed)


@dataclass(frozen=True)
class ParallelVoiceSettings:
    progress_silence_s: float = 2.5
    max_progress_updates: int = 1
    commit: SpeechCommitSettings = field(default_factory=SpeechCommitSettings)


class ParallelVoiceCoordinator:
    """One speech owner for a final STT turn, Fast Voice, and Jev execution."""

    def __init__(self, *, spine: VoiceActionSpine, fast_voice: FastVoiceModel,
                 synthesizer: SpeechSynthesizer | None = None,
                 audio_output: AudioOutput | None = None,
                 event_observer: Callable[[SpeechEvent], None] | None = None,
                 settings: ParallelVoiceSettings | None = None) -> None:
        self.spine, self.fast_voice = spine, fast_voice
        self.synthesizer, self.audio_output = synthesizer, audio_output
        self.event_observer = event_observer
        self.settings = settings or ParallelVoiceSettings()
        self.events: asyncio.Queue[SpeechEvent] = asyncio.Queue()
        self.state = SpeechState.LISTENING
        self.spoken: list[str] = []
        self.telemetry: dict[str, VoiceTurnTelemetry] = {}
        self._consumer: asyncio.Task | None = None
        self._tasks: set[asyncio.Task] = set()
        self._speech_lock = asyncio.Lock()
        self._epoch = 0
        self._current_turn: str | None = None
        self._current_revision: int | None = None
        self._current_control_task: ControlTaskState | None = None
        self._progress_timers: dict[str, asyncio.Task] = {}

    async def start(self) -> None:
        if self._consumer is None:
            self._consumer = asyncio.create_task(self.run(), name="parallel-voice-events")

    def dispatch(self, event: SpeechEvent) -> None:
        self.events.put_nowait(event)

    def _track(self, coroutine, name: str) -> asyncio.Task:
        task = asyncio.create_task(coroutine, name=name)
        self._tasks.add(task)

        def completed(done: asyncio.Task) -> None:
            self._tasks.discard(done)
            if done.cancelled():
                return
            error = done.exception()
            if error is not None:
                emit(logger, "parallel_voice_task_failed", task=done.get_name(),
                     error_type=type(error).__name__, error=str(error))

        task.add_done_callback(completed)
        return task

    async def run(self) -> None:
        while True:
            event = await self.events.get()
            try:
                if self.event_observer:
                    self.event_observer(event)
                if isinstance(event, SpeechStarted):
                    self._interrupt()
                elif isinstance(event, FinalTranscript):
                    self._begin(event)
            finally:
                self.events.task_done()

    def _interrupt(self, *, cancel_control: bool = False) -> None:
        old = self._current_turn
        self._epoch += 1
        self.state = SpeechState.INTERRUPTED
        if old and old in self.telemetry:
            old_metric = self.telemetry[old]
            old_metric.stale_output_cancelled = True
            old_metric.mark("interrupted")
        else:
            old_metric = None
        control = self._current_control_task
        if cancel_control and control and control.running:
            for task in tuple(self._tasks):
                if task.get_name() == f"control:{control.task_id}":
                    task.cancel()
                    emit(logger, "voice_control_cancelled_by_new_turn",
                         turn_id=control.turn_id, revision=control.revision,
                         task_id=control.task_id)
        progress_task = self._progress_timers.get(control.task_id) if control else None
        if progress_task and not progress_task.done() and control:
            self._emit_progress("progress_cancelled_by_new_turn", control)
        for task in tuple(self._tasks):
            if old and old in task.get_name() and ("fast:" in task.get_name() or
                                                   "speech:" in task.get_name() or
                                                   ("control:" in task.get_name()
                                                    and not (control and control.running))):
                task.cancel()
        if control:
            timer = self._progress_timers.get(control.task_id)
            if timer and not timer.done():
                timer.cancel()
        if self.audio_output:
            self._track(self._stop_audio(old_metric), f"audio-stop:{old}")
        self._current_turn = None
        self._current_revision = None

    async def _stop_audio(self, metric: VoiceTurnTelemetry | None) -> None:
        await self.audio_output.stop_playback()
        if metric:
            metric.mark("audio_stopped")

    def _begin(self, event: FinalTranscript) -> None:
        turn_id = event.state.turn_id
        revision = event.state.revision
        previous = self._current_control_task
        if ((self._current_turn and
             (self._current_turn != turn_id or self._current_revision != revision))
                or (previous and previous.running and
                    (previous.turn_id != turn_id or previous.revision != revision))):
            self._interrupt(cancel_control=True)
        self._current_turn = turn_id
        self._current_revision = revision
        self.state = SpeechState.SPECULATING
        epoch = self._epoch
        telemetry = VoiceTurnTelemetry(turn_id, time.monotonic(), revision=revision)
        self.telemetry[turn_id] = telemetry
        route_ready: asyncio.Future[Route] = asyncio.get_running_loop().create_future()
        text = event.state.text
        hint = provisional_route(text)
        self._track(self._run_fast(text, turn_id, epoch, hint, route_ready, telemetry),
                    f"fast:{turn_id}:r{revision}:e{epoch}")
        self._track(self._run_control(text, turn_id, revision, epoch, route_ready, telemetry),
                    f"control:{turn_id}:r{revision}:e{epoch}")

    def _current(self, turn_id: str, epoch: int) -> bool:
        return self._current_turn == turn_id and self._epoch == epoch

    async def _run_fast(self, text: str, turn_id: str, epoch: int, hint: Route,
                        route_ready: asyncio.Future[Route],
                        metric: VoiceTurnTelemetry) -> None:
        metric.mark("fast_request_start")
        queue: asyncio.Queue[str | None] = asyncio.Queue()

        async def generate() -> None:
            try:
                async for token in self.fast_voice.stream(text, hint):
                    metric.mark("first_fast_token")
                    queue.put_nowait(token)
            finally:
                queue.put_nowait(None)

        producer = self._track(generate(), f"fast:model:{turn_id}")
        try:
            route = await route_ready
            if not self._current(turn_id, epoch):
                return
            if route is Route.CHAT:
                if hint is not Route.CHAT:
                    producer.cancel()
                    metric.mark("chat_request_start")
                    await self._speak_tokens(self.fast_voice.stream(text, Route.CHAT),
                                             turn_id, epoch, SpeechKind.FAST)
                else:
                    await self._speak_tokens(queued_tokens(queue), turn_id, epoch, SpeechKind.FAST)
            elif route in {Route.ACTION, Route.UNCERTAIN}:
                chunks = [chunk async for chunk in queued_tokens(queue)]
                proposed = "".join(chunks).strip()
                ack = proposed if proposed in SAFE_ACKS else "Sure, let me check."
                metric.frontchannel_used = True
                await self._speak_tokens(iter_tokens([ack]), turn_id, epoch, SpeechKind.ACK)
        except asyncio.CancelledError:
            metric.stale_output_cancelled = True
            raise
        except Exception as error:
            emit(logger, "fast_voice_failed", turn_id=turn_id, error=str(error))
            if self._current(turn_id, epoch) and route_ready.done() and route_ready.result() is not Route.CHAT:
                metric.frontchannel_used = True
                await self._speak_tokens(iter_tokens(["Sure, let me check."]),
                                         turn_id, epoch, SpeechKind.ACK)
        finally:
            producer.cancel()
            await asyncio.gather(producer, return_exceptions=True)

    async def _run_control(self, text: str, turn_id: str, revision: int, epoch: int,
                           route_ready: asyncio.Future[Route],
                           metric: VoiceTurnTelemetry) -> None:
        control: ControlTaskState | None = None
        try:
            evaluation = await self.spine.evaluate_semantics(
                text, active_local=getattr(self.spine, "active_local", None), final=True)
            route = (Route.ACTION if evaluation.capability else
                     Route.CHAT if provisional_route(text) is Route.CHAT else Route.UNCERTAIN)
            metric.route = route
            metric.mark("route_decision")
            emit(logger, "voice_route", turn_id=turn_id, route=route.value)
            if not route_ready.done():
                route_ready.set_result(route)
            if route is Route.CHAT:
                return
            if route is Route.UNCERTAIN:
                metric.mark("final_started")
                if self._current(turn_id, epoch):
                    await self._speak_tokens(iter_tokens(["I can't verify that right now."]),
                                             turn_id, epoch, SpeechKind.FINAL)
                metric.mark("final_answer")
                return
            self.state = SpeechState.ACTION_RUNNING
            control = ControlTaskState(
                task_id=f"{turn_id}:r{revision}:e{epoch}", turn_id=turn_id,
                revision=revision, epoch=epoch, started_at=time.monotonic(), metric=metric,
            )
            self._current_control_task = control
            metric.mark("jev_tool_start")
            emit(logger, "voice_control_start", turn_id=turn_id, revision=revision,
                 task_id=control.task_id, goal_type=evaluation.local_goal_type,
                 capability=evaluation.capability)
            self._on_progress(turn_id, epoch, ProgressEvent("TASK_STARTED", "jev"), control)
            if evaluation.local_goal_type == "FIND":
                emit(logger, "voice_level_two_start", turn_id=turn_id,
                     revision=revision, task_id=control.task_id, goal_type="FIND")
            token = progress_sink.set(
                lambda event: self._on_progress(turn_id, epoch, event, control))
            try:
                outcome = await self.spine.reconcile_final(evaluation)
            finally:
                progress_sink.reset(token)
                self._finish_control(control)
            metric.mark("jev_tool_complete")
            report = outcome.report
            status = self._status(report)
            emit(logger, "voice_control_complete", turn_id=turn_id, revision=revision,
                 task_id=control.task_id, goal_type=evaluation.local_goal_type,
                 operators=list(getattr(report, "executed", ())) if report else [],
                 outcome=getattr(report.outcome, "value", report.outcome) if report else None,
                 browser_status=(self.spine.world.get("browser.last_task_status")
                                 if evaluation.local_goal_type == "FIND" and hasattr(self.spine, "world")
                                 else None),
                 elapsed_since_task_start_s=control.completed_at - control.started_at)
            self._on_progress(turn_id, epoch, ProgressEvent(
                "TASK_COMPLETE" if status is ExecutionStatus.SUCCEEDED else "TASK_FAILED", "jev"),
                control)
            context = ControlContext(turn_id, active_capability=evaluation.capability,
                                     execution_status=status,
                                     goal_type=evaluation.local_goal_type,
                                     missing_slots=evaluation.build.missing_slots if evaluation.build else (),
                                     reason=getattr(report, "reason", None))
            if context.missing_slots:
                final = clarification_response(text, context)
            elif status is ExecutionStatus.SUCCEEDED:
                final = "Found it." if evaluation.local_goal_type == "FIND" else "Done."
            elif status is ExecutionStatus.NOT_REQUESTED:
                final = "I need a little more detail to do that."
            else:
                final = corrective_response(context)
            metric.mark("final_started")
            fast_success_after_ack = (
                status is ExecutionStatus.SUCCEEDED
                and evaluation.local_goal_type != "FIND"
                and metric.acknowledgement_submitted
                and control.completed_at is not None
                and control.completed_at - control.started_at < self.settings.progress_silence_s
                and not context.missing_slots
                and self._current(turn_id, epoch)
            )
            if fast_success_after_ack:
                emit(logger, "completion_suppressed_as_redundant", turn_id=turn_id,
                     task_id=control.task_id,
                     elapsed_since_task_start_s=control.completed_at - control.started_at)
            elif self._current(turn_id, epoch):
                spoken = await self._speak_tokens(
                    iter_tokens([final]), turn_id, epoch, SpeechKind.FINAL)
                if spoken and final == "Done.":
                    self._emit_progress("completion_spoken", control)
            metric.mark("final_answer")
        except asyncio.CancelledError:
            if control and control.running:
                self._finish_control(control)
            if control and control.metric.route is Route.ACTION:
                emit(logger, "voice_control_cancelled", turn_id=turn_id,
                     revision=revision, task_id=control.task_id,
                     elapsed_since_task_start_s=time.monotonic() - control.started_at)
            if not route_ready.done():
                route_ready.cancel()
            raise
        except Exception as error:
            if not route_ready.done():
                route_ready.set_result(Route.UNCERTAIN)
                metric.route = Route.UNCERTAIN
                metric.mark("route_decision")
            self._on_progress(turn_id, epoch, ProgressEvent("TASK_FAILED", "jev"), control)
            metric.mark("final_started")
            if self._current(turn_id, epoch):
                await self._speak_tokens(iter_tokens(["I couldn't complete that."]),
                                         turn_id, epoch, SpeechKind.FINAL)
            metric.mark("final_answer")
            emit(logger, "parallel_voice_control_failed", turn_id=turn_id, error=str(error))
        finally:
            timer = self._progress_timers.pop(control.task_id, None) if control else None
            if timer and control and not control.progress_playback_started:
                timer.cancel()

    @staticmethod
    def _status(report) -> ExecutionStatus:
        if report is None:
            return ExecutionStatus.NOT_REQUESTED
        value = str(getattr(report, "outcome", ""))
        return {"SATISFIED": ExecutionStatus.SUCCEEDED,
                "INFEASIBLE": ExecutionStatus.INFEASIBLE,
                "BLOCKED": ExecutionStatus.BLOCKED}.get(value, ExecutionStatus.FAILED)

    def _on_progress(self, turn_id: str, epoch: int, event: ProgressEvent,
                     control: ControlTaskState | None = None) -> None:
        metric = control.metric if control else self.telemetry[turn_id]
        metric.progress_events.append(event)
        emit(logger, "voice_progress", turn_id=turn_id, type=event.type,
             source=event.source, data=dict(event.data), at=event.at,
             task_id=control.task_id if control else None,
             revision=control.revision if control else metric.revision,
             epoch=control.epoch if control else epoch)
        if event.type == "STEP_STARTED":
            if control:
                control.current_step = event.data.get("operator", "step")
        elif event.type == "STEP_COMPLETED":
            if control:
                control.current_step = None
        if (event.type in {"TASK_STARTED", "STEP_STARTED"}
                and control is not None
                and control.task_id not in self._progress_timers):
            self._emit_progress("progress_scheduled", control)
            self._progress_timers[control.task_id] = self._track(
                self._delayed_progress(control), f"speech:progress:{control.task_id}")

    async def _delayed_progress(self, control: ControlTaskState) -> None:
        turn_id, epoch, metric = control.turn_id, control.epoch, control.metric
        delay = self.settings.progress_silence_s - (time.monotonic() - control.started_at)
        if delay > 0:
            await asyncio.sleep(delay)
        self._emit_progress("progress_threshold_reached", control)
        if (not self._current(turn_id, epoch) or not control.running
                or metric.progress_updates_spoken >= self.settings.max_progress_updates):
            if not control.running:
                self._emit_progress("progress_suppressed_task_completed", control)
            return
        text = ("I'm still working on that step." if control.current_step
                else "I'm still working on your request.")
        control.progress_stage = "threshold"
        await self._speak_tokens(iter_tokens([text]), turn_id, epoch,
                                 SpeechKind.PROGRESS, control_task=control)

    async def _speak_tokens(self, tokens: AsyncIterator[str], turn_id: str, epoch: int,
                            kind: SpeechKind, *,
                            control_task: ControlTaskState | None = None) -> bool:
        delivered = False
        async for phrase in speakable_chunks(tokens, self.settings.commit):
            if not self._current(turn_id, epoch):
                return delivered
            async with self._speech_lock:
                if not self._current(turn_id, epoch):
                    return delivered
                if (kind in {SpeechKind.FAST, SpeechKind.ACK}
                        and "final_started" in self.telemetry[turn_id].timestamps):
                    return delivered
                if kind is SpeechKind.PROGRESS:
                    if control_task is None or not self._progress_valid(control_task):
                        if control_task is not None:
                            self._emit_progress("progress_suppressed_before_playback", control_task)
                        return delivered
                    control_task.progress_stage = "queued"
                self.state = SpeechState.FINAL_SPEAKING if kind is SpeechKind.FINAL else SpeechState.FAST_SPEAKING
                if kind is SpeechKind.ACK:
                    self.telemetry[turn_id].acknowledgement_submitted = True
                if kind is not SpeechKind.PROGRESS:
                    self.spoken.append(phrase)
                if kind is SpeechKind.FINAL:
                    emit(logger, "voice_final_speech_submitted", turn_id=turn_id,
                         revision=self.telemetry[turn_id].revision, epoch=epoch,
                         task_id=(self._current_control_task.task_id
                                  if self._current_control_task else None))
                if self.synthesizer:
                    if kind is not SpeechKind.PROGRESS:
                        self.telemetry[turn_id].mark("first_tts_chunk_submitted")
                    frames = self.synthesizer.synthesize_stream(iter_tokens([phrase]))
                    if self.audio_output:
                        if kind is SpeechKind.PROGRESS and control_task is not None:
                            await self.audio_output.play(self._valid_progress_frames(
                                frames, phrase, control_task, epoch, mark_before_first_frame=False))
                        else:
                            await self.audio_output.play(frames)
                    else:
                        if kind is SpeechKind.PROGRESS and control_task is not None:
                            async for _ in self._valid_progress_frames(
                                    frames, phrase, control_task, epoch,
                                    mark_before_first_frame=True):
                                pass
                        else:
                            async for _ in frames:
                                pass
                    delivered = (kind is not SpeechKind.PROGRESS
                                 or control_task is None
                                 or control_task.progress_playback_started)
                elif kind is SpeechKind.PROGRESS and control_task is not None:
                    if not self._progress_valid(control_task):
                        self._emit_progress("progress_suppressed_before_playback", control_task)
                        return delivered
                    self._mark_progress_playback(control_task, phrase)
                    delivered = True
                else:
                    delivered = True
        return delivered

    def _progress_valid(self, control: ControlTaskState) -> bool:
        return (control.running and self._current(control.turn_id, control.epoch))

    async def _valid_progress_frames(self, frames: AsyncIterator, phrase: str,
                                     control: ControlTaskState, epoch: int, *,
                                     mark_before_first_frame: bool) -> AsyncIterator:
        if not self._progress_valid(control) or not self._current(control.turn_id, epoch):
            self._emit_progress("progress_suppressed_before_playback", control)
            return
        control.progress_stage = "tts"
        self.telemetry[control.turn_id].mark("first_tts_chunk_submitted")
        self._emit_progress("progress_submitted_to_tts", control)
        if mark_before_first_frame:
            self._mark_progress_playback(control, phrase)
        first_frame = True
        async for frame in frames:
            if first_frame:
                if (not mark_before_first_frame
                        and (not self._progress_valid(control)
                             or not self._current(control.turn_id, epoch))):
                    control.progress_stage = "suppressed"
                    self._emit_progress("progress_suppressed_before_playback", control)
                    return
                if not mark_before_first_frame:
                    self._mark_progress_playback(control, phrase)
                first_frame = False
            yield frame

    def _mark_progress_playback(self, control: ControlTaskState, phrase: str) -> None:
        if control.progress_playback_started:
            return
        control.progress_playback_started = True
        control.progress_stage = "playback"
        metric = self.telemetry[control.turn_id]
        metric.progress_speech_used = True
        metric.progress_updates_spoken += 1
        self.spoken.append(phrase)
        self._emit_progress("progress_playback_started", control)

    def _finish_control(self, control: ControlTaskState) -> None:
        if not control.running:
            return
        control.running = False
        control.completed_at = time.monotonic()
        timer = self._progress_timers.get(control.task_id)
        if timer and not timer.done() and not control.progress_playback_started:
            if control.progress_stage == "tts":
                self._emit_progress("progress_suppressed_before_playback", control)
            self._emit_progress("progress_suppressed_task_completed", control)
            timer.cancel()

    def _emit_progress(self, event: str, control: ControlTaskState) -> None:
        emit(logger, event, turn_id=control.turn_id, task_id=control.task_id,
             epoch=control.epoch, revision=control.revision,
             elapsed_since_task_start_s=max(0.0, time.monotonic() - control.started_at))

    async def wait_idle(self) -> None:
        await self.events.join()
        while self._tasks:
            await asyncio.gather(*tuple(self._tasks), return_exceptions=True)
        if self.state is not SpeechState.INTERRUPTED:
            self.state = SpeechState.LISTENING

    async def stop(self) -> None:
        self._interrupt()
        if self._consumer:
            self._consumer.cancel()
        for task in tuple(self._tasks):
            task.cancel()
        await asyncio.gather(*tuple(self._tasks), return_exceptions=True)
        if self._consumer:
            await asyncio.gather(self._consumer, return_exceptions=True)
        self._consumer = None


async def iter_tokens(chunks: list[str]) -> AsyncIterator[str]:
    for chunk in chunks:
        yield chunk


async def queued_tokens(queue: asyncio.Queue[str | None]) -> AsyncIterator[str]:
    while (chunk := await queue.get()) is not None:
        yield chunk
