"""Interactive chat session over the voice-to-action spine.

This is a thin observability layer, not a new architecture. It does not change the
spine: `RecordingGlobalJev` and `RecordingLocalJev` wrap the configured Jev objects
to capture *what Jev was given*, which is the interesting part of this design (it
receives the utterance and `active_local` only, never execution state).

Each turn productizes a JSON-serializable trace of the whole pipeline:

    transcript -> Global Jev -> Local Jev -> extraction -> goal
               -> plan <- world state -> execution -> world state

Two details worth noting:

* The displayed plan is re-derived from the pre-execution world state by calling the
  planner again. That is only safe because operator projections are pure; it is a
  direct payoff of keeping planning side-effect free.
* Session objects that are live Python instances (the transaction controller, the
  confirmation ledger) are summarized rather than serialized, since they are not
  world state a client should reason about.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Mapping

from ping_ponder.observability import emit

from .goals import IntentHypothesis, SemanticGoal
from .jev import GlobalRoute, LocalRoute
from .reply import ReplyComposer, ReplyContext, TemplateReplyComposer, classify
from .planner import PlanningFailed
from .registry import CapabilityDescriptor
from .spine import VoiceActionSpine

logger = logging.getLogger(__name__)

# World keys holding live session objects rather than planner facts. They are shown as
# a short summary so the UI can still display confirmation state honestly.
_OPAQUE_PREFIXES = ("session.",)


def jsonable(value: Any) -> Any:
    """Reduce a world value to something a JSON client can render."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [jsonable(item) for item in value]
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        try:
            return jsonable(dump(mode="json"))
        except Exception:
            return repr(value)
    return repr(value)


def summarize_world(world: Mapping[str, Any]) -> dict[str, Any]:
    """Planner-relevant world state, with live session objects summarized.

    Capability-private keys are kept (they are real execution facts) but live objects
    are replaced by a short marker so the client never sees an unserializable handle.
    """
    view: dict[str, Any] = {}
    for key, value in world.items():
        if key.startswith(_OPAQUE_PREFIXES) and not isinstance(value, (bool, int, float, str, type(None))):
            view[key] = type(value).__name__
        else:
            view[key] = jsonable(value)
    return view


def changed_keys(before: Mapping[str, Any], after: Mapping[str, Any]) -> list[str]:
    return sorted(key for key in set(before) | set(after) if before.get(key) != after.get(key))


class RecordingGlobalJev:
    """Global Jev wrapper that records the exact input it was given."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.last_input: dict[str, Any] | None = None
        self.name = getattr(inner, "name", type(inner).__name__)

    async def route(self, utterance: str, *, active_local: str | None = None,
                    context: Mapping[str, Any] | None = None) -> GlobalRoute:
        # Recorded before the call so the trace shows what Jev saw even on failure.
        self.last_input = {"utterance": utterance, "active_local": active_local,
                           "context": dict(context or {})}
        return await self._inner.route(utterance, active_local=active_local, context=context)


class RecordingLocalJev:
    """Local Jev wrapper that records the capability and utterance it was given."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.last_input: dict[str, Any] | None = None
        self.name = getattr(inner, "name", type(inner).__name__)

    async def interpret(self, utterance: str, *, descriptor: CapabilityDescriptor) -> LocalRoute:
        self.last_input = {"utterance": utterance, "capability": descriptor.name,
                           "goal_types": list(descriptor.goal_schemas)}
        return await self._inner.interpret(utterance, descriptor=descriptor)


@dataclass
class TurnTrace:
    """One turn's full pipeline trace, ready to serialize for a client."""

    mode: str
    utterance: str
    global_jev: dict[str, Any] = field(default_factory=dict)
    local_jev: dict[str, Any] = field(default_factory=dict)
    extraction: list[dict[str, Any]] = field(default_factory=list)
    goal: str | None = None
    goal_arguments: dict[str, Any] = field(default_factory=dict)
    missing_slots: list[str] = field(default_factory=list)
    derived_plan: list[dict[str, Any]] = field(default_factory=list)
    executed: list[dict[str, Any]] = field(default_factory=list)
    outcome: str | None = None
    reason: str | None = None
    replans: int = 0
    world_before: dict[str, Any] = field(default_factory=dict)
    world_after: dict[str, Any] = field(default_factory=dict)
    world_changed: list[str] = field(default_factory=list)
    active_local_before: str | None = None
    active_local_after: str | None = None
    discarded_speculation: str | None = None
    hypothesis: dict[str, Any] | None = None
    speculation: dict[str, Any] | None = None
    timing: dict[str, Any] = field(default_factory=dict)
    confirmation: dict[str, Any] = field(default_factory=dict)
    # What would be spoken to the user. Presentation only: the outcome above is decided
    # by the planner and executor, never by the composer.
    reply: str | None = None
    reply_composer: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode, "utterance": self.utterance,
            "global_jev": self.global_jev, "local_jev": self.local_jev,
            "extraction": self.extraction, "goal": self.goal,
            "goal_arguments": self.goal_arguments, "missing_slots": self.missing_slots,
            "derived_plan": self.derived_plan, "executed": self.executed,
            "outcome": self.outcome, "reason": self.reason, "replans": self.replans,
            "world_before": self.world_before, "world_after": self.world_after,
            "world_changed": self.world_changed,
            "active_local": {"before": self.active_local_before, "after": self.active_local_after},
            "discarded_speculation": self.discarded_speculation,
            "hypothesis": self.hypothesis, "speculation": self.speculation,
            "timing": self.timing, "confirmation": self.confirmation,
            "reply": self.reply, "reply_composer": self.reply_composer,
        }


class ChatSession:
    """A conversation over one spine, with a trace per turn.

    Serialized by the spine's own lock, so one session cannot interleave turns; a
    server may hold several sessions independently.
    """

    def __init__(self, spine: VoiceActionSpine, *, global_jev: RecordingGlobalJev,
                 local_jevs: Mapping[str, RecordingLocalJev] | None = None,
                 replies: ReplyComposer | None = None) -> None:
        self.spine = spine
        self.global_jev = global_jev
        self.local_jevs = dict(local_jevs or {})
        self.replies: ReplyComposer = replies or TemplateReplyComposer()
        self.turns: list[TurnTrace] = []
        # Filled in by the wiring; direct construction leaves them unknown so `state()`
        # reports honestly rather than implying a backend that was never selected.
        self.backend: Any = None
        self.extractor_name: str | None = None

    def _local_recorder(self, capability: str) -> RecordingLocalJev | None:
        return self.local_jevs.get(capability)

    def _confirmation_view(self) -> dict[str, Any]:
        ledger = self.spine.confirmations
        context = ledger.current()
        return {
            "topic_epoch": ledger.topic_epoch,
            "presented": context.describe() if context else None,
            "presented_subject": context.subject if context else None,
            "presented_capability": context.capability if context else None,
        }

    def _world_view(self) -> dict[str, Any]:
        return summarize_world(self.spine.world.as_dict())

    def _descriptor(self, capability: str) -> CapabilityDescriptor | None:
        return self.spine.registry.optional(capability)

    def _plan_view(self, goal: SemanticGoal | None, descriptor: CapabilityDescriptor | None,
                   world_before: Mapping[str, Any]) -> list[dict[str, Any]]:
        """Re-derive the plan from the pre-execution world state, for display only.

        Safe because operator projections are pure. Derived from a snapshot so the
        display cannot mutate live session state; failures are reported, not hidden.
        """
        if goal is None or descriptor is None:
            return []
        try:
            schema = descriptor.schema(goal.goal_type)
        except KeyError:
            return []
        from .world import WorldState

        snapshot = WorldState(values=dict(world_before))
        try:
            plan = self.spine.planner.plan(goal, schema, snapshot, descriptor.operators,
                                           limits=descriptor.planning_limits())
        except PlanningFailed as error:
            return [{"unplannable": str(error)}]
        return [{"operator": step.operator.name, "arguments": jsonable(dict(step.arguments))}
                for step in plan.steps]

    async def say(self, text: str, *, mode: str = "final") -> TurnTrace:
        """Process one utterance. `mode='partial'` exercises speculation instead."""
        if mode not in {"final", "partial"}:
            raise ValueError("mode must be 'final' or 'partial'")
        self.global_jev.last_input = None
        for recorder in self.local_jevs.values():
            recorder.last_input = None
        world_before = self._world_view()

        if mode == "partial":
            trace = await self._partial_turn(text, world_before)
        else:
            trace = await self._final_turn(text, world_before)

        trace.confirmation = self._confirmation_view()
        await self._compose_reply(trace)
        self.turns.append(trace)
        return trace

    async def _compose_reply(self, trace: TurnTrace) -> None:
        """Attach the user-facing sentence. Never changes the outcome."""
        situation = classify(outcome=trace.outcome, capability=trace.global_jev.get("capability"),
                             goal=trace.goal, missing_slots=trace.missing_slots,
                             mode=trace.mode, discarded=trace.discarded_speculation)
        context = ReplyContext(
            situation=situation, utterance=trace.utterance,
            capability=trace.global_jev.get("capability"),
            goal_type=trace.local_jev.get("goal_type"), missing_slots=tuple(trace.missing_slots),
            reason=trace.reason, executed=tuple(step["operator"] for step in trace.executed
                                                 if step["succeeded"]))
        try:
            trace.reply = await self.replies.compose(context)
        except Exception as error:  # a reply must never break a turn
            emit(logger, "reply_composition_failed", error_type=type(error).__name__)
            trace.reply = None
            return
        # Report who actually spoke, not who was configured: a failed model silently
        # delegates to the template, and conflating the two hides a broken backend.
        source = getattr(self.replies, "last_source", None)
        trace.reply_composer = (self.replies.name if source in (None, "llm", "unused")
                                else f"{self.replies.name} [{source}]")
        emit(logger, "turn_reply", situation=situation.value, composer=self.replies.name,
             source=source or "primary", reply=trace.reply)

    async def _partial_turn(self, text: str, world_before: dict[str, Any]) -> TurnTrace:
        outcome = await self.spine.observe_partial(text)
        hypothesis: IntentHypothesis = outcome.hypothesis
        recorder = self._local_recorder(hypothesis.capability or "")
        report = outcome.speculative_report
        trace = TurnTrace(
            mode="partial", utterance=text,
            global_jev={"input": self.global_jev.last_input,
                        "capability": outcome.global_capability,
                        "confidence": hypothesis.capability_confidence},
            local_jev={"input": recorder.last_input if recorder else None,
                       "goal_type": hypothesis.goal_type,
                       "confidence": hypothesis.goal_confidence},
            hypothesis={"capability": hypothesis.capability, "goal_type": hypothesis.goal_type,
                        "is_final": hypothesis.is_final, "accepted": outcome.accepted,
                        "discard_reason": outcome.discard_reason},
            outcome=report.outcome.value if report else None,
            reason=report.reason if report else outcome.discard_reason,
            executed=[self._step_view(record) for record in (report.records if report else ())],
            world_before=world_before, world_after=self._world_view(),
            active_local_before=self.spine.active_local, active_local_after=self.spine.active_local,
            speculation={"authoritative": False,
                         "executed": list(report.executed) if report else [],
                         "note": "a partial transcript is revocable and cannot set the goal"},
        )
        trace.world_changed = changed_keys(trace.world_before, trace.world_after)
        emit(logger, "chat_turn", mode="partial", utterance=text,
             capability=outcome.global_capability, executed=trace.speculation["executed"])
        return trace

    async def _final_turn(self, text: str, world_before: dict[str, Any]) -> TurnTrace:
        outcome = await self.spine.resolve_final(text)
        # An active Local Jev is speculative while Global Jev arbitrates. If Global
        # returns None, it may still have been called and then discarded (or yielded
        # no goal). Use the prior active capability for observability in that case;
        # otherwise the UI incorrectly reports "not called".
        trace_capability = outcome.global_capability or outcome.active_local_before or ""
        descriptor = self._descriptor(trace_capability)
        recorder = self._local_recorder(trace_capability)
        report = outcome.report
        trace = TurnTrace(
            mode="final", utterance=text,
            global_jev={"input": self.global_jev.last_input,
                        "capability": outcome.global_capability,
                        "confidence": None},
            local_jev={"input": recorder.last_input if recorder else None,
                       "goal_type": outcome.local_goal_type,
                       "criteria": list(descriptor.goal_schemas) if descriptor else []},
            extraction=[self._extraction_view(record) for record in
                        (outcome.build.extractions if outcome.build else ())],
            goal=outcome.goal.describe() if outcome.goal else None,
            goal_arguments=jsonable(dict(outcome.goal.arguments)) if outcome.goal else {},
            missing_slots=list(outcome.build.missing_slots) if outcome.build else [],
            derived_plan=self._plan_view(outcome.goal, descriptor, report.world_before.as_dict()
                                         if report else {}),
            executed=[self._step_view(record) for record in (report.records if report else ())],
            outcome=report.outcome.value if report else None,
            reason=report.reason if report else None,
            replans=report.replans if report else 0,
            world_before=world_before, world_after=self._world_view(),
            active_local_before=outcome.active_local_before, active_local_after=outcome.active_local_after,
            discarded_speculation=outcome.discarded_speculation,
            timing={"effective_semantic_latency_s": round(outcome.effective_semantic_latency, 4),
                    "serial_equivalent_latency_s": round(outcome.serial_equivalent_latency, 4),
                    "latency_saved_s": round(outcome.latency_saved, 4),
                    "global_s": round(outcome.global_latency, 4),
                    "local_s": round(outcome.local_latency, 4),
                    "stages": {key: round(value, 4)
                               for key, value in outcome.stage_timings.items()}},
        )
        trace.world_changed = changed_keys(trace.world_before, trace.world_after)
        self._auto_present(descriptor, report)
        emit(logger, "chat_turn", mode="final", utterance=text, capability=outcome.global_capability,
             goal=trace.goal, outcome=trace.outcome, executed=[step["operator"] for step in trace.executed])
        return trace

    def _auto_present(self, descriptor: CapabilityDescriptor | None, report: Any) -> None:
        """Offer the confirmed action to the user, as the assistant would.

        A capability that declares `confirmation_subject` and has just become ready to
        confirm is presented here, so the UI can surface a real explicit-yes control.
        Without this, a prepared transfer has no path to authorization in the chat.
        """
        if descriptor is None or descriptor.confirmation_subject is None or report is None:
            return
        if not report.satisfied:
            return
        if self.spine.confirmations.current() is not None:
            return
        try:
            subject = descriptor.confirmation_subject(self.spine.world)
        except Exception as error:
            emit(logger, "confirmation_subject_failed", capability=descriptor.name,
                 error_type=type(error).__name__)
            return
        if not subject:
            return
        presented_capability = descriptor.name
        # A capability opts in to hosted confirmation by declaring a subject.
        self.spine.present_for_confirmation(presented_capability, subject)

    @staticmethod
    def _step_view(record: Any) -> dict[str, Any]:
        return {"operator": record.step.operator.name,
                "arguments": jsonable(dict(record.step.arguments)),
                "observed": jsonable(dict(record.observed)),
                "succeeded": record.succeeded,
                "error": record.error}

    @staticmethod
    def _extraction_view(record: Any) -> dict[str, Any]:
        return {"slot": record.slot, "question": record.question, "filled": record.filled,
                "text": record.text, "confidence": record.confidence,
                "extractor": record.extractor, "reason": record.reason,
                "span": [record.start, record.end] if record.start is not None else None}

    def present_for_confirmation(self, capability: str, subject: str) -> dict[str, Any]:
        self.spine.present_for_confirmation(capability, subject)
        return self._confirmation_view()

    def confirm(self, capability: str, subject: str) -> dict[str, Any]:
        accepted = self.spine.confirm_action(capability, subject)
        view = self._confirmation_view()
        view["accepted"] = accepted
        return view

    def state(self) -> dict[str, Any]:
        """Everything a UI needs to render the session: capabilities, world, epoch."""
        return {
            "capabilities": [
                {"name": descriptor.name, "description": descriptor.description,
                 "goals": [{"goal_type": schema.goal_type,
                            "slots": [{"name": slot.name, "question": slot.question,
                                       "required": slot.required, "kind": slot.kind.value}
                                      for slot in schema.slots]}
                           for schema in descriptor.goal_schemas.values()],
                 "operators": [{"name": operator.name,
                                "parameters": list(operator.parameters),
                                "speculative_safe": operator.metadata.speculative_safe,
                                "requires_final": operator.metadata.requires_final}
                               for operator in descriptor.operators]}
                for descriptor in self.spine.registry],
            "global_jev_choices": list(self.spine.registry.choices()),
            "active_local": self.spine.active_local,
            "world": self._world_view(),
            "confirmation": self._confirmation_view(),
            "stats": {"partials": self.spine.stats.partials, "finals": self.spine.stats.finals,
                      "speculation_started": self.spine.stats.speculative_starts,
                      "speculation_kept": self.spine.stats.speculative_kept,
                      "speculation_discarded": self.spine.stats.speculative_discarded,
                      "active_local_changes": self.spine.stats.active_local_changes},
            # Which Jev and extractor actually answered. Reported, not inferred, so a
            # misconfigured model is visible instead of silently falling back.
            "backends": {
                "jev": getattr(getattr(self, "backend", None), "jev_backend", "unknown"),
                "global_jev": self.global_jev.name,
                "local_jev": next(iter(self.local_jevs.values())).name if self.local_jevs else None,
                "extractor": getattr(self, "extractor_name", None),
                "describe": (self.backend.describe() if getattr(self, "backend", None) else None),
            },
        }
