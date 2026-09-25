"""Voice-to-action spine: streaming speech to authoritative semantic goal to plan.

Pipeline:

    streaming text
      -> Global Jev  -> capability (arbitration)
      -> Local Jev   -> semantic goal type
      -> span extractor -> open-ended arguments
      -> goal builder   -> SemanticGoal
      -> planner <- world state, capability operators
      -> executor -> observe -> world state

Two authority levels are kept strictly apart:

* `observe_partial` produces an `IntentHypothesis`: revocable, unable to mutate the
  authoritative goal, and able to start only operators marked `speculative_safe`.
* `resolve_final` produces a `SemanticGoal` and executes it under the planner.

When `active_local` is set, Global Jev and that capability's Local Jev run
concurrently. The Local result is speculative until Global Jev confirms the
utterance still belongs to the active capability; an ambiguous Global `None` can
preserve a valid active-local goal, while an explicit capability switch wins.
World state is never passed to Jev, so a capability can never be selected because
an action is currently possible.
"""

from __future__ import annotations

import asyncio
import logging
import time
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Mapping

from ping_ponder.observability import emit

from .confirmation import CONFIRMATION_LEDGER_KEY, ConfirmationLedger
from .executor import ExecutionReport, Executor
from .goal_builder import GoalBuild, GoalBuilder
from .goals import IntentHypothesis, SemanticGoal
from .jev import GlobalJev
from .planner import DeterministicPlanner
from .registry import CapabilityDescriptor, CapabilityRegistry
from .world import WorldState

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PartialOutcome:
    """Result of one partial transcript: a hypothesis plus any speculative execution."""

    hypothesis: IntentHypothesis
    accepted: bool = False
    discard_reason: str | None = None
    speculative_report: ExecutionReport | None = None
    global_capability: str | None = None

    @property
    def acted(self) -> bool:
        return self.speculative_report is not None


@dataclass(frozen=True)
class FinalOutcome:
    """Result of one final transcript: the authoritative goal and its execution."""

    global_capability: str | None
    local_goal_type: str | None
    build: GoalBuild | None
    report: ExecutionReport | None
    active_local_before: str | None
    active_local_after: str | None
    discarded_speculation: str | None = None
    effective_semantic_latency: float = 0.0
    serial_equivalent_latency: float = 0.0
    global_latency: float = 0.0
    local_latency: float = 0.0
    stage_timings: Mapping[str, float] = field(default_factory=dict)
    preserved_partial_speculation: bool = False

    @property
    def latency_saved(self) -> float:
        return max(0.0, self.serial_equivalent_latency - self.effective_semantic_latency)

    @property
    def goal(self) -> SemanticGoal | None:
        return self.build.goal if self.build else None

    @property
    def satisfied(self) -> bool:
        return self.report is not None and self.report.satisfied

    def describe(self) -> str:
        goal = self.goal.describe() if self.goal else "(no goal)"
        plan = self.report.describe() if self.report else "(not planned)"
        return f"{goal} | active_local {self.active_local_before} -> {self.active_local_after} | {plan}"


def normalize_transcript(text: str) -> str:
    """Normalize only representation details that cannot change utterance meaning."""
    return " ".join(unicodedata.normalize("NFKC", text).split()).casefold()


@dataclass(frozen=True)
class SemanticEvaluation:
    """Transcript-derived meaning, deliberately detached from mutable world state."""

    text: str
    normalized_text: str
    active_local_before: str | None
    global_capability: str | None
    capability: str | None
    local_goal_type: str | None
    build: GoalBuild | None
    capability_confidence: float | None = None
    goal_confidence: float | None = None
    discarded_speculation: str | None = None
    effective_semantic_latency: float = 0.0
    serial_equivalent_latency: float = 0.0
    global_latency: float = 0.0
    local_latency: float = 0.0
    local_failed: bool = False
    timings: Mapping[str, float] = field(default_factory=dict)


@dataclass
class SpineStats:
    partials: int = 0
    finals: int = 0
    speculative_starts: int = 0
    speculative_kept: int = 0
    speculative_discarded: int = 0
    active_local_changes: int = 0

    @property
    def speculation_keep_rate(self) -> float:
        return self.speculative_kept / self.speculative_starts if self.speculative_starts else 0.0


class VoiceActionSpine:
    """Wires Global Jev, active Local Jev, goal builder, planner, and executor."""

    def __init__(self, *, registry: CapabilityRegistry, global_jev: GlobalJev,
                 goal_builder: GoalBuilder, world: WorldState,
                 planner: DeterministicPlanner | None = None,
                 executor: Executor | None = None,
                 confirmations: ConfirmationLedger | None = None) -> None:
        self.registry = registry
        self.global_jev = global_jev
        self.goal_builder = goal_builder
        self.planner = planner or DeterministicPlanner()
        # Session-scoped and mutable by design; shared by reference so an epoch change
        # survives the executor replacing world state after every step.
        self.confirmations = confirmations or ConfirmationLedger()
        self.executor = executor or Executor(self.planner, confirmations=self.confirmations)
        self._world = world.updated({CONFIRMATION_LEDGER_KEY: self.confirmations})
        self.active_local: str | None = None
        self.stats = SpineStats()
        self._lock = asyncio.Lock()

    @property
    def world(self) -> WorldState:
        return self._world

    def observe_world(self, changes: Mapping[str, Any]) -> WorldState:
        """External observation path; the executor also updates world state after each step."""
        merged = dict(changes)
        # The ledger is shared by reference and must not be replaced by an observer.
        merged.setdefault(CONFIRMATION_LEDGER_KEY, self.confirmations)
        self._world = self._world.updated(merged)
        emit(logger, "world_state_observed", changes={key: str(value) for key, value in changes.items()})
        return self._world

    def _run_local(self, descriptor: CapabilityDescriptor, utterance: str):
        return descriptor.local_jev.interpret(utterance, descriptor=descriptor)

    async def _evaluate_local(self, descriptor: CapabilityDescriptor, utterance: str,
                              *, final: bool):
        """Run one capability's Local Jev, profile lookup, extraction, and goal build."""
        stage: dict[str, float] = {}
        started = time.monotonic()
        local_started = time.monotonic()
        local = await self._run_local(descriptor, utterance)
        stage["local_s"] = time.monotonic() - local_started
        build = None
        if local.goal_type is not None:
            profile_started = time.monotonic()
            schema = descriptor.schema(local.goal_type)
            stage["profile_selection_s"] = time.monotonic() - profile_started
            builder_started = time.monotonic()
            build = await self.goal_builder.build(
                capability=descriptor.name, schema=schema, utterance=utterance, final=final)
            stage["extraction_s"] = build.extraction_latency_s
            stage["goal_builder_s"] = build.goal_builder_latency_s
            stage["goal_build_wall_s"] = build.goal_build_wall_latency_s
        else:
            stage["profile_selection_s"] = 0.0
            stage["extraction_s"] = 0.0
            stage["goal_builder_s"] = 0.0
            stage["goal_build_wall_s"] = 0.0
        stage["local_pipeline_s"] = time.monotonic() - started
        emit(logger, "local_pipeline_end", capability=descriptor.name,
             goal_type=local.goal_type, timings=stage)
        return local, build, stage

    async def evaluate_semantics(self, text: str, *, active_local: str | None = None,
                                 final: bool = False) -> SemanticEvaluation:
        """Resolve transcript meaning without reading or mutating application state."""
        before_active = active_local if active_local is not None else self.active_local
        started = time.monotonic()
        discarded: str | None = None
        local = None
        build = None
        local_failed = False
        stage_timings: dict[str, float] = {}
        candidate_local_stage: dict[str, float] = {}

        if before_active is not None and before_active in self.registry:
            active_descriptor = self.registry.get(before_active)
            global_task = asyncio.create_task(
                self._timed(self.global_jev.route(text, active_local=before_active)))
            local_task = asyncio.create_task(
                self._timed(self._evaluate_local(active_descriptor, text, final=final)))
            global_result, local_result = await asyncio.gather(
                global_task, local_task, return_exceptions=True)
            if isinstance(global_result, BaseException):
                raise global_result
            route, global_latency = global_result
            if isinstance(local_result, BaseException):
                emit(logger, "local_jev_failed", capability=before_active,
                     error_type=type(local_result).__name__)
                discarded = "local_failed"
                local_failed = True
                self.stats.speculative_discarded += 1
            else:
                (local, build, candidate_local_stage), _ = local_result

            active_local_fallback = (
                route.capability is None and local is not None and local.goal_type is not None)
            if active_local_fallback:
                capability = before_active
                emit(logger, "active_local_fallback", active_local=before_active,
                     goal=local.goal_type, routed=None)
            else:
                capability = route.capability
                if capability != before_active:
                    discarded = f"switched_to_{capability or 'None'}"
                    local = None
                    build = None
                    local_failed = False
                    emit(logger, "speculative_local_discarded", active_local=before_active,
                         routed=capability, reason=discarded)
        else:
            route, global_latency = await self._timed(
                self.global_jev.route(text, active_local=before_active))
            capability = route.capability

        local_latency = candidate_local_stage.get("local_s", 0.0)
        if capability == before_active and local is not None:
            stage_timings.update(candidate_local_stage)
        if capability is not None and local is None and not local_failed:
            descriptor = self.registry.get(capability)
            (local, build, selected_local_stage), _ = await self._timed(
                self._evaluate_local(descriptor, text, final=final))
            local_latency = selected_local_stage.get("local_s", 0.0)
            stage_timings.update(selected_local_stage)

        local_goal_type = local.goal_type if local is not None else None
        extraction_latency = stage_timings.get("extraction_s", 0.0)
        build_latency = stage_timings.get("goal_builder_s", 0.0)
        stage_timings.update({
            "global_s": global_latency,
            "local_s": local_latency,
            "semantic_total_s": time.monotonic() - started,
        })
        emit(logger, "semantic_pipeline_timing", capability=capability,
             goal_type=local_goal_type, timings=stage_timings,
             arbitration="concurrent_active_local" if before_active in self.registry else "global_then_local")

        return SemanticEvaluation(
            text=text,
            normalized_text=normalize_transcript(text),
            active_local_before=before_active,
            global_capability=route.capability,
            capability=capability,
            local_goal_type=local_goal_type,
            build=build,
            capability_confidence=route.confidence,
            goal_confidence=local.confidence if local is not None else None,
            discarded_speculation=discarded,
            effective_semantic_latency=time.monotonic() - started,
            serial_equivalent_latency=(global_latency + local_latency
                                       + extraction_latency + build_latency),
            global_latency=global_latency,
            local_latency=local_latency,
            local_failed=local_failed,
            timings=stage_timings,
        )

    async def apply_partial(self, evaluation: SemanticEvaluation) -> PartialOutcome:
        """Apply a current semantic candidate using speculative-safe operators only."""
        self.stats.partials += 1
        capability = evaluation.capability
        hypothesis = IntentHypothesis(
            capability=capability,
            goal_type=evaluation.local_goal_type,
            transcript=evaluation.text,
            capability_confidence=evaluation.capability_confidence,
            goal_confidence=evaluation.goal_confidence,
        )
        if capability is None:
            self.stats.speculative_discarded += 1
            return PartialOutcome(hypothesis=hypothesis, discard_reason="no_capability",
                                  global_capability=evaluation.global_capability)
        if (evaluation.active_local_before is not None
                and capability != evaluation.active_local_before):
            self.stats.speculative_discarded += 1
            return PartialOutcome(hypothesis=hypothesis, discard_reason="capability_not_active",
                                  global_capability=evaluation.global_capability)
        if evaluation.local_goal_type is None or evaluation.build is None:
            return PartialOutcome(hypothesis=hypothesis, accepted=True,
                                  global_capability=evaluation.global_capability)

        async with self._lock:
            descriptor = self.registry.get(capability)
            schema = descriptor.schema(evaluation.local_goal_type)
            self.stats.speculative_starts += 1
            report = await self.executor.execute(
                evaluation.build.goal, schema, self._world, descriptor,
                final=False, speculation_only=True)
            if report.records:
                self._world = report.world_after
                self._refresh_confirmations(capability)
            self.stats.speculative_kept += 1
        emit(logger, "partial_speculation_executed", capability=capability,
             goal=evaluation.local_goal_type, executed=list(report.executed),
             outcome=report.outcome.value)
        return PartialOutcome(hypothesis=hypothesis, accepted=True,
                              speculative_report=report,
                              global_capability=evaluation.global_capability)

    async def reconcile_final(self, evaluation: SemanticEvaluation) -> FinalOutcome:
        """Apply authoritative meaning against freshly observed application state."""
        self.stats.finals += 1
        async with self._lock:
            capability = evaluation.capability
            before_active = evaluation.active_local_before
            if capability is None:
                self._set_active_local(None)
                return FinalOutcome(
                    global_capability=capability,
                    local_goal_type=None, build=None, report=None,
                    active_local_before=before_active,
                    active_local_after=self.active_local,
                    discarded_speculation=evaluation.discarded_speculation,
                    effective_semantic_latency=evaluation.effective_semantic_latency,
                    serial_equivalent_latency=evaluation.serial_equivalent_latency,
                    global_latency=evaluation.global_latency,
                    local_latency=evaluation.local_latency,
                    stage_timings=evaluation.timings,
                )
            if evaluation.local_failed:
                return FinalOutcome(
                    global_capability=capability,
                    local_goal_type=None, build=None, report=None,
                    active_local_before=before_active,
                    active_local_after=self.active_local,
                    discarded_speculation=evaluation.discarded_speculation or "local_failed",
                    effective_semantic_latency=evaluation.effective_semantic_latency,
                    serial_equivalent_latency=evaluation.serial_equivalent_latency,
                    global_latency=evaluation.global_latency,
                    local_latency=evaluation.local_latency,
                    stage_timings=evaluation.timings,
                )

            self._set_active_local(capability)
            build = evaluation.build.promoted() if evaluation.build is not None else None
            report: ExecutionReport | None = None
            if build is not None and build.complete:
                descriptor = self.registry.get(capability)
                schema = descriptor.schema(evaluation.local_goal_type)
                report = await self.executor.execute(
                    build.goal, schema, self._world, descriptor, final=True)
                self._world = report.world_after
                self._refresh_confirmations(capability)
            elif build is not None:
                emit(logger, "final_goal_incomplete", capability=capability,
                     goal=evaluation.local_goal_type,
                     missing=list(build.missing_slots))

            preserved = report is not None and report.satisfied and any(
                record.step.operator.metadata.allows_speculation() for record in report.records
            ) and evaluation.discarded_speculation is None
            outcome = FinalOutcome(
                global_capability=capability,
                local_goal_type=evaluation.local_goal_type, build=build, report=report,
                active_local_before=before_active, active_local_after=self.active_local,
                discarded_speculation=evaluation.discarded_speculation,
                effective_semantic_latency=evaluation.effective_semantic_latency,
                serial_equivalent_latency=evaluation.serial_equivalent_latency,
                global_latency=evaluation.global_latency,
                local_latency=evaluation.local_latency,
                stage_timings={**evaluation.timings,
                               "planning_s": report.planning_latency_s if report else 0.0},
                preserved_partial_speculation=preserved)
            emit(logger, "final_goal_resolved", capability=capability,
                 goal=evaluation.local_goal_type,
                 arguments={key: str(value) for key, value in build.goal.arguments.items()} if build else {},
                 missing=list(build.missing_slots) if build else [],
                 executed=list(report.executed) if report else [], outcome=report.outcome.value if report else None,
                 discarded_speculation=evaluation.discarded_speculation)
            return outcome

    async def observe_partial(self, text: str, *, active_local: str | None = None) -> PartialOutcome:
        """Compatibility wrapper for text/debug callers."""
        evaluation = await self.evaluate_semantics(text, active_local=active_local, final=False)
        return await self.apply_partial(evaluation)

    async def resolve_final(self, text: str, *, active_local: str | None = None) -> FinalOutcome:
        """Compatibility wrapper for authoritative text/debug callers."""
        evaluation = await self.evaluate_semantics(text, active_local=active_local, final=True)
        return await self.reconcile_final(evaluation)

    @staticmethod
    async def _timed(awaitable):
        """Await a coroutine and report its own latency, for concurrency accounting."""
        started = time.monotonic()
        value = await awaitable
        return value, time.monotonic() - started

    def _set_active_local(self, capability: str | None) -> None:
        if capability != self.active_local:
            emit(logger, "active_local_change", before=self.active_local, after=capability)
            self.active_local = capability
            self.stats.active_local_changes += 1
            # Moving to another capability ends the topic the confirmation belonged to.
            self.confirmations.change_topic()
            self._world = self._world.updated({CONFIRMATION_LEDGER_KEY: self.confirmations})

    def invalidate_stale_confirmation(self, capability: str, subject: str | None) -> None:
        """Drop a pending confirmation for an action that has materially changed."""
        self.confirmations.refresh_if_changed(capability, subject)
        self._world = self._world.updated({CONFIRMATION_LEDGER_KEY: self.confirmations})

    def _refresh_confirmations(self, capability: str | None) -> None:
        """Drop any pending confirmation whose action no longer matches world state.

        Every capability with a `confirmation_subject` is re-checked, not only the one
        just routed to, so an edit made by another capability cannot leave a stale
        confirmation live.
        """
        for descriptor in self.registry:
            if descriptor.confirmation_subject is None:
                continue
            try:
                subject = descriptor.confirmation_subject(self._world)
            except Exception as error:
                emit(logger, "confirmation_subject_failed", capability=descriptor.name,
                     error_type=type(error).__name__)
                subject = None
            self.confirmations.refresh_if_changed(descriptor.name, subject)
        self._world = self._world.updated({CONFIRMATION_LEDGER_KEY: self.confirmations})

    def present_for_confirmation(self, capability: str, subject: str) -> None:
        """Host hook: the assistant has just asked the user to confirm this exact action.

        Only after this call can a following "yes" authorize `subject`. The planner and
        the extractor cannot reach this method.
        """
        self.confirmations.activate(capability, subject)
        self._world = self._world.updated({CONFIRMATION_LEDGER_KEY: self.confirmations})

    def confirm_action(self, capability: str, subject: str) -> bool:
        """Host hook: record an explicit user confirmation for this exact action.

        Returns whether the confirmation was accepted. It is refused when the presented
        action, the capability, or the topic epoch no longer match, so a stale "yes"
        cannot authorize a materially different action.
        """
        accepted = self.confirmations.confirms(capability, subject)
        if not accepted:
            # Refuse loudly rather than setting an authorization flag the operator gate
            # would have to reject later: the caller must not believe this was accepted.
            emit(logger, "confirmation_refused", capability=capability, subject=subject,
                 presented=self.confirmations.current().describe() if self.confirmations.current() else None,
                 topic_epoch=self.confirmations.topic_epoch)
            return False
        # Authorization stays a host-set world key, so the planner still cannot produce it.
        self._world = self._world.updated({"transfer.user_authorized": True})
        return True
