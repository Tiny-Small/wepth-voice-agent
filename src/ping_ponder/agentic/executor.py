"""Executor: run a plan one step at a time, observe, update world state, replan.

The executor never assumes the initial plan stays valid. After every step it
observes the actual result, merges it into world state, and replans until the goal
is satisfied or replanning fails. Operator failures are reported, not hidden.
"""

from __future__ import annotations

import inspect
import logging
import time
from contextvars import ContextVar
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any, Mapping

from ping_ponder.observability import emit
from ping_ponder.progress import publish_progress

from .confirmation import CONFIRMATION_LEDGER_KEY, ConfirmationLedger
from .goals import GoalSchema, SemanticGoal
from .planner import DeterministicPlanner, GoalInfeasible, PlanStep, PlanningFailed
from .registry import CapabilityDescriptor
from .world import WorldState

logger = logging.getLogger(__name__)


class ObservationError(RuntimeError):
    """A capability returned invalid or unavailable authoritative state."""


class ExecutionOutcome(StrEnum):
    SATISFIED = "SATISFIED"
    # Speculation ran; the goal is NOT claimed to be satisfied because the
    # transcript was provisional.
    SPECULATED = "SPECULATED"
    # The goal cannot be achieved at all (a policy limit, a missing antecedent). This is
    # a semantic outcome, not an internal search or operator failure.
    INFEASIBLE = "INFEASIBLE"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"


def _signature(step: PlanStep) -> str:
    """Stable identity of a ground step, used to avoid repeating a failed attempt."""
    rendered = ",".join(f"{key}={value!r}" for key, value in sorted(step.arguments.items()))
    return f"{step.operator.name}({rendered})"


@dataclass(frozen=True)
class StepRecord:
    step: PlanStep
    before: WorldState
    after: WorldState
    observed: Mapping[str, Any]
    replanned: bool = False
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.error is None


@dataclass(frozen=True)
class ExecutionReport:
    goal: SemanticGoal
    outcome: ExecutionOutcome
    world_before: WorldState
    world_after: WorldState
    records: tuple[StepRecord, ...] = ()
    reason: str | None = None
    replans: int = 0
    planning_latency_s: float = 0.0

    @property
    def executed(self) -> tuple[str, ...]:
        return tuple(record.step.operator.name for record in self.records if record.succeeded)

    @property
    def satisfied(self) -> bool:
        """True only for authoritative execution, never for a speculative run."""
        return self.outcome is ExecutionOutcome.SATISFIED

    @property
    def speculative(self) -> bool:
        return self.outcome is ExecutionOutcome.SPECULATED

    def describe(self) -> str:
        prefix = " -> ".join(self.executed) if self.executed else "(no actions)"
        return f"{self.outcome.value}: {prefix}" + (f" [{self.reason}]" if self.reason else "")


@dataclass(frozen=True)
class ExecutionLimits:
    max_steps: int = 8
    max_replans: int = 4


class Executor:
    """Observe/replan execution loop over one capability's operators."""

    def __init__(self, planner: DeterministicPlanner, *, limits: ExecutionLimits | None = None,
                 confirmations: ConfirmationLedger | None = None) -> None:
        self.planner = planner
        self.limits = limits or ExecutionLimits()
        self.step_ceiling = self.limits.max_steps
        # Session-scoped and shared by reference. Operators read it through world state,
        # so every world update must re-attach the same object.
        self.confirmations = confirmations or ConfirmationLedger()
        self._planning_elapsed: ContextVar[float] = ContextVar(
            f"executor_planning_elapsed_{id(self)}", default=0.0)

    def _timed_planner_call(self, function, *args, **kwargs):
        started = time.monotonic()
        try:
            return function(*args, **kwargs)
        finally:
            elapsed = time.monotonic() - started
            self._planning_elapsed.set(self._planning_elapsed.get() + elapsed)
            emit(logger, "planner_call_end", method=function.__name__, latency_seconds=elapsed)

    def _with_session(self, world: WorldState) -> WorldState:
        """Re-attach session-scoped objects that are not part of planner state.

        A ledger already present in world state wins: it belongs to the session and may
        have been bound by the host between steps. The executor only supplies its own
        when the world carries none, so callers that never manage a session still work.
        """
        existing = world.get(CONFIRMATION_LEDGER_KEY)
        if isinstance(existing, ConfirmationLedger):
            return world
        return world.updated({CONFIRMATION_LEDGER_KEY: self.confirmations})

    async def _observe(self, descriptor: CapabilityDescriptor,
                       current: WorldState) -> tuple[WorldState, dict[str, Any]]:
        """Read and validate authoritative application facts, if available."""
        if descriptor.observer is None:
            return current, {}
        try:
            produced = descriptor.observer()
            values = await produced if inspect.isawaitable(produced) else produced
            observed = dict(values)
        except Exception as error:
            raise ObservationError(f"capability '{descriptor.name}' observation failed: {error}") from error
        unknown = sorted(set(observed) - set(descriptor.world_schema))
        if unknown:
            raise ObservationError(
                f"capability '{descriptor.name}' observed keys outside world schema: {unknown}")
        return self._with_session(current.updated(observed)), observed

    async def _run_steps(self, goal: SemanticGoal, schema: GoalSchema, world: WorldState,
                         current: WorldState, steps: tuple[PlanStep, ...],
                         records: list[StepRecord], replans: int,
                         descriptor: CapabilityDescriptor) -> ExecutionReport:
        """Run a fixed speculative step list, observing each result."""
        for step in steps:
            operator = step.operator
            if operator.executor is None:
                return ExecutionReport(goal=goal, outcome=ExecutionOutcome.FAILED,
                                       world_before=world, world_after=current, records=tuple(records),
                                       reason=f"operator '{operator.name}' has no executor", replans=replans)
            if not operator.preconditions_hold(current, step.arguments):
                continue
            emit(logger, "executor_step_start", operator=operator.name, speculative=True,
                 arguments={key: str(value) for key, value in step.arguments.items()})
            try:
                produced = operator.executor(current, step.arguments)
                observed = dict(await produced if inspect.isawaitable(produced) else produced)
            except Exception as error:
                emit(logger, "executor_step_failed", operator=operator.name, speculative=True,
                     error_type=type(error).__name__, error=str(error))
                records.append(StepRecord(step=step, before=current, after=current, observed={},
                                          error=f"{type(error).__name__}: {error}"))
                continue
            undeclared = sorted(set(observed) - {effect.key for effect in operator.effects})
            if undeclared:
                records.append(StepRecord(step=step, before=current, after=current, observed=observed,
                                          error=f"executor observed undeclared effects: {undeclared}"))
                continue
            before = current
            updated = self._with_session(current.updated(observed))
            try:
                updated, authoritative = await self._observe(descriptor, updated)
            except ObservationError as error:
                records.append(StepRecord(step=step, before=before, after=before, observed=observed,
                                          error=str(error)))
                continue
            merged_observed = dict(observed)
            merged_observed.update(authoritative)
            if (descriptor.observer is not None and not operator.metadata.self_observing
                    and updated.fingerprint() == before.fingerprint()):
                records.append(StepRecord(step=step, before=before, after=updated,
                                          observed=merged_observed, error="no observed progress"))
                continue
            records.append(StepRecord(step=step, before=before, after=updated, observed=merged_observed))
            current = updated
            emit(logger, "executor_step_end", operator=operator.name, speculative=True,
                 observed=dict(merged_observed))
        return ExecutionReport(
            goal=goal,
            outcome=ExecutionOutcome.SPECULATED if records else ExecutionOutcome.BLOCKED,
            world_before=world, world_after=current, records=tuple(records),
            reason=None if records else "no speculative-safe prerequisite applies", replans=replans)

    async def execute(self, goal: SemanticGoal, schema: GoalSchema, world: WorldState,
                      descriptor: CapabilityDescriptor, *,
                      final: bool = True, speculation_only: bool = False) -> ExecutionReport:
        token = self._planning_elapsed.set(0.0)
        try:
            report = await self._execute(goal, schema, world, descriptor,
                                         final=final, speculation_only=speculation_only)
            elapsed = self._planning_elapsed.get()
            emit(logger, "planning_end", goal_type=goal.goal_type, latency_seconds=elapsed)
            return replace(report, planning_latency_s=elapsed)
        finally:
            self._planning_elapsed.reset(token)

    async def _execute(self, goal: SemanticGoal, schema: GoalSchema, world: WorldState,
                       descriptor: CapabilityDescriptor, *,
                       final: bool = True, speculation_only: bool = False) -> ExecutionReport:
        current = self._with_session(world)
        records: list[StepRecord] = []
        replans = 0
        try:
            current, _ = await self._observe(descriptor, current)
        except ObservationError as error:
            return ExecutionReport(goal=goal, outcome=ExecutionOutcome.FAILED,
                                   world_before=world, world_after=current, records=(), reason=str(error))
        if self._timed_planner_call(self.planner.satisfied, goal, schema, current):
            return ExecutionReport(goal=goal, outcome=ExecutionOutcome.SATISFIED,
                                   world_before=world, world_after=current)
        if speculation_only:
            # Speculation follows the speculative-safe prerequisite closure rather
            # than requiring the goal itself to become satisfiable; a partial
            # transcript may lack the arguments the final step needs.
            steps = self._timed_planner_call(
                self.planner.speculative_steps, goal, schema, current, descriptor.operators,
                max_steps=self.limits.max_steps)
            report = await self._run_steps(goal, schema, world, current, steps, records, replans, descriptor)
            return report
        failed_signatures: set[str] = set()
        while len(records) < self.limits.max_steps:
            try:
                plan = self._timed_planner_call(
                    self.planner.plan, goal, schema, current, descriptor.operators,
                    final=final, speculation_only=speculation_only,
                    exclude=failed_signatures, limits=descriptor.planning_limits())
            except GoalInfeasible as error:
                # A semantic outcome: this goal cannot be achieved. Report the reasons,
                # not a search artefact, so the reply layer can say something true.
                emit(logger, "goal_infeasible", goal=goal.describe(), reasons=list(error.reasons))
                return ExecutionReport(goal=goal, outcome=ExecutionOutcome.INFEASIBLE,
                                       world_before=world, world_after=current, records=tuple(records),
                                       reason="; ".join(error.reasons), replans=replans)
            except PlanningFailed as error:
                # Surface why the goal became unreachable, including an operator that
                # execution rejected, so a blocked report is diagnosable.
                failures = [record.error for record in records if record.error]
                reason = str(error)
                if failures:
                    reason = f"{reason}; failed steps: {'; '.join(failures)}"
                return ExecutionReport(goal=goal, outcome=ExecutionOutcome.BLOCKED,
                                       world_before=world, world_after=current, records=tuple(records),
                                       reason=reason, replans=replans)
            if plan.empty:
                return ExecutionReport(goal=goal, outcome=ExecutionOutcome.SATISFIED,
                                       world_before=world, world_after=current, records=tuple(records),
                                       replans=replans)
            step = plan.steps[0]
            operator = step.operator
            signature = _signature(step)
            if not operator.preconditions_hold(current, step.arguments):
                # World changed under us; replan rather than force the step.
                replans += 1
                if replans > self.limits.max_replans:
                    return ExecutionReport(goal=goal, outcome=ExecutionOutcome.BLOCKED,
                                           world_before=world, world_after=current, records=tuple(records),
                                           reason=f"preconditions for {operator.name} no longer hold",
                                           replans=replans)
                emit(logger, "executor_replan", operator=operator.name,
                     unmet=list(operator.unsatisfied_preconditions(current, step.arguments)))
                continue
            if operator.executor is None:
                return ExecutionReport(goal=goal, outcome=ExecutionOutcome.FAILED,
                                       world_before=world, world_after=current, records=tuple(records),
                                       reason=f"operator '{operator.name}' has no executor", replans=replans)
            emit(logger, "executor_step_start", operator=operator.name,
                 arguments={key: str(value) for key, value in step.arguments.items()})
            publish_progress("STEP_STARTED", descriptor.name, operator=operator.name)
            try:
                produced = operator.executor(current, step.arguments)
                # Executors may be sync or async; awaiting a non-awaitable result is an error, not a fallback.
                observed = dict(await produced if inspect.isawaitable(produced) else produced)
            except Exception as error:  # operator failure is observed, then replanned
                emit(logger, "executor_step_failed", operator=operator.name,
                     error_type=type(error).__name__, error=str(error))
                record = StepRecord(step=step, before=current, after=current, observed={},
                                    error=f"{type(error).__name__}: {error}")
                records.append(record)
                # Remember the failure so replanning tries a different route instead of
                # retrying the same operator indefinitely.
                failed_signatures.add(signature)
                replans += 1
                if replans > self.limits.max_replans:
                    return ExecutionReport(goal=goal, outcome=ExecutionOutcome.FAILED,
                                           world_before=world, world_after=current, records=tuple(records),
                                           reason=f"{operator.name} failed: {error}", replans=replans)
                continue
            # Guard: an executor must not report changes its operator never declared,
            # otherwise the planner's projection silently diverges from reality.
            undeclared = sorted(set(observed) - {effect.key for effect in operator.effects})
            if undeclared:
                record = StepRecord(step=step, before=current, after=current, observed=observed,
                                    error=f"executor observed undeclared effects: {undeclared}")
                records.append(record)
                return ExecutionReport(goal=goal, outcome=ExecutionOutcome.FAILED,
                                       world_before=world, world_after=current, records=tuple(records),
                                       reason=f"{operator.name} observed undeclared effects {undeclared}",
                                       replans=replans)
            before = current
            updated = self._with_session(current.updated(observed))
            try:
                if operator.metadata.self_observing:
                    authoritative = {}
                else:
                    updated, authoritative = await self._observe(descriptor, updated)
            except ObservationError as error:
                record = StepRecord(step=step, before=before, after=before, observed=observed,
                                    error=str(error))
                records.append(record)
                return ExecutionReport(goal=goal, outcome=ExecutionOutcome.FAILED,
                                       world_before=world, world_after=before, records=tuple(records),
                                       reason=str(error), replans=replans)
            merged_observed = dict(observed)
            merged_observed.update(authoritative)
            if descriptor.observer is not None and updated.fingerprint() == before.fingerprint():
                record = StepRecord(step=step, before=before, after=updated,
                                    observed=merged_observed, error="no observed progress")
                records.append(record)
                failed_signatures.add(signature)
                replans += 1
                if replans > self.limits.max_replans:
                    return ExecutionReport(goal=goal, outcome=ExecutionOutcome.FAILED,
                                           world_before=world, world_after=updated,
                                           records=tuple(records), reason="no observed progress",
                                           replans=replans)
                continue
            record = StepRecord(step=step, before=before, after=updated, observed=merged_observed)
            records.append(record)
            current = updated
            publish_progress("STEP_COMPLETED", descriptor.name, operator=operator.name)
            if operator.metadata.terminal_on_execution:
                satisfied = self._timed_planner_call(self.planner.satisfied, goal, schema, current)
                status = ExecutionOutcome.SATISFIED if satisfied else ExecutionOutcome.BLOCKED
                reason = None if satisfied else (
                    f"delegated operation ended with {current.get('browser.last_task_status') or 'no status'}"
                )
                return ExecutionReport(goal=goal, outcome=status, world_before=world,
                                       world_after=current, records=tuple(records),
                                       reason=reason, replans=replans)
            if schema.command and operator.satisfies(goal.goal_type):
                return ExecutionReport(goal=goal, outcome=ExecutionOutcome.SATISFIED,
                                       world_before=world, world_after=current,
                                       records=tuple(records), replans=replans)
            emit(logger, "executor_step_end", operator=operator.name,
                 observed=dict(merged_observed),
                 goal_satisfied=self._timed_planner_call(self.planner.satisfied, goal, schema, current))
        if self._timed_planner_call(self.planner.satisfied, goal, schema, current):
            return ExecutionReport(goal=goal, outcome=ExecutionOutcome.SATISFIED,
                                   world_before=world, world_after=current, records=tuple(records),
                                   replans=replans)
        return ExecutionReport(goal=goal, outcome=ExecutionOutcome.BLOCKED,
                               world_before=world, world_after=current, records=tuple(records),
                               reason=f"step budget {self.limits.max_steps} exhausted", replans=replans)
