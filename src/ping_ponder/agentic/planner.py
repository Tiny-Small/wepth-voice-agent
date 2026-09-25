"""Deterministic, bounded forward planner over registered action operators.

The planner is given a semantic goal, the current world state, and the
capability's operators. It searches for the cheapest short action sequence whose
effects make the goal's satisfaction conditions hold. It is not LLM-based and it
never sees the transcript.

Goal satisfaction is tested against world state (`GoalSchema.satisfied_when`),
with goal arguments bound into any `Arg(...)` conditions, so the same semantic
goal produces different plans under different world states.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Mapping

from ping_ponder.observability import emit

from .goals import GoalSchema, SemanticGoal
from .operators import ActionOperator, OperatorError
from .world import WorldState

logger = logging.getLogger(__name__)


def _step_key(step: "PlanStep") -> str:
    rendered = ",".join(f"{key}={value!r}" for key, value in sorted(step.arguments.items()))
    return f"{step.operator.name}({rendered})"


class PlanningFailed(RuntimeError):
    """No bounded sequence of registered operators satisfies the goal."""


class GoalInfeasible(PlanningFailed):
    """The goal cannot be achieved by any plan, not merely unfound within the bounds.

    Raised when the search has established that a precondition nothing can produce is
    blocking the goal - a policy limit, a missing antecedent, or a presupposition. It is
    a *semantic* outcome ("this cannot be done") and belongs in a user-facing reply,
    unlike an internal search failure.
    """

    def __init__(self, goal: str, reasons: tuple[str, ...]) -> None:
        self.goal = goal
        self.reasons = reasons
        detail = "; ".join(reasons) if reasons else "no applicable operator"
        super().__init__(f"cannot achieve {goal}: {detail}")


@dataclass(frozen=True)
class PlanStep:
    operator: ActionOperator
    arguments: Mapping[str, Any]

    def describe(self) -> str:
        rendered = ", ".join(f"{key}={value!r}" for key, value in sorted(self.arguments.items()))
        return f"{self.operator.name}({rendered})"


@dataclass(frozen=True)
class Plan:
    goal: SemanticGoal
    steps: tuple[PlanStep, ...]
    total_cost: float

    @property
    def empty(self) -> bool:
        return not self.steps

    def describe(self) -> str:
        if self.empty:
            return f"satisfied({self.goal.describe()})"
        return " -> ".join(step.describe() for step in self.steps)


@dataclass(frozen=True)
class PlanningLimits:
    """Search bounds. Depth must cover the longest operator chain a capability needs.

    The default covers a short media/browser chain; a capability with a longer
    prerequisite chain raises its own limits rather than making every plan search
    deeper.
    """

    max_depth: int = 4
    max_candidates: int = 512

    @classmethod
    def for_chain(cls, longest_chain: int, *, slack: int = 2) -> "PlanningLimits":
        depth = max(2, longest_chain + slack)
        return cls(max_depth=depth, max_candidates=max(512, 64 * depth))


@dataclass
class _SearchState:
    world: WorldState
    steps: tuple[PlanStep, ...]
    cost: float


class DeterministicPlanner:
    """Cheapest-first bounded BFS over ground operator instances."""

    def __init__(self, *, limits: PlanningLimits | None = None) -> None:
        self.limits = limits or PlanningLimits()

    def satisfied(self, goal: SemanticGoal, schema: GoalSchema, world: WorldState) -> bool:
        """A command goal is satisfied once its effect is already present in world state."""
        if schema.command:
            return bool(schema.already_done_when) and world.conditions_hold(schema.already_done_when)
        # A partial goal with required slots missing cannot be satisfied yet.
        # Keep safe prerequisite speculation possible without evaluating bound
        # conditions such as SEARCH(query=...) against absent arguments.
        if schema.missing_slots(goal.arguments):
            return False
        return world.conditions_hold(schema.satisfied_when, goal.arguments)

    def unsatisfied(self, goal: SemanticGoal, schema: GoalSchema, world: WorldState) -> tuple[str, ...]:
        if schema.command:
            if not schema.already_done_when:
                return (f"{goal.capability}.{goal.goal_type} has not run yet",)
            return world.unmet(schema.already_done_when)
        return world.unmet(schema.satisfied_when, goal.arguments)

    def ground(self, operators: tuple[ActionOperator, ...], goal: SemanticGoal) -> tuple[PlanStep, ...]:
        """Bind each operator's parameters from the goal arguments; skip unbound ones."""
        grounded: list[PlanStep] = []
        for operator in operators:
            try:
                bound = operator.bind(goal.arguments)
            except OperatorError:
                continue
            grounded.append(PlanStep(operator=operator, arguments=bound))
        return tuple(grounded)

    def prerequisite_keys(self, schema: GoalSchema, operators: tuple[ActionOperator, ...]) -> tuple[str, ...]:
        """World keys that must be true to reach the goal, transitively.

        Seeds are the goal's satisfaction keys; the closure follows producers'
        preconditions. This is what makes a prerequisite such as `spotify.running`
        speculatively startable even though opening Spotify does not itself satisfy
        PLAY (its effect is not one of the goal's satisfaction conditions).
        """
        seeds = {condition.key for condition in schema.satisfied_when}
        closure: set[str] = set()
        frontier = list(seeds)
        while frontier:
            key = frontier.pop()
            for operator in operators:
                if not any(effect.key == key for effect in operator.effects):
                    continue
                for condition in operator.preconditions:
                    if condition.key not in closure and condition.key not in seeds:
                        closure.add(condition.key)
                        frontier.append(condition.key)
        return tuple(sorted(closure))

    def speculative_steps(self, goal: SemanticGoal, schema: GoalSchema, world: WorldState,
                          operators: tuple[ActionOperator, ...], *, max_steps: int = 4) -> tuple[PlanStep, ...]:
        """Greedy speculative-safe prerequisite closure.

        Only operators declaring `speculative_safe` are considered, and only when
        their preconditions hold now. Operators whose effects are unreachable
        irrelevant keys are skipped, so a partial utterance cannot wander.
        """
        relevant = set(self.prerequisite_keys(schema, operators)) | {c.key for c in schema.satisfied_when}
        current = world
        chosen: list[PlanStep] = []
        for _ in range(max_steps):
            best: PlanStep | None = None
            for step in self.ground(operators, goal):
                operator = step.operator
                if not operator.metadata.allows_speculation():
                    continue
                if not any(effect.key in relevant for effect in operator.effects):
                    continue
                if not operator.preconditions_hold(current, step.arguments):
                    continue
                projected = operator.apply_effects(current, step.arguments)
                if projected.fingerprint() == current.fingerprint():
                    continue
                if best is None or operator.cost < best.operator.cost:
                    best = step
            if best is None:
                break
            chosen.append(best)
            current = best.operator.apply_effects(current, best.arguments)
        if chosen:
            emit(logger, "speculative_plan", goal=goal.describe(),
                 steps=[step.operator.name for step in chosen])
        return tuple(chosen)

    def _exhausted(self, goal: SemanticGoal, schema: GoalSchema,
                   grounded: tuple[PlanStep, ...], blockers: dict[str, tuple[str, ...]],
                   bounds: PlanningLimits) -> PlanningFailed:
        """Classify a stalled search as infeasible or merely unfound.

        Reaching the candidate budget is not itself evidence of infeasibility: a large but
        solvable problem hits it too. Infeasibility is when the search cannot progress past
        a precondition that nothing in the capability can produce, so that is what is
        checked. The distinction matters because one is a semantic outcome to report to
        the user and the other is an internal bound.
        """
        producible = {effect.key for step in grounded for effect in step.operator.effects}
        genuine: list[str] = []
        for name, unmet in sorted(blockers.items()):
            for condition in unmet:
                key = condition.split(" ")[0]
                if key not in producible:
                    genuine.append(f"{name} requires {condition}")
        if genuine:
            return GoalInfeasible(goal.describe(), tuple(dict.fromkeys(genuine)))
        blocked = self._blocker_summary(blockers)
        detail = f" - {blocked}" if blocked else ""
        return PlanningFailed(
            f"cannot achieve {goal.describe()}: search exceeded "
            f"{bounds.max_candidates} candidates{detail}")

    @staticmethod
    def _blocker_summary(blockers: dict[str, tuple[str, ...]]) -> str:
        """The preconditions observed to fail during the real search, most specific first.

        Reports what actually blocked the search rather than a search bound. Prefers
        conditions on capability-private keys (a policy limit) over a bare "is false"
        requirement, so an over-limit transfer names the limit.
        """
        if not blockers:
            return ""
        scored: list[tuple[int, str]] = []
        for name, unmet in blockers.items():
            for condition in unmet:
                # "X is false" is the weakest signal: usually an operator's normal
                # precondition rather than the reason nothing worked.
                score = 0 if condition.endswith("is false") else 1 if condition.endswith("is true") else 2
                scored.append((score, f"{name} requires {condition}"))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return "; ".join(dict.fromkeys(text for _, text in scored[:3]))

    @staticmethod
    def _explain(goal: SemanticGoal, schema: GoalSchema, world: WorldState,
                 grounded: tuple[PlanStep, ...], bounds: PlanningLimits,
                 blockers: dict[str, tuple[str, ...]]) -> str:
        """Human-readable reason the goal is unreachable.

        Reports the operator that is actually blocking, following the prerequisite
        chain rather than only naming operators that write a satisfaction key. That is
        what turns "no plan within depth 8" into "amount exceeds the transfer limit".
        """
        wanted = {condition.key for condition in schema.satisfied_when}
        if schema.command:
            seeds = [step for step in grounded if step.operator.satisfies(goal.goal_type)]
        else:
            seeds = [step for step in grounded
                     if any(effect.key in wanted for effect in step.operator.effects)] or list(grounded)

        # Walk producers backwards until we find steps whose preconditions fail now.
        frontier = list(seeds)
        seen: set[str] = set()
        details: list[str] = []
        while frontier:
            step = frontier.pop(0)
            if step.operator.name in seen:
                continue
            seen.add(step.operator.name)
            unmet = step.operator.unsatisfied_preconditions(world, step.arguments)
            if unmet:
                details.append(f"{step.operator.name} requires {', '.join(unmet)}")
                continue
            # This step is available; look at what produces its preconditions.
            for condition in step.operator.preconditions:
                for candidate in grounded:
                    if candidate.operator.name in seen:
                        continue
                    if any(effect.key == condition.key for effect in candidate.operator.effects):
                        frontier.append(candidate)
        if not details:
            # Fall back to anything that was seen blocked during expansion.
            details = [f"{name} requires {', '.join(reasons)}"
                       for name, reasons in sorted(blockers.items()) if reasons]
        if not details:
            return (f"no registered operator can achieve {goal.describe()} "
                    f"(searched depth {bounds.max_depth})")
        return f"cannot achieve {goal.describe()}: " + "; ".join(sorted(set(details)))

    def plan(self, goal: SemanticGoal, schema: GoalSchema, world: WorldState,
             operators: tuple[ActionOperator, ...],
             *, final: bool = True, speculation_only: bool = False,
             exclude: frozenset[str] | set[str] = frozenset(),
             limits: PlanningLimits | None = None) -> Plan:
        bounds = limits or self.limits
        if schema.presupposes:
            unmet = world.unmet(schema.presupposes, goal.arguments)
            if unmet:
                # A presupposition is about the starting state only. Stating it here,
                # once, is what prevents a plan from manufacturing its own precondition.
                raise PlanningFailed(
                    f"cannot achieve {goal.describe()}: it presupposes {', '.join(unmet)}")
        if self.satisfied(goal, schema, world):
            emit(logger, "plan_already_satisfied", goal=goal.describe())
            return Plan(goal=goal, steps=(), total_cost=0.0)

        eligible = tuple(
            operator for operator in operators
            if operator.metadata.allows(final=final)
            and (not speculation_only or operator.metadata.allows_speculation())
            # For a command goal only its own operator may be terminal; other
            # command operators (e.g. Forward while planning Back) are excluded.
            and (not schema.command or not operator.satisfies_goals
                 or operator.satisfies(goal.goal_type))
        )
        grounded = self.ground(eligible, goal)
        if exclude:
            grounded = tuple(step for step in grounded if _step_key(step) not in exclude)
        if not grounded:
            raise PlanningFailed(f"no eligible operator can advance {goal.describe()}")

        command_goals = {goal.goal_type} if schema.command else set()

        frontier: list[_SearchState] = [_SearchState(world=world, steps=(), cost=0.0)]
        visited: dict[tuple[tuple[str, Any], ...], float] = {world.fingerprint(): 0.0}
        expanded = 0
        # Operators that could satisfy the goal but whose preconditions never held,
        # with the reason. Used to explain a failure instead of reporting only a bound.
        blockers: dict[str, tuple[str, ...]] = {}
        for _depth in range(bounds.max_depth):
            next_frontier: list[_SearchState] = []
            for state in frontier:
                for step in grounded:
                    if not step.operator.preconditions_hold(state.world, step.arguments):
                        unmet = step.operator.unsatisfied_preconditions(state.world, step.arguments)
                        blockers.setdefault(step.operator.name, unmet)
                        continue
                    try:
                        projected = step.operator.apply_effects(state.world, step.arguments)
                    except (KeyError, OperatorError) as error:
                        emit(logger, "plan_grounding_error", operator=step.operator.name, error=str(error))
                        continue
                    candidate = _SearchState(world=projected,
                                             steps=(*state.steps, step),
                                             cost=state.cost + step.operator.cost)
                    if command_goals and step.operator.satisfies(goal.goal_type):
                        plan = Plan(goal=goal, steps=candidate.steps, total_cost=candidate.cost)
                        emit(logger, "plan_found", goal=goal.describe(),
                             steps=[s.operator.name for s in plan.steps], total_cost=plan.total_cost,
                             world_changed=candidate.world.diff(world), command=True)
                        return plan
                    expanded += 1
                    if expanded > bounds.max_candidates:
                        raise self._exhausted(goal, schema, grounded, blockers, bounds)
                    if self.satisfied(goal, schema, projected):
                        plan = Plan(goal=goal, steps=candidate.steps, total_cost=candidate.cost)
                        emit(logger, "plan_found", goal=goal.describe(), steps=[s.operator.name for s in plan.steps],
                             total_cost=plan.total_cost, world_changed=candidate.world.diff(world))
                        return plan
                    key = projected.fingerprint()
                    if key in visited and visited[key] <= candidate.cost:
                        continue
                    visited[key] = candidate.cost
                    next_frontier.append(candidate)
            if not next_frontier:
                break
            frontier = sorted(next_frontier, key=lambda item: item.cost)[: bounds.max_candidates]
        raise PlanningFailed(self._explain(goal, schema, world, grounded, bounds, blockers))
