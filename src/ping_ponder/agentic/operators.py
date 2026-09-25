"""Action operators: parameters, preconditions, effects, cost, executor, policy."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

from .world import Condition, Effect, WorldState

Args = Mapping[str, Any]


class OperatorError(RuntimeError):
    """Executor failure for one operator; the spinere observes and replans."""


@dataclass(frozen=True)
class OperatorMetadata:
    """Execution policy for speculation and authority.

    `speculative_safe` gates execution from a partial transcript.
    `requires_final` refuses execution until the transcript is authoritative; it
    defaults to the complement of `speculative_safe`.
    `reversible`/`idempotent` describe replay behaviour under replanning.
    """

    speculative_safe: bool = False
    reversible: bool = False
    idempotent: bool = False
    requires_final: bool | None = None
    # Set when the operation returns its own authoritative post-action observation.
    # The executor must not merge a different capability observer over that result.
    self_observing: bool = False
    # A bounded delegated operation has reached a terminal result; don't invoke it
    # again just because its result did not satisfy the semantic goal.
    terminal_on_execution: bool = False

    def __post_init__(self) -> None:
        if self.requires_final is None:
            object.__setattr__(self, "requires_final", not self.speculative_safe)
        if self.speculative_safe and self.requires_final:
            raise ValueError("an operator cannot be speculative-safe and require a final transcript")

    def allows(self, *, final: bool) -> bool:
        return final or not self.requires_final

    def allows_speculation(self) -> bool:
        return self.speculative_safe and not self.requires_final


class OperatorExecutor(Protocol):
    """Applies one operator to observed world state and returns the observed change."""

    async def __call__(self, world: WorldState, arguments: Args) -> Args: ...


@dataclass(frozen=True)
class ActionOperator:
    name: str
    parameters: tuple[str, ...] = ()
    preconditions: tuple[Condition, ...] = ()
    effects: tuple[Effect, ...] = ()
    cost: float = 1.0
    executor: OperatorExecutor | None = None
    metadata: OperatorMetadata = field(default_factory=OperatorMetadata)
    satisfies_goals: tuple[str, ...] = ()
    # Bound from the goal when present. `optional_parameters` covers slots the goal
    # schema marks as not required; an unbound optional parameter is simply absent from
    # the step arguments rather than making the operator ungroundable. Declared after
    # the positional block so existing positional construction keeps its meaning.
    optional_parameters: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("operator name is required")
        if len(self.parameters) != len(set(self.parameters)):
            raise ValueError(f"duplicate parameters in operator '{self.name}'")
        if set(self.parameters) & set(self.optional_parameters):
            raise ValueError(f"operator '{self.name}' lists a parameter as both required and optional")
        if self.cost < 0:
            raise ValueError("operator cost must be nonnegative")
        undeclared = {effect.key.split(".", 1)[0] for effect in self.effects if "." in effect.key}
        precondition_namespaces = {condition.key.split(".", 1)[0] for condition in self.preconditions if "." in condition.key}
        if not undeclared and precondition_namespaces:
            raise ValueError(f"operator '{self.name}' declares no effects")
        if not self.effects:
            raise ValueError(f"operator '{self.name}' must declare at least one effect")

    def bind(self, arguments: Args) -> Args:
        missing = [name for name in self.parameters if name not in arguments]
        if missing:
            raise OperatorError(f"operator '{self.name}' is missing arguments: {sorted(missing)}")
        bound = {name: arguments[name] for name in self.parameters}
        bound.update({name: arguments[name] for name in self.optional_parameters
                      if arguments.get(name) not in (None, "")})
        return bound

    def preconditions_hold(self, world: WorldState, arguments: Args) -> bool:
        return world.conditions_hold(self.preconditions, arguments)

    def unsatisfied_preconditions(self, world: WorldState, arguments: Args) -> tuple[str, ...]:
        return world.unmet(self.preconditions, arguments)

    def satisfies(self, goal_type: str) -> bool:
        """Whether running this operator directly achieves the given command goal."""
        return goal_type in self.satisfies_goals

    def apply_effects(self, world: WorldState, arguments: Args) -> WorldState:
        return world.updated({effect.key: effect.resolve(world, arguments) for effect in self.effects})
