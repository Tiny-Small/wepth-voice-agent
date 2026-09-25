"""Open-ended capability registry.

The registry is the single source of truth for Global Jev's choices, each
capability's Local Jev, its semantic goal schemas, its planner operators, and its
open-ended argument slots. Adding a capability is a registration, never an edit
to Global Jev.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Iterator, Mapping, Protocol

from .goals import GoalSchema
from .operators import ActionOperator

if TYPE_CHECKING:  # pragma: no cover
    from .planner import PlanningLimits

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids a runtime import cycle
    from .jev import LocalJev

NONE_CAPABILITY = "None"


class UnknownCapability(LookupError):
    """Raised when a routing decision names an unregistered capability."""


class CapabilityObserver(Protocol):
    """Reads current capability facts without predicting operator effects."""

    def __call__(self) -> Mapping[str, Any] | Awaitable[Mapping[str, Any]]: ...


@dataclass(frozen=True)
class CapabilityDescriptor:
    """Everything Global Jev, the goal builder, and the planner need for one capability."""

    name: str
    description: str
    local_jev: "LocalJev"
    goal_schemas: Mapping[str, GoalSchema]
    operators: tuple[ActionOperator, ...]
    # Keys this capability reads or writes, with initial values. Used to assemble
    # the planner's world state; never passed to Jev.
    world_schema: Mapping[str, Any] = field(default_factory=dict)
    # Optional: names the action a user would currently be confirming, derived from
    # world state. The spine uses it to drop a pending confirmation when that action
    # materially changes, so a stale "yes" cannot authorize a different action. Kept
    # here rather than called from a projection, which must stay pure for planning.
    confirmation_subject: Callable[[Any], str | None] | None = None
    # Optional authoritative read of application state before planning and after
    # each action. It is deliberately outside Jev inputs.
    observer: CapabilityObserver | None = None

    def __post_init__(self) -> None:
        if not self.name or not self.description.strip():
            raise ValueError("capability requires a name and description")
        if not self.goal_schemas:
            raise ValueError(f"capability '{self.name}' must declare at least one goal schema")
        for goal_type, schema in self.goal_schemas.items():
            if goal_type != schema.goal_type:
                raise ValueError(f"capability '{self.name}' schema key '{goal_type}' does not match its goal type")
        names = [operator.name for operator in self.operators]
        if len(names) != len(set(names)):
            raise ValueError(f"capability '{self.name}' has duplicate operator names")

    def schema(self, goal_type: str) -> GoalSchema:
        try:
            return self.goal_schemas[goal_type]
        except KeyError as error:
            raise KeyError(f"capability '{self.name}' has no goal schema '{goal_type}'") from error

    def operator(self, name: str) -> ActionOperator:
        for operator in self.operators:
            if operator.name == name:
                return operator
        raise KeyError(f"capability '{self.name}' has no operator '{name}'")

    def operator_names(self) -> tuple[str, ...]:
        return tuple(operator.name for operator in self.operators)

    def world_keys(self) -> tuple[str, ...]:
        """Every world key this capability declares, for assembling planner state."""
        return tuple(sorted(self.world_schema))

    def planning_limits(self) -> "PlanningLimits":
        """Search bounds sized to this capability's longest operator chain."""
        from .planner import PlanningLimits

        return PlanningLimits.for_chain(len(self.operators))


class CapabilityRegistry:
    """Mutable registry with dynamic Global Jev choices."""

    def __init__(self, descriptors: Mapping[str, CapabilityDescriptor] | None = None) -> None:
        self._descriptors: dict[str, CapabilityDescriptor] = {}
        for name, descriptor in (descriptors or {}).items():
            self.register(descriptor, name=name)

    def register(self, descriptor: CapabilityDescriptor, *, name: str | None = None) -> CapabilityDescriptor:
        key = name or descriptor.name
        if not key or key == NONE_CAPABILITY:
            raise ValueError(f"capability name '{key}' is reserved")
        if key != descriptor.name:
            raise ValueError("registry key must match the descriptor name")
        self._descriptors[key] = descriptor
        return descriptor

    def get(self, name: str) -> CapabilityDescriptor:
        try:
            return self._descriptors[name]
        except KeyError as error:
            raise UnknownCapability(name) from error

    def optional(self, name: str | None) -> CapabilityDescriptor | None:
        if name is None or name == NONE_CAPABILITY:
            return None
        return self._descriptors.get(name)

    def names(self) -> tuple[str, ...]:
        return tuple(self._descriptors)

    def choices(self) -> tuple[str, ...]:
        """Global Jev's dynamic choice list: registered capabilities plus 'None'."""
        return (*self._descriptors, NONE_CAPABILITY)

    def criteria(self) -> dict[str, str]:
        """Choice criteria derived from registered descriptions, not hard-coded."""
        criteria = {NONE_CAPABILITY: "Conversation, small talk, or an utterance that belongs to no registered capability."}
        for name, descriptor in self._descriptors.items():
            criteria[name] = descriptor.description
        return criteria

    def descriptions(self) -> str:
        return "\n".join(f"- {name}: {descriptor.description}" for name, descriptor in self._descriptors.items())

    def __contains__(self, name: object) -> bool:
        return name in self._descriptors

    def __iter__(self) -> Iterator[CapabilityDescriptor]:
        return iter(self._descriptors.values())

    def __len__(self) -> int:
        return len(self._descriptors)
