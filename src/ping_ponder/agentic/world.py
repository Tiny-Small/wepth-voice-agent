"""Planner world state: dotted keys, bound conditions, declared effects.

World state describes *execution* facts (is Spotify running, what is playing).
It is consumed by the planner and executor only; it is never given to Jev.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Callable, Mapping


class UnknownWorldKey(KeyError):
    """Raised when a condition or effect names a key outside the declared schema."""


def _unfreeze(value: Any) -> Any:
    """Effects and observed updates may arrive as lists/tuples; store hashable values."""
    if isinstance(value, list):
        return tuple(value)
    return value


@dataclass(frozen=True)
class Arg:
    """A condition/effect value bound from the operator or goal argument map."""

    name: str

    def resolve(self, args: Mapping[str, Any]) -> Any:
        try:
            return args[self.name]
        except KeyError as error:
            raise KeyError(f"argument '{self.name}' is not bound") from error


BoundValue = Any | Arg


class Compare(StrEnum):
    EQ = "eq"
    NE = "ne"
    TRUTHY = "truthy"
    FALSY = "falsy"
    CONTAINS = "contains"
    EXISTS = "exists"


Predicate = Callable[["WorldState"], bool]
# A bound predicate also receives the goal's arguments, so a condition can compare
# world state against what this goal actually asked for (for example, whether the
# prepared transfer is for the requested recipient).
BoundPredicate = Callable[["WorldState", Mapping[str, Any]], bool]


@dataclass(frozen=True)
class Condition:
    """One testable fact about world state; `value` may be literal or an Arg.

    A `predicate` covers comparisons the declarative vocabulary cannot express
    (index bounds, cross-key arithmetic). It replaces the key/compare evaluation
    and is documented on the operator that relies on it.
    """

    key: str
    compare: Compare = Compare.TRUTHY
    value: BoundValue = None
    predicate: Predicate | None = None
    bound_predicate: BoundPredicate | None = None

    def resolved_value(self, args: Mapping[str, Any]) -> Any:
        return self.value.resolve(args) if isinstance(self.value, Arg) else self.value

    def evaluate(self, world: "WorldState", args: Mapping[str, Any] = MappingProxyType({})) -> bool:
        if self.bound_predicate is not None:
            return bool(self.bound_predicate(world, args))
        if self.predicate is not None:
            return bool(self.predicate(world))
        present = self.key in world
        current = world.get(self.key)
        if self.compare is Compare.EXISTS:
            return present
        if self.compare is Compare.TRUTHY:
            return bool(current)
        if self.compare is Compare.FALSY:
            return not bool(current)
        if self.compare is Compare.EQ:
            return present and current == self.resolved_value(args)
        if self.compare is Compare.NE:
            return current != self.resolved_value(args)
        if self.compare is Compare.CONTAINS:
            if not present or current is None:
                return False
            expected = self.resolved_value(args)
            try:
                return expected in current
            except TypeError:
                return False
        raise AssertionError(f"unhandled comparison {self.compare}")

    def describe(self) -> str:
        if self.bound_predicate is not None:
            return f"{self.key} bound predicate"
        if self.predicate is not None:
            return f"{self.key} predicate"
        if self.compare is Compare.TRUTHY:
            return f"{self.key} is true"
        if self.compare is Compare.FALSY:
            return f"{self.key} is false"
        if self.compare is Compare.EXISTS:
            return f"{self.key} exists"
        symbol = {Compare.EQ: "==", Compare.NE: "!=", Compare.CONTAINS: "contains"}[self.compare]
        return f"{self.key} {symbol} {self.value!r}"


@dataclass(frozen=True)
class Effect:
    """A world-state write declared by an operator.

    `value` is a literal or an Arg; `derive` computes from (world, args) and takes
    precedence when supplied (used for counting operators such as Skip).
    """

    key: str
    value: BoundValue = None
    derive: Callable[[Mapping[str, Any], Mapping[str, Any]], Any] | None = None

    def resolve(self, world: "WorldState", args: Mapping[str, Any]) -> Any:
        if self.derive is not None:
            return self.derive(world.values, args)
        return self.value.resolve(args) if isinstance(self.value, Arg) else self.value

    def describe(self) -> str:
        if self.derive is not None:
            return f"{self.key} := f(world)"
        return f"{self.key} := {self.value!r}"


def _hashable(value: Any) -> Any:
    """Reduce a world value to something hashable without losing cycle-detection fidelity."""
    if value is None or isinstance(value, (bool, int, float, str, bytes, tuple, frozenset)):
        return value
    if isinstance(value, Mapping):
        return tuple(sorted((str(key), _hashable(item)) for key, item in value.items()))
    if isinstance(value, (list, set)):
        return tuple(_hashable(item) for item in value)
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        try:
            return ("model", type(value).__name__, _hashable(dump(mode="json")))
        except Exception:
            return ("model", type(value).__name__, repr(value))
    return (type(value).__name__, repr(value))


@dataclass(frozen=True)
class WorldState:
    """Immutable snapshot of dotted-key execution facts."""

    values: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", MappingProxyType(dict(self.values)))

    @classmethod
    def from_schema(cls, schema: Mapping[str, Any]) -> "WorldState":
        return cls(values=dict(schema))

    def get(self, key: str, default: Any = None) -> Any:
        return self.values.get(key, default)

    def require(self, key: str) -> Any:
        if key not in self.values:
            raise UnknownWorldKey(key)
        return self.values[key]

    def __contains__(self, key: object) -> bool:
        return key in self.values

    def updated(self, changes: Mapping[str, Any]) -> "WorldState":
        merged = dict(self.values)
        for key, value in changes.items():
            merged[key] = _unfreeze(value)
        return WorldState(values=merged)

    def conditions_hold(self, conditions: tuple[Condition, ...], args: Mapping[str, Any] = MappingProxyType({})) -> bool:
        return all(condition.evaluate(self, args) for condition in conditions)

    def unmet(self, conditions: tuple[Condition, ...], args: Mapping[str, Any] = MappingProxyType({})) -> tuple[str, ...]:
        return tuple(condition.describe() for condition in conditions if not condition.evaluate(self, args))

    def fingerprint(self) -> tuple[tuple[str, Any], ...]:
        """Hashable identity of this state, for visited-set deduplication.

        Values may be arbitrary objects (nested domain models, controllers), so each
        is reduced to a stable, hashable form. Unhashable values fall back to their
        `repr`, which is sufficient for cycle detection and never raises.
        """
        return tuple(sorted((key, _hashable(value)) for key, value in self.values.items()))

    def diff(self, other: "WorldState") -> dict[str, Any]:
        return {key: value for key, value in other.values.items() if self.values.get(key) != value}

    def as_dict(self) -> dict[str, Any]:
        return dict(self.values)
