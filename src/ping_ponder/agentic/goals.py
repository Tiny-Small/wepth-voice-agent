"""Semantic goals, goal schemas, and revocable partial-speech hypotheses.

A `SemanticGoal` is authoritative: the utterance is complete and the user's
intent is settled. An `IntentHypothesis` is a provisional reading of a partial
transcript. It is revocable, may never be executed as-is, and may drive only
operators explicitly marked speculative-safe.

Neither type carries execution state. Goal satisfaction is expressed as world
conditions on the goal schema, and evaluated by the planner/executor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Callable, Mapping

from .world import Condition


class SlotKind(StrEnum):
    """How a slot's extracted span becomes an argument value.

    SPAN keeps the transcript text as-is. NUMBER additionally coerces the *already
    extracted span* into a typed value. Coercion never searches the raw transcript:
    extraction stays with the extractor, and this only types what it returned.
    """

    SPAN = "span"
    NUMBER = "number"


class UnknownGoalType(ValueError):
    """Raised when a Local Jev returns a goal type the capability does not declare."""


@dataclass(frozen=True)
class SlotSpec:
    """One open-ended argument and the extractive question that fills it."""

    name: str
    question: str
    required: bool = True
    confidence_threshold: float = 0.30
    kind: SlotKind = SlotKind.SPAN
    # Optional predicate over the extracted span text. A capability uses it to reject
    # an implausible answer (a currency slot should not accept "Sarah"), including for
    # a weak extractor whose answer would otherwise be accepted merely because it is a
    # real transcript span. Validation happens before the value becomes a goal argument.
    validator: Callable[[str], bool] | None = None

    def __post_init__(self) -> None:
        if not self.name or not self.question:
            raise ValueError("slot name and extraction question are required")
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("slot confidence threshold must be within [0, 1]")

    def accepts(self, text: str) -> bool:
        return self.validator is None or bool(self.validator(text))


@dataclass(frozen=True)
class GoalSchema:
    """A goal type a Local Jev may return, plus its slots and satisfaction test.

    Either `satisfied_when` describes the world state that satisfies the goal, or
    `command=True` marks a one-shot command goal (Back, Forward) whose satisfaction
    is that an operator declaring `satisfies_goals` for this type has executed.
    Exactly one of the two must be provided.
    """

    goal_type: str
    slots: tuple[SlotSpec, ...] = ()
    satisfied_when: tuple[Condition, ...] = ()
    # Conditions that must hold in the state the plan starts from. This encodes a
    # *presupposition* ("skipping presupposes playback") rather than a goal or an
    # effect, and it cannot be satisfied by planning: an operator that would establish
    # it is not allowed to justify its own precondition.
    presupposes: tuple[Condition, ...] = ()
    command: bool = False
    # For a command goal, the world state that means "the command has already taken
    # effect". Without this an idempotent command such as Confirm would be replanned
    # forever, since a command goal is never satisfied by the declarative test.
    already_done_when: tuple[Condition, ...] = ()
    # Concise semantic meaning supplied to Local Jev for closely related goal types.
    intent_description: str | None = None

    def __post_init__(self) -> None:
        names = [slot.name for slot in self.slots]
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate slot names in goal '{self.goal_type}'")
        if bool(self.satisfied_when) == bool(self.command):
            raise ValueError(
                f"goal '{self.goal_type}' must declare either satisfied_when conditions or command=True, not both")
        if self.already_done_when and not self.command:
            raise ValueError(f"goal '{self.goal_type}' declares already_done_when but is not a command goal")

    def slot(self, name: str) -> SlotSpec:
        for spec in self.slots:
            if spec.name == name:
                return spec
        raise KeyError(f"goal '{self.goal_type}' has no slot '{name}'")

    @property
    def required_slots(self) -> tuple[SlotSpec, ...]:
        return tuple(slot for slot in self.slots if slot.required)

    def missing_slots(self, arguments: Mapping[str, Any]) -> tuple[str, ...]:
        return tuple(slot.name for slot in self.required_slots
                     if arguments.get(slot.name) in (None, "", ()))


@dataclass(frozen=True)
class SemanticGoal:
    """Authoritative, capability-scoped intent. Contains no execution state."""

    capability: str
    goal_type: str
    arguments: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.capability or not self.goal_type:
            raise ValueError("semantic goal requires a capability and goal type")
        object.__setattr__(self, "arguments", MappingProxyType(dict(self.arguments)))

    def argument(self, name: str, default: Any = None) -> Any:
        return self.arguments.get(name, default)

    def with_arguments(self, **updates: Any) -> "SemanticGoal":
        merged = dict(self.arguments)
        merged.update(updates)
        return SemanticGoal(capability=self.capability, goal_type=self.goal_type, arguments=merged)

    def describe(self) -> str:
        rendered = ", ".join(f"{key}={value!r}" for key, value in sorted(self.arguments.items()))
        return f"{self.capability}.{self.goal_type}({rendered})"


@dataclass(frozen=True)
class IntentHypothesis:
    """Provisional, revocable reading of a partial transcript.

    `is_final=False` always. A hypothesis can start speculative-safe operators but
    must not mutate authoritative semantic state or run a `requires_final`
    operator. `confidence` and `capability_confidence` are reported, not acted on
    beyond the speculation policy.
    """

    capability: str | None
    goal_type: str | None
    arguments: Mapping[str, Any] = field(default_factory=dict)
    transcript: str = ""
    is_final: bool = False
    observed_at_ms: float = 0.0
    capability_confidence: float | None = None
    goal_confidence: float | None = None
    accepted: bool = False

    def __post_init__(self) -> None:
        if self.is_final:
            raise ValueError("an IntentHypothesis is always provisional; use SemanticGoal when final")
        object.__setattr__(self, "arguments", MappingProxyType(dict(self.arguments)))

    def accepted_by(self, capability: str | None) -> bool:
        return capability is not None and self.capability == capability

    def describe(self) -> str:
        return f"hypothesis({self.capability or 'None'}.{self.goal_type or 'None'}, {self.transcript!r})"
