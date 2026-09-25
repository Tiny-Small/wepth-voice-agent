"""Global and Local Jev over the existing provider-neutral Decisions transport.

Global Jev answers exactly one question: which registered capability does this
partial utterance belong to? Local Jev answers: within this capability, what
semantic outcome does the user want?

Both are bounded classifiers. They receive the utterance and light session
context (`active_local`) only. Execution-level world state (is Spotify running,
what is focused, current URL) is deliberately never passed in: a capability must
not be selected or reinterpreted because some action happens to be executable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from ping_ponder.observability import emit
from ping_ponder.providers.base import InferenceResponse, StructuredOutputError
from ping_ponder.providers.decisions import ChoiceAnswer, DecisionsProvider, DecisionsResponse

from .registry import CapabilityDescriptor, NONE_CAPABILITY, CapabilityRegistry

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GlobalRoute:
    """Which capability the utterance belongs to, and at what confidence."""

    capability: str | None
    confidence: float | None
    reason: str | None = None
    raw_choice: str | None = None

    @property
    def is_none(self) -> bool:
        return self.capability is None

    @property
    def switched_from(self) -> bool:
        return False  # populated by the spine, kept for symmetry with LocalRoute


@dataclass(frozen=True)
class LocalRoute:
    """The semantic goal type within a capability; never an executable action."""

    capability: str
    goal_type: str | None
    confidence: float | None
    raw_choice: str | None = None
    latency_seconds: float = 0.0

    @property
    def is_none(self) -> bool:
        return self.goal_type is None

    def describe(self) -> str:
        return f"{self.capability}.{self.goal_type or 'None'}"


class LocalJev(Protocol):
    """Within one capability, classify the semantic outcome of an utterance."""

    async def interpret(self, utterance: str, *, descriptor: CapabilityDescriptor) -> LocalRoute: ...


class GlobalJev(Protocol):
    """Route an utterance to a registered capability."""

    async def route(self, utterance: str, *, active_local: str | None = None,
                    context: Mapping[str, Any] | None = None) -> GlobalRoute: ...


def _choice(instructions: str, criteria: Mapping[str, str]) -> dict[str, Any]:
    return {"type": "choice", "instructions": instructions, "criteria": dict(criteria)}


def _answers_match(response: InferenceResponse[DecisionsResponse], questions: Mapping[str, Any]) -> bool:
    return response.value.answers.keys() == questions.keys()


class JevGlobalRouter:
    """Global Jev: one dynamic choice question over currently registered capabilities.

    The choice criteria come from `CapabilityRegistry.criteria()`, so a newly
    registered capability appears in Global Jev without touching this class.
    """

    def __init__(self, provider: DecisionsProvider, registry: CapabilityRegistry, *, model: str) -> None:
        if not model:
            raise ValueError("a Global Jev model is required")
        self.provider = provider
        self.registry = registry
        self.model = model
        # Surfaced in the chat trace so which Jev answered is never a guess.
        self.name = f"global:{model}"
        self.last_inference: InferenceResponse[DecisionsResponse] | None = None

    def questions(self) -> dict[str, Any]:
        allowed = "|".join(self.registry.choices())
        return {
            "capability": _choice(
                "Which single registered capability does this utterance belong to? "
                f"Answer exactly one of: {allowed}. "
                "Choose None when no registered capability applies. "
                "Classify the request only; do not solve it and do not choose an action. "
                "Use active local context as a strong prior for short follow-ups; switch "
                "only when the utterance clearly belongs elsewhere.",
                self.registry.criteria(),
            )
        }

    async def route(self, utterance: str, *, active_local: str | None = None,
                    context: Mapping[str, Any] | None = None) -> GlobalRoute:
        questions = self.questions()
        # Only semantic session context enters Global Jev. No world state, ever.
        state: dict[str, Any] = {
            "utterance": utterance,
            "active_local": active_local or NONE_CAPABILITY,
            "registered_capabilities": list(self.registry.names()),
        }
        if context:
            state.update({key: value for key, value in context.items() if key in {"conversation_summary", "recent_utterances"}})
        response = await self.provider.decide(model=self.model, state=state, questions=questions)
        self.last_inference = response
        if not _answers_match(response, questions):
            emit(logger, "structured_output_validation_failed", agent="global_jev", error_type="answer_keys_mismatch")
            raise StructuredOutputError("Global Jev answer keys do not match requested questions")
        answer = response.value.answers["capability"]
        if not isinstance(answer, ChoiceAnswer):
            raise StructuredOutputError("Global Jev capability answer has the wrong type")
        choice = answer.choice
        if choice not in self.registry.choices():
            emit(logger, "structured_output_validation_failed", agent="global_jev",
                 error_type="unknown_capability", choice=choice)
            raise StructuredOutputError(f"Global Jev returned an unknown capability '{choice}'")
        capability = None if choice == NONE_CAPABILITY else choice
        emit(logger, "global_jev_routing", model=response.model, resolved_model=response.resolved_model,
             wall_latency_ms=response.latency_seconds * 1000, capability=capability,
             active_local=active_local, confidence=answer.confidence)
        return GlobalRoute(capability=capability, confidence=answer.confidence, raw_choice=choice)


class JevLocalJev:
    """Local Jev: one dynamic choice question over the capability's goal schemas.

    The answer is a semantic goal type such as PLAY or PAUSE - never an operator
    such as OpenSpotify. Goal schemas and their descriptions come from the
    capability descriptor.
    """

    def __init__(self, provider: DecisionsProvider, *, model: str) -> None:
        if not model:
            raise ValueError("a Local Jev model is required")
        self.provider = provider
        self.model = model
        self.name = f"local:{model}"
        self.last_inference: InferenceResponse[DecisionsResponse] | None = None

    def questions(self, descriptor: CapabilityDescriptor) -> dict[str, Any]:
        criteria = {goal_type: schema_description(goal_type, schema)
                    for goal_type, schema in descriptor.goal_schemas.items()}
        criteria[NONE_CAPABILITY] = f"The utterance does not express any {descriptor.name} goal."
        allowed = "|".join([*descriptor.goal_schemas, NONE_CAPABILITY])
        return {
            "goal": _choice(
                f"Within {descriptor.name}, which semantic outcome does the user want? "
                f"Answer exactly one of: {allowed}. "
                "Report the user's intent, not an action to take. "
                "Do not answer with an action name such as Open, Search, or Play unless it is a declared goal. "
                f"If {descriptor.name} is closed, still report the intended goal.",
                criteria,
            )
        }

    async def interpret(self, utterance: str, *, descriptor: CapabilityDescriptor) -> LocalRoute:
        questions = self.questions(descriptor)
        state = {
            "utterance": utterance,
            "capability": descriptor.name,
            "goals": list(descriptor.goal_schemas),
        }
        response = await self.provider.decide(model=self.model, state=state, questions=questions)
        self.last_inference = response
        if not _answers_match(response, questions):
            emit(logger, "structured_output_validation_failed", agent="local_jev",
                 capability=descriptor.name, error_type="answer_keys_mismatch")
            raise StructuredOutputError("Local Jev answer keys do not match requested questions")
        answer = response.value.answers["goal"]
        if not isinstance(answer, ChoiceAnswer):
            raise StructuredOutputError("Local Jev goal answer has the wrong type")
        choice = answer.choice
        if choice not in descriptor.goal_schemas and choice != NONE_CAPABILITY:
            emit(logger, "structured_output_validation_failed", agent="local_jev",
                 capability=descriptor.name, error_type="unknown_goal", choice=choice)
            raise StructuredOutputError(f"Local Jev returned an unknown goal '{choice}'")
        goal_type = None if choice == NONE_CAPABILITY else choice
        emit(logger, "local_jev_interpretation", model=response.model, capability=descriptor.name,
             wall_latency_ms=response.latency_seconds * 1000, goal_type=goal_type,
             confidence=answer.confidence)
        return LocalRoute(capability=descriptor.name, goal_type=goal_type, confidence=answer.confidence,
                          raw_choice=choice, latency_seconds=response.latency_seconds)


def schema_description(goal_type: str, schema: Any) -> str:
    """Render a goal schema as classifier criteria, including its slots."""
    slots = ", ".join(f"{slot.name} (open-ended: {slot.question})" for slot in schema.slots)
    suffix = f" Arguments: {slots}." if slots else " No open-ended arguments."
    intent = f" Meaning: {schema.intent_description}" if schema.intent_description else ""
    return f"User wants {goal_type}.{intent}.{suffix}"
