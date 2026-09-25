from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from ping_ponder.agentic.goal_builder import GoalBuilder
from ping_ponder.agentic.goals import GoalSchema, SlotSpec
from ping_ponder.agentic.span import LlamaStructuredSlotExtractor, SlotRequest
from ping_ponder.providers.base import InferenceResponse


class StructuredReply:
    def __init__(self, values):
        self.values = values
        self.models = []
        self.messages = []

    async def infer(self, *, model, messages, response_model):
        self.models.append(response_model)
        self.messages.append(messages)
        value = response_model.model_validate(self.values)
        return InferenceResponse(value=value, provider="fake", model=model,
                                 latency_seconds=0.01, retries=0, usage=None)


@pytest.mark.asyncio
async def test_llama_preserves_declared_slots_and_target_qualifier_through_goal_builder():
    utterance = "Find the section of the MDN grid guide explaining template areas."
    provider = StructuredReply({
        "site": "MDN",
        "target": "the MDN grid guide explaining template areas",
    })
    extractor = LlamaStructuredSlotExtractor(provider, model="llama-test")
    schema = GoalSchema("FIND", slots=(
        SlotSpec("site", "Which website?", required=False),
        SlotSpec("target", "What should be found?"),
    ), satisfied_when=("placeholder",))

    built = await GoalBuilder(extractor).build(capability="Browser", schema=schema,
                                               utterance=utterance)

    assert built.complete
    assert built.goal.goal_type == "FIND"
    assert dict(built.goal.arguments) == {
        "site": "MDN", "target": "the MDN grid guide explaining template areas"
    }
    assert "template areas" in built.goal.argument("target")
    assert set(provider.models[0].model_fields) == {"site", "target"}
    request = json.loads(provider.messages[0][1]["content"])
    assert request["goal_type"] == "FIND"
    assert request["declared_slots"] == [
        {"name": "site", "question": "Which website?", "required": False, "kind": "span"},
        {"name": "target", "question": "What should be found?", "required": True, "kind": "span"},
    ]
    assert "omit words between relevant terms" in provider.messages[0][0]["content"]


@pytest.mark.asyncio
async def test_llama_does_not_invent_optional_site_when_absent():
    utterance = "Find the section of the grid guide explaining template areas."
    provider = StructuredReply({"site": None, "target": "the section of the grid guide explaining template areas"})
    extractor = LlamaStructuredSlotExtractor(provider, model="llama-test")

    spans = await extractor.extract_many(utterance, (
        SlotRequest("site", "Which website?"),
        SlotRequest("target", "What should be found?"),
    ))

    assert "site" not in spans
    assert spans["target"].text.endswith("template areas")


@pytest.mark.asyncio
async def test_llama_cannot_add_undeclared_slots_or_change_goal_type():
    provider = StructuredReply({"target": "Browser Use repository", "site": "GitHub"})
    extractor = LlamaStructuredSlotExtractor(provider, model="llama-test")
    schema = GoalSchema("FIND", slots=(SlotSpec("target", "What should be found?"),),
                        satisfied_when=("placeholder",))

    built = await GoalBuilder(extractor).build(
        capability="Browser", schema=schema,
        utterance="Find the Browser Use repository on GitHub.")

    assert not built.complete
    assert built.goal.goal_type == "FIND"
    assert set(provider.models[0].model_fields) == {"target"}
    assert "site" not in built.goal.arguments


@pytest.mark.asyncio
async def test_goal_builder_rejects_llama_value_that_fails_slot_validation():
    provider = StructuredReply({"target": "the CSS Grid guide"})
    extractor = LlamaStructuredSlotExtractor(provider, model="llama-test")
    schema = GoalSchema("FIND", slots=(SlotSpec(
        "target", "What should be found?",
        validator=lambda value: "template areas" in value,
    ),), satisfied_when=("placeholder",))

    built = await GoalBuilder(extractor).build(
        capability="Browser", schema=schema,
        utterance="Find the CSS Grid guide.")

    assert not built.complete
    assert built.missing_slots == ("target",)
    assert built.extraction("target").reason == "rejected_by_slot"


@pytest.mark.asyncio
async def test_malformed_llama_output_fails_safely_through_goal_builder():
    provider = StructuredReply({"target": "not in utterance", "unexpected": "extra"})
    extractor = LlamaStructuredSlotExtractor(provider, model="llama-test")
    schema = GoalSchema("FIND", slots=(SlotSpec("target", "What should be found?"),),
                        satisfied_when=("placeholder",))

    built = await GoalBuilder(extractor).build(
        capability="Browser", schema=schema, utterance="Find a destination")

    assert not built.complete
    assert built.goal.goal_type == "FIND"
    assert built.extraction("target").filled is False
    assert built.extraction("target").reason == "error:ValidationError"
    assert "unexpected" not in provider.models[0].model_fields
    fallback_request = json.loads(provider.messages[1][1]["content"])
    assert fallback_request["goal_type"] == "FIND"
    assert fallback_request["declared_slots"][0]["required"] is True
