"""Capability registry, dynamic Global Jev choices, and Local Jev goal typing."""

import pytest

from ping_ponder.agentic.capabilities import build_browser_descriptor, build_spotify_descriptor
from ping_ponder.agentic.goals import GoalSchema, SlotSpec
from ping_ponder.agentic.jev import JevGlobalRouter, JevLocalJev, LocalRoute
from ping_ponder.agentic.operators import ActionOperator, OperatorMetadata
from ping_ponder.agentic.registry import CapabilityDescriptor, CapabilityRegistry, UnknownCapability
from ping_ponder.agentic.world import Condition, Effect
from ping_ponder.agentic.wiring import build_default_spine
from ping_ponder.providers.base import InferenceResponse, TokenUsage
from ping_ponder.providers.decisions import DecisionsResponse
from ping_ponder.agentic.span import ExtractedSpan


def no_local_jev():
    return build_spotify_descriptor(None).local_jev


def extra_descriptor(name: str = "VSCode") -> CapabilityDescriptor:
    return CapabilityDescriptor(
        name=name,
        description="Editor commands.",
        local_jev=no_local_jev(),
        goal_schemas={"OPEN_FILE": GoalSchema("OPEN_FILE", slots=(SlotSpec("path", "Which file?"),),
                                              satisfied_when=(Condition("editor.open_file"),))},
        operators=(ActionOperator("OpenFile", ("path",), (), (Effect("editor.open_file"),)),),
    )


class FakeDecisionsProvider:
    def __init__(self, choice: str, confidence: float = 0.9):
        self.choice = choice
        self.confidence = confidence
        self.calls = []

    async def decide(self, *, model, state, questions):
        self.calls.append({"model": model, "state": state, "questions": questions})
        key = next(iter(questions))
        return InferenceResponse(value=DecisionsResponse.model_validate(
            {"answers": {key: {"type": "choice", "choice": self.choice, "confidence": self.confidence}}}),
            provider="fake", model=model, latency_seconds=0.01, retries=0, usage=TokenUsage(total_tokens=5))


def test_registry_choices_are_generated_from_registrations():
    registry = CapabilityRegistry({
        "Spotify": build_spotify_descriptor(None),
        "Browser": build_browser_descriptor(None),
    })
    assert registry.choices() == ("Spotify", "Browser", "None")
    registry.register(extra_descriptor())
    assert registry.choices() == ("Spotify", "Browser", "VSCode", "None")
    # Choice order follows registration order and always ends with the None option.
    assert registry.choices()[-1] == "None"
    assert "Editor commands." in registry.criteria()["VSCode"]


def test_registering_capability_does_not_touch_global_jev():
    registry = CapabilityRegistry({"Spotify": build_spotify_descriptor(None)})
    router = JevGlobalRouter(FakeDecisionsProvider("Spotify"), registry, model="m")
    assert sorted(router.questions()["capability"]["criteria"]) == ["None", "Spotify"]
    registry.register(extra_descriptor())
    assert sorted(router.questions()["capability"]["criteria"]) == ["None", "Spotify", "VSCode"]
    assert "VSCode" in router.questions()["capability"]["instructions"]


def test_registry_rejects_reserved_and_unknown_names():
    registry = CapabilityRegistry()
    with pytest.raises(ValueError):
        registry.register(extra_descriptor("None"))
    with pytest.raises(UnknownCapability):
        registry.get("Nope")


@pytest.mark.asyncio
async def test_global_jev_receives_active_local_but_never_world_state():
    registry = CapabilityRegistry({"Spotify": build_spotify_descriptor(None),
                                   "Browser": build_browser_descriptor(None)})
    provider = FakeDecisionsProvider("Spotify")
    router = JevGlobalRouter(provider, registry, model="router")
    route = await router.route("play some jazz", active_local="Browser")
    assert route.capability == "Spotify" and route.confidence == 0.9
    state = provider.calls[0]["state"]
    assert state["active_local"] == "Browser"
    assert state["registered_capabilities"] == ["Spotify", "Browser"]
    # No execution-level world state may ever reach Global Jev.
    assert not any(key.startswith(("spotify.", "browser.", "media.", "filesystem.")) for key in state)
    assert "spotify.running" not in state and "browser.open" not in state


@pytest.mark.asyncio
async def test_global_jev_rejects_unknown_capability_choice():
    from ping_ponder.providers.base import StructuredOutputError

    registry = CapabilityRegistry({"Spotify": build_spotify_descriptor(None)})
    router = JevGlobalRouter(FakeDecisionsProvider("Terminal"), registry, model="m")
    with pytest.raises(StructuredOutputError):
        await router.route("do a thing")


@pytest.mark.asyncio
async def test_rule_router_routes_plain_search_to_browser_but_spotify_search_to_spotify():
    from ping_ponder.agentic.wiring import RuleBasedGlobalJev
    router = RuleBasedGlobalJev()
    assert (await router.route("Search for Jev documentation")).capability == "Browser"
    assert (await router.route("Search Spotify for Jazz")).capability == "Spotify"


@pytest.mark.asyncio
@pytest.mark.parametrize("choice,expected", [("PLAY", "PLAY"), ("PAUSE", "PAUSE"), ("None", None)])
async def test_local_jev_returns_semantic_goal_not_an_action(choice, expected):
    descriptor = build_spotify_descriptor(None)
    provider = FakeDecisionsProvider(choice)
    local = JevLocalJev(provider, model="local")
    route = await local.interpret("play some jazz", descriptor=descriptor)
    assert route.goal_type == expected
    criteria = provider.calls[0]["questions"]["goal"]["criteria"]
    # Every declared goal type is offered; no operator name is a choice.
    assert set(criteria) == {*descriptor.goal_schemas, "None"}
    assert not ({"OpenSpotify", "SearchSpotify", "PlayResult"} & set(criteria))
    assert "Open" not in " ".join(criteria)


@pytest.mark.asyncio
async def test_browser_local_criteria_distinguish_finding_from_opening_an_address():
    descriptor = build_browser_descriptor(None)
    provider = FakeDecisionsProvider("FIND")
    local = JevLocalJev(provider, model="local")

    await local.interpret("Find the Browser Use repository on GitHub", descriptor=descriptor)

    criteria = provider.calls[0]["questions"]["goal"]["criteria"]
    assert "human-readable target" in criteria["FIND"]
    assert "grounded page interactions" in criteria["FIND"]
    assert "URL, domain, or trusted site alias" in criteria["NAVIGATE"]


@pytest.mark.asyncio
async def test_explicit_find_repository_is_not_downgraded_to_site_search():
    descriptor = build_browser_descriptor(None)
    provider = FakeDecisionsProvider("SEARCH_WEBSITE", confidence=0.66)
    local = JevLocalJev(provider, model="local")

    route = await local.interpret("Find browser use repository on github", descriptor=descriptor)

    assert route.goal_type == "FIND"
    assert route.raw_choice == "SEARCH_WEBSITE"


@pytest.mark.asyncio
async def test_explicit_site_search_keeps_search_website_goal():
    descriptor = build_browser_descriptor(None)
    provider = FakeDecisionsProvider("SEARCH_WEBSITE")
    local = JevLocalJev(provider, model="local")

    route = await local.interpret("Search GitHub for browser-use", descriptor=descriptor)

    assert route.goal_type == "SEARCH_WEBSITE"


@pytest.mark.asyncio
async def test_find_search_results_keeps_search_website_goal():
    descriptor = build_browser_descriptor(None)
    local = JevLocalJev(FakeDecisionsProvider("SEARCH_WEBSITE"), model="local")

    route = await local.interpret("Find search results for browser-use on GitHub", descriptor=descriptor)

    assert route.goal_type == "SEARCH_WEBSITE"


@pytest.mark.asyncio
async def test_navigate_rejects_a_human_readable_item_as_a_destination():
    class NavigateLocal:
        async def interpret(self, utterance, *, descriptor):
            return LocalRoute("Browser", "NAVIGATE", 0.99)

    descriptor = build_browser_descriptor(NavigateLocal())

    class ItemExtractor:
        name = "item-extractor"

        async def extract(self, utterance, question, *, slot, confidence_threshold=0.0):
            text = "Browser Use repository"
            start = utterance.index(text)
            return ExtractedSpan(slot, question, text, start, start + len(text), 0.99, self.name)

    utterance = "Go to Browser Use repository"
    service = build_default_spine(
        extractor=ItemExtractor(), registry=CapabilityRegistry({"Browser": descriptor}),
    )
    outcome = await service.resolve_final(utterance)

    assert outcome.goal.goal_type == "NAVIGATE"
    assert not outcome.build.complete
    assert outcome.build.missing_slots == ("target",)
    assert outcome.build.rejected["target"] == "rejected_by_slot"
    assert outcome.report is None


def test_goal_schemas_declare_slots_and_satisfaction():
    spotify = build_spotify_descriptor(None)
    play = spotify.schema("PLAY")
    assert [slot.name for slot in play.slots] == ["query"]
    # Questions are phrased as a question about the user's intent, not a bare
    # imperative: SQuAD-trained readers answer this form far more reliably.
    question = play.slot("query").question
    assert question.endswith("?") and "user" in question
    assert not question.startswith(("Play", "Search", "Open"))
    assert play.missing_slots({}) == ("query",)
    assert play.missing_slots({"query": "Jazz"}) == ()
    with pytest.raises(KeyError):
        spotify.schema("NOPE")


def test_operator_metadata_encodes_speculation_policy():
    spotify = build_spotify_descriptor(None)
    assert spotify.operator("OpenSpotify").metadata.allows_speculation()
    play = spotify.operator("PlayResult").metadata
    assert play.requires_final and not play.allows_speculation()
    assert spotify.operator("OpenSpotify").metadata.idempotent
    with pytest.raises(ValueError):
        OperatorMetadata(speculative_safe=True, requires_final=True)


def test_descriptor_rejects_inconsistent_declarations():
    with pytest.raises(ValueError):
        CapabilityDescriptor("X", "d", no_local_jev(), {}, ())
    with pytest.raises(ValueError):
        CapabilityDescriptor("X", "d", no_local_jev(),
                             {"PLAY": GoalSchema("PAUSE", satisfied_when=(Condition("a"),))}, ())
