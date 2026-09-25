"""Bounded Jev browser actions are selected from fresh grounded observations."""

from dataclasses import replace

import pytest

from ping_ponder.agentic.browser_use_slice import (
    BrowserAction, BrowserCompletionStatus, BrowserDecision, BrowserElement,
    BrowserObservation, BrowserTaskExecutor, JevBrowserActionChooser,
    BrowserTask, grounded_action_candidates, BrowserActionKind,
    StaleBrowserAction,
)
from ping_ponder.agentic.goals import SemanticGoal
from ping_ponder.providers.base import InferenceResponse
from ping_ponder.providers.decisions import DecisionsResponse


GOAL = SemanticGoal("Browser", "FIND", {"site": "Example", "target": "Blue Atlas guide"})
START = BrowserObservation("start", "about:blank", "Blank", "Empty DOM tree", frozenset(), "start")
SEARCH = BrowserObservation("search", "https://example.org/search?q=blue+atlas", "Search", "results",
                            frozenset({7, 8}), "search", elements=(
                                BrowserElement(7, "a", "Blue Atlas guide", "/guides/blue-atlas"),
                                BrowserElement(8, "a", "Red Atlas guide", "/guides/red-atlas"),
                            ))
MIDDLE = BrowserObservation("middle", "https://example.org/guides", "Guides", "more guides",
                            frozenset({11}), "middle", elements=(
                                BrowserElement(11, "a", "Blue Atlas guide", "/guides/blue-atlas"),
                            ))
DESTINATION = BrowserObservation("final", "https://example.org/guides/blue-atlas",
                                 "Blue Atlas guide · Example", "The Blue Atlas guide", frozenset(), "final")


class PageFlow:
    def __init__(self, *, middle=False):
        self.page = START
        self.middle = middle
        self.actions = []

    async def observe(self):
        return self.page

    async def act(self, action):
        self.actions.append(action)
        if action.kind.value == "navigate":
            self.page = SEARCH
        elif action.index == 7:
            self.page = MIDDLE if self.middle else DESTINATION
        elif action.index == 11:
            self.page = DESTINATION
        return {"extracted_content": "acted"}


class Controller:
    def __init__(self, *, batch=False):
        self.calls = 0
        self.batch = batch

    async def next_actions(self, task, observation, available_actions, memory):
        self.calls += 1
        if observation.observation_id == "start":
            return BrowserDecision(actions=[BrowserAction(kind="navigate", url="https://example.org/search?q=blue+atlas")])
        if self.batch:
            return BrowserDecision(actions=[BrowserAction(kind="click", index=7, observation_id="search"),
                                            BrowserAction(kind="click", index=8, observation_id="search")])
        return BrowserDecision(completion_status="uncertain")


class Decisions:
    def __init__(self, *choices):
        self.choices = list(choices)
        self.calls = []

    async def decide(self, *, model, state, questions):
        self.calls.append((state, questions))
        choice = self.choices.pop(0)
        return InferenceResponse(value=DecisionsResponse.model_validate({"answers": {
            "action": {"type": "choice", "choice": choice, "confidence": 0.96}
        }}), provider="fake", model=model, latency_seconds=0.01, retries=0, usage=None)


class Completion:
    async def verify(self, task, observation):
        return (BrowserCompletionStatus.SATISFIED if "blue-atlas" in observation.url
                else BrowserCompletionStatus.UNSATISFIED)


@pytest.mark.asyncio
async def test_choice_clicks_matching_grounded_element_and_one_action_per_observation():
    browser, controller, decisions = PageFlow(), Controller(), Decisions("CLICK_7")
    result = await BrowserTaskExecutor(browser, controller, completion_verifier=Completion(),
                                       action_chooser=JevBrowserActionChooser(decisions, model="jev")).execute(GOAL)
    assert result.completion.status is BrowserCompletionStatus.SATISFIED
    assert [(a.kind.value, a.index) for a in browser.actions] == [("navigate", None), ("click", 7)]
    assert controller.calls == 1
    assert result.timing_totals["jev_action_choice_calls"] == 1
    assert result.timing_totals["jev_action_escalations"] == 0
    assert "CLICK_7" in decisions.calls[0][1]["action"]["criteria"]
    assert "CLICK_8" in decisions.calls[0][1]["action"]["criteria"]


@pytest.mark.asyncio
async def test_next_observation_rebuilds_choice_with_fresh_index():
    browser, decisions = PageFlow(middle=True), Decisions("CLICK_7", "CLICK_11")
    result = await BrowserTaskExecutor(browser, Controller(), completion_verifier=Completion(),
                                       action_chooser=JevBrowserActionChooser(decisions, model="jev")).execute(GOAL)
    assert result.completion.status is BrowserCompletionStatus.SATISFIED
    assert [a.index for a in browser.actions] == [None, 7, 11]
    assert browser.actions[-1].observation_id == "middle"
    assert "CLICK_7" not in decisions.calls[1][1]["action"]["criteria"]


@pytest.mark.asyncio
async def test_stale_jev_click_is_rejected_and_rebuilt_from_new_observation():
    class ReplacedPage(PageFlow):
        async def act(self, action):
            if action.index == 7:
                self.page = MIDDLE
                raise StaleBrowserAction("search results changed before click")
            return await super().act(action)

    browser, decisions = ReplacedPage(middle=True), Decisions("CLICK_7", "CLICK_11")
    result = await BrowserTaskExecutor(browser, Controller(), completion_verifier=Completion(),
                                       action_chooser=JevBrowserActionChooser(decisions, model="jev")).execute(GOAL)
    assert result.completion.status is BrowserCompletionStatus.SATISFIED
    assert [action.index for action in browser.actions] == [None, 11]
    assert "CLICK_7" not in decisions.calls[1][1]["action"]["criteria"]


@pytest.mark.asyncio
async def test_unknown_choice_cannot_execute_an_element_not_in_current_observation():
    browser, controller = PageFlow(), Controller()
    result = await BrowserTaskExecutor(browser, controller,
                                       action_chooser=JevBrowserActionChooser(Decisions("CLICK_999"), model="jev")).execute(GOAL)
    assert result.completion.status is BrowserCompletionStatus.UNCERTAIN
    assert [a.kind.value for a in browser.actions] == ["navigate"]
    assert controller.calls == 2
    assert result.timing_totals["jev_action_escalations"] == 1


@pytest.mark.asyncio
async def test_opt_in_trace_records_page_candidates_and_choice_error_provenance():
    browser, controller = PageFlow(), Controller()
    result = await BrowserTaskExecutor(
        browser, controller, action_chooser=JevBrowserActionChooser(Decisions("CLICK_999"), model="jev"),
        diagnostic_trace=True, max_decisions=2,
    ).execute(GOAL)
    observations = [item for item in result.timings if item["stage"] == "observation"]
    assert observations[1]["url"] == SEARCH.url
    assert observations[1]["title"] == SEARCH.title
    generated = [item for item in result.timings if item["stage"] == "action_candidate_generation"][-1]
    assert any(item["option"] == "CLICK_7" and "Blue Atlas guide" in item["description"]
               for item in generated["candidate_options"])
    choice = next(item for item in result.timings if item["stage"] == "jev_action_choice")
    assert choice["decision"] == "ESCALATE_TO_LUNA"
    assert choice["choice_origin"] == "error_fallback"
    assert choice["invalid_choice"] is True


@pytest.mark.asyncio
async def test_trace_distinguishes_an_explicit_jev_escalation():
    result = await BrowserTaskExecutor(
        PageFlow(), Controller(),
        action_chooser=JevBrowserActionChooser(Decisions("ESCALATE_TO_LUNA"), model="jev"),
        diagnostic_trace=True, max_decisions=2,
    ).execute(GOAL)
    choice = next(item for item in result.timings if item["stage"] == "jev_action_choice")
    assert choice["choice_origin"] == "provider"


@pytest.mark.parametrize("formulation, expected_choice, expected_instruction", [
    ("baseline", "ESCALATE_TO_LUNA", "Choose ESCALATE_TO_LUNA when"),
    ("none", "NONE_OF_THE_ABOVE", "Choose NONE_OF_THE_ABOVE only if none of the supplied"),
    ("strict", "ESCALATE_TO_LUNA", "Select ESCALATE_TO_LUNA ONLY when none"),
])
@pytest.mark.asyncio
async def test_choice_formulations_preserve_one_choice_and_map_none_to_runtime_escalation(
    formulation, expected_choice, expected_instruction,
):
    from ping_ponder.agentic.browser_use_slice import BrowserActionCandidate

    decisions = Decisions(expected_choice)
    chooser = JevBrowserActionChooser(decisions, model="jev", formulation=formulation)
    candidates = (
        BrowserActionCandidate("SEND_KEYS_ENTER", "SEND_KEYS Enter in the just-filled search field",
                               BrowserAction(kind="send_keys", key="Enter", observation_id="search")),
        BrowserActionCandidate("ESCALATE_TO_LUNA", "Ask Luna for one next action"),
    )

    selected = await chooser.choose(BrowserTask.from_semantic_goal(GOAL), SEARCH, candidates)

    question = decisions.calls[0][1]["action"]
    assert len(decisions.calls) == 1
    assert question["instructions"].startswith("Choose exactly ONE supplied")
    assert expected_instruction in question["instructions"]
    assert "SEND_KEYS_ENTER" in question["criteria"]
    if formulation == "none":
        assert question["criteria"]["NONE_OF_THE_ABOVE"] == "None of the supplied concrete actions applies"
        assert "ESCALATE_TO_LUNA" not in question["criteria"]
        assert selected == "ESCALATE_TO_LUNA"
        assert chooser.last_choice == "NONE_OF_THE_ABOVE"
    else:
        assert question["criteria"]["ESCALATE_TO_LUNA"] == "Ask Luna for one next action"
        assert selected == expected_choice
        assert chooser.last_choice == expected_choice


@pytest.mark.asyncio
async def test_none_formulation_trace_keeps_raw_none_and_runtime_escalation_provenance():
    browser, controller = PageFlow(), Controller()
    browser.page = SEARCH
    chooser = JevBrowserActionChooser(Decisions("NONE_OF_THE_ABOVE"), model="jev", formulation="none")
    result = await BrowserTaskExecutor(browser, controller, action_chooser=chooser,
                                       diagnostic_trace=True, max_decisions=1).execute(GOAL)

    choice = next(item for item in result.timings if item["stage"] == "jev_action_choice")
    assert choice["decision"] == "ESCALATE_TO_LUNA"
    assert choice["choice_value"] == "NONE_OF_THE_ABOVE"
    assert choice["choice_origin"] == "provider"
    assert choice["escalated"] is True
    assert controller.calls == 1


@pytest.mark.asyncio
async def test_escalation_executes_one_luna_action_then_returns_to_jev():
    browser, controller = PageFlow(middle=True), Controller(batch=True)
    decisions = Decisions("ESCALATE_TO_LUNA", "CLICK_11")
    result = await BrowserTaskExecutor(browser, controller, completion_verifier=Completion(),
                                       action_chooser=JevBrowserActionChooser(decisions, model="jev")).execute(GOAL)
    assert result.completion.status is BrowserCompletionStatus.SATISFIED
    assert [a.index for a in browser.actions] == [None, 7, 11]
    assert controller.calls == 2
    assert result.timing_totals["jev_action_escalations"] == 1


@pytest.mark.asyncio
async def test_uncertain_completion_uses_luna_before_action_choice():
    class Uncertain:
        async def verify(self, task, observation):
            return BrowserCompletionStatus.UNCERTAIN

    matching = replace(MIDDLE, title="Guides")
    browser, controller, decisions = PageFlow(), Controller(), Decisions("CLICK_7")
    browser.page = matching
    result = await BrowserTaskExecutor(browser, controller, completion_verifier=Uncertain(),
                                       action_chooser=JevBrowserActionChooser(decisions, model="jev")).execute(GOAL)
    assert result.completion.status is BrowserCompletionStatus.UNCERTAIN
    assert controller.calls == 1
    assert decisions.calls == []
    assert result.timing_totals["jev_completion_fallbacks"] == 1


@pytest.mark.asyncio
async def test_no_labeled_clicks_leaves_open_ended_navigation_to_luna():
    browser, controller, decisions = PageFlow(), Controller(), Decisions("CLICK_7")
    browser.page = START
    await BrowserTaskExecutor(browser, controller,
                              action_chooser=JevBrowserActionChooser(decisions, model="jev"),
                              max_decisions=1).execute(GOAL)
    assert controller.calls == 1
    assert decisions.calls == []


def test_link_heavy_search_filters_query_links_using_visible_label_not_href_query():
    elements = tuple(BrowserElement(index, "a", f"Unrelated filter {index}",
                                    "/search?q=Blue+Atlas+guide") for index in range(1, 31))
    elements += (BrowserElement(31, "a", "Blue Atlas guide", "/guides/blue-atlas"),)
    page = replace(SEARCH, elements=elements, interactive_indices=frozenset(range(1, 32)))
    candidates = grounded_action_candidates(BrowserTask.from_semantic_goal(GOAL), page,
                                            tuple(BrowserActionKind))
    assert "CLICK_31" in {candidate.option for candidate in candidates}
    assert "CLICK_1" not in {candidate.option for candidate in candidates}


def test_overcrowded_matching_set_offers_only_escalation():
    elements = tuple(BrowserElement(index, "a", "Blue Atlas guide", f"/guides/{index}")
                     for index in range(1, 30))
    page = replace(SEARCH, elements=elements, interactive_indices=frozenset(range(1, 30)))
    candidates = grounded_action_candidates(BrowserTask.from_semantic_goal(GOAL), page,
                                            tuple(BrowserActionKind))
    assert [candidate.option for candidate in candidates] == ["ESCALATE_TO_LUNA"]
