"""Phase 1 completion checks at the Browser Use loop boundary."""

from dataclasses import replace

import pytest

from ping_ponder.agentic.browser_use_slice import (
    BrowserAction, BrowserCompletionStatus, BrowserDecision, BrowserElement, BrowserObservation,
    BrowserTaskExecutor, JevBrowserCompletionVerifier,
)
from ping_ponder.agentic.goals import SemanticGoal
from ping_ponder.providers.base import InferenceResponse
from ping_ponder.providers.decisions import DecisionsResponse


GOAL = SemanticGoal("Browser", "FIND", {"site": "GitHub", "target": "Browser Use repository"})
SEARCH = BrowserObservation("search", "https://github.com/search?q=Browser+Use&type=repositories",
                            "Repository search results · GitHub", "[7] browser-use/browser-use",
                            frozenset({7}), "search")
REPOSITORY = BrowserObservation("repo", "https://github.com/browser-use/browser-use",
                                "GitHub - browser-use/browser-use", "browser-use/browser-use repository",
                                frozenset(), "repo")
LOOKALIKE = BrowserObservation(
    "lookalike", "https://github.com/Khang-9966/Computer-Browser-Phone-Use-Agent-Datasets",
    "Computer Browser Phone Use Agent Datasets · GitHub", "A different repository",
    frozenset(), "lookalike",
)


class Pages:
    def __init__(self, destination=REPOSITORY):
        self.page = SEARCH
        self.destination = destination
        self.actions = []

    async def observe(self):
        return self.page

    async def act(self, action):
        self.actions.append(action)
        self.page = self.destination
        return {"extracted_content": "clicked"}


class Controller:
    def __init__(self):
        self.calls = 0

    async def next_actions(self, task, observation, available_actions, memory):
        self.calls += 1
        if self.calls == 1:
            return BrowserDecision(actions=[BrowserAction(kind="click", index=7, observation_id="search")])
        return BrowserDecision(completion_status="uncertain")


class Decisions:
    def __init__(self, answer):
        self.answer = answer
        self.calls = []

    async def decide(self, *, model, state, questions):
        self.calls.append((model, state, questions))
        return InferenceResponse(value=DecisionsResponse.model_validate({"answers": {
            "completion": {"type": "choice", "choice": self.answer, "confidence": 0.95}
        }}), provider="fake", model=model, latency_seconds=0.01, retries=0, usage=None)


@pytest.mark.asyncio
async def test_grounded_repository_finishes_without_terminal_controller_call():
    browser, controller, decisions = Pages(), Controller(), Decisions("SATISFIED")
    result = await BrowserTaskExecutor(browser, controller,
                                       completion_verifier=JevBrowserCompletionVerifier(decisions, model="jev")).execute(GOAL)
    assert result.completion.status is BrowserCompletionStatus.SATISFIED
    assert controller.calls == 1
    assert len(decisions.calls) == 1
    assert result.completion.evidence["url"] == REPOSITORY.url
    assert result.timing_totals["jev_completion_s"] >= 0
    assert result.timing_totals["jev_completion_calls"] == 1
    assert result.timing_totals["jev_completion_fallbacks"] == 0
    assert decisions.calls[0][1]["goal"]["target"] == "Browser Use repository"
    assert "search" not in decisions.calls[0][1]["observation"]["url"]


@pytest.mark.asyncio
async def test_not_satisfied_continues_existing_browser_loop():
    controller, decisions = Controller(), Decisions("NOT_SATISFIED")
    result = await BrowserTaskExecutor(Pages(), controller,
                                       completion_verifier=JevBrowserCompletionVerifier(decisions, model="jev")).execute(GOAL)
    assert result.completion.status is BrowserCompletionStatus.UNCERTAIN
    assert controller.calls == 2
    assert result.timing_totals["jev_completion_fallbacks"] == 0


@pytest.mark.asyncio
async def test_uncertain_uses_existing_controller_fallback():
    controller, decisions = Controller(), Decisions("UNCERTAIN")
    result = await BrowserTaskExecutor(Pages(), controller,
                                       completion_verifier=JevBrowserCompletionVerifier(decisions, model="jev")).execute(GOAL)
    assert result.completion.status is BrowserCompletionStatus.UNCERTAIN
    assert controller.calls == 2
    assert result.timing_totals["jev_completion_fallbacks"] == 1


@pytest.mark.asyncio
async def test_jev_cannot_satisfy_goal_from_unseen_target():
    destination = replace(REPOSITORY, url="https://github.com/other/project",
                          title="GitHub - other/project", dom="A different repository")
    controller, decisions = Controller(), Decisions("NOT_SATISFIED")
    result = await BrowserTaskExecutor(Pages(destination), controller,
                                       completion_verifier=JevBrowserCompletionVerifier(decisions, model="jev")).execute(GOAL)
    assert result.completion.status is BrowserCompletionStatus.UNCERTAIN
    assert controller.calls == 2
    assert len(decisions.calls) == 1
    assert result.timing_totals["jev_completion_fallbacks"] == 0


@pytest.mark.asyncio
async def test_lookalike_repository_does_not_trigger_jev_completion():
    controller, decisions = Controller(), Decisions("NOT_SATISFIED")
    result = await BrowserTaskExecutor(Pages(LOOKALIKE), controller,
                                       completion_verifier=JevBrowserCompletionVerifier(decisions, model="jev")).execute(GOAL)
    assert result.completion.status is BrowserCompletionStatus.UNCERTAIN
    assert len(decisions.calls) == 1


@pytest.mark.asyncio
async def test_descriptive_destination_reaches_completion_without_contiguous_target_phrase():
    goal = SemanticGoal("Browser", "FIND", {
        "site": "Wikipedia", "target": "Python programming language article",
    })
    destination = BrowserObservation(
        "python-article", "https://en.wikipedia.org/wiki/Python_(programming_language)",
        "Python (programming language) - Wikipedia", "Python is a high-level programming language.",
        frozenset(), "python-article",
    )
    decisions = Decisions("SATISFIED")
    result = await BrowserTaskExecutor(Pages(destination), Controller(),
        completion_verifier=JevBrowserCompletionVerifier(decisions, model="jev")).execute(goal)
    assert result.completion.status is BrowserCompletionStatus.SATISFIED
    assert len(decisions.calls) == 1


@pytest.mark.asyncio
async def test_wrong_requested_domain_never_reaches_completion():
    goal = SemanticGoal("Browser", "FIND", {"site": "docs.python.org", "target": "asyncio reference"})
    decisions = Decisions("SATISFIED")
    result = await BrowserTaskExecutor(Pages(REPOSITORY), Controller(),
        completion_verifier=JevBrowserCompletionVerifier(decisions, model="jev")).execute(goal)
    assert result.completion.status is BrowserCompletionStatus.UNCERTAIN
    assert decisions.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("observation", [
    BrowserObservation("blank", "about:blank", "Empty Tab", "", frozenset(), "blank"),
    BrowserObservation("error", "chrome-error://chromewebdata/", "This site can’t be reached",
                       "ERR_NAME_NOT_RESOLVED", frozenset(), "error"),
    BrowserObservation("results", "https://www.google.com/search?q=python+asyncio",
                       "Google Search", "results", frozenset(), "results"),
    BrowserObservation("github-results", "https://github.com/search?q=Browser+Use&type=repositories",
                       "GitHub", "Repositories for Browser Use", frozenset(), "github-results"),
    BrowserObservation("loading", "https://docs.python.org/3/library/asyncio.html",
                       "Loading", "", frozenset(), "loading"),
])
async def test_non_destination_observation_does_not_reach_completion(observation):
    decisions = Decisions("SATISFIED")
    result = await BrowserTaskExecutor(Pages(observation), Controller(),
        completion_verifier=JevBrowserCompletionVerifier(decisions, model="jev")).execute(GOAL)
    assert result.completion.status is BrowserCompletionStatus.UNCERTAIN
    assert decisions.calls == []


@pytest.mark.asyncio
async def test_luna_cannot_claim_lookalike_repository_satisfies_goal():
    class ClaimingController(Controller):
        async def next_actions(self, task, observation, available_actions, memory):
            self.calls += 1
            if self.calls == 1:
                return BrowserDecision(actions=[BrowserAction(kind="click", index=7, observation_id="search")])
            return BrowserDecision(completion_status="satisfied")

    result = await BrowserTaskExecutor(Pages(LOOKALIKE), ClaimingController()).execute(GOAL)
    assert result.completion.status is BrowserCompletionStatus.UNCERTAIN


@pytest.mark.asyncio
async def test_explicit_expected_url_remains_an_exact_hard_constraint():
    expected = "https://docs.python.org/3/library/asyncio.html"
    goal = SemanticGoal("Browser", "FIND", {"target": "asyncio documentation", "expected_url": expected})
    decisions = Decisions("SATISFIED")
    result = await BrowserTaskExecutor(Pages(REPOSITORY), Controller(),
        completion_verifier=JevBrowserCompletionVerifier(decisions, model="jev")).execute(goal)
    assert result.completion.status is BrowserCompletionStatus.UNCERTAIN
    assert decisions.calls == []


@pytest.mark.asyncio
async def test_luna_cannot_bypass_jev_completion_for_semantic_find():
    class ClaimingController(Controller):
        async def next_actions(self, task, observation, available_actions, memory):
            self.calls += 1
            if self.calls == 1:
                return BrowserDecision(actions=[BrowserAction(kind="click", index=7, observation_id="search")])
            return BrowserDecision(completion_status="satisfied")

    result = await BrowserTaskExecutor(Pages(REPOSITORY), ClaimingController()).execute(GOAL)
    assert result.completion.status is BrowserCompletionStatus.UNCERTAIN


@pytest.mark.asyncio
async def test_completion_receives_grounded_clicked_result_text():
    class GroundedPages(Pages):
        async def observe(self):
            if self.page is SEARCH:
                return replace(SEARCH, elements=(BrowserElement(
                    index=7, tag="a", text="browser-use/browser-use", href="https://github.com/browser-use/browser-use"),))
            return self.destination

    decisions = Decisions("SATISFIED")
    result = await BrowserTaskExecutor(GroundedPages(), Controller(),
        completion_verifier=JevBrowserCompletionVerifier(decisions, model="jev")).execute(GOAL)
    assert result.completion.status is BrowserCompletionStatus.SATISFIED
    evidence = decisions.calls[0][1]["observation"]["grounded_navigation_evidence"]
    assert evidence == ["browser-use/browser-use", "https://github.com/browser-use/browser-use"]


@pytest.mark.asyncio
async def test_completion_prompt_preserves_uncertainty_for_underspecified_targets():
    decisions = Decisions("UNCERTAIN")
    goal = SemanticGoal("Browser", "FIND", {"target": "Mercury"})
    observed = BrowserObservation("mercury", "https://en.wikipedia.org/wiki/Mercury_(planet)",
                                  "Mercury (planet) - Wikipedia", "Mercury is a planet.",
                                  frozenset(), "mercury")
    from ping_ponder.agentic.browser_use_slice import BrowserTask
    result = await JevBrowserCompletionVerifier(decisions, model="jev").verify(
        BrowserTask.from_semantic_goal(goal), observed,
    )
    assert result is BrowserCompletionStatus.UNCERTAIN
    question = decisions.calls[0][2]["completion"]
    assert "multiple plausible meanings or destinations" in question["instructions"]
    assert "ambiguous" in question["criteria"]["UNCERTAIN"]


@pytest.mark.asyncio
async def test_bare_one_term_without_site_remains_uncertain():
    goal = SemanticGoal("Browser", "FIND", {"target": "Mercury"})
    destination = BrowserObservation("planet", "https://en.wikipedia.org/wiki/Mercury_(planet)",
                                     "Mercury (planet) - Wikipedia", "Mercury is a planet.",
                                     frozenset(), "planet")
    decisions = Decisions("SATISFIED")
    result = await BrowserTaskExecutor(Pages(destination), Controller(),
        completion_verifier=JevBrowserCompletionVerifier(decisions, model="jev")).execute(goal)
    assert result.completion.status is BrowserCompletionStatus.UNCERTAIN
    assert decisions.calls == []
