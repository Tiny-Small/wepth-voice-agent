"""The Level-2 executors must obey the same grounded FIND contract."""

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from ping_ponder.agentic.browser_use_slice import (
    BrowserAction,
    BrowserCompletionStatus,
    BrowserDecision,
    BrowserObservation,
    BrowserTaskExecutor,
    JevBrowserCompletionVerifier,
)
from ping_ponder.agentic.goals import SemanticGoal
from ping_ponder.agentic.jev_ultrafast_backend import JevUltrafastExecutor
from ping_ponder.providers.base import InferenceResponse
from ping_ponder.providers.decisions import DecisionsResponse

GOAL = SemanticGoal("Browser", "FIND", {"site": "GitHub", "target": "Browser Use repository"})
SEARCH = BrowserObservation(
    "search", "https://github.com/search?q=Browser+Use&type=repositories",
    "Repository search results · GitHub", "[7] browser-use/browser-use",
    frozenset({7}), "search",
)
REPOSITORY = BrowserObservation(
    "repository", "https://github.com/browser-use/browser-use",
    "GitHub - browser-use/browser-use", "browser-use/browser-use repository",
    frozenset(), "repository",
)
LOOKALIKE = BrowserObservation(
    "lookalike", "https://github.com/other/browser-use-agent-datasets",
    "browser-use-agent-datasets repository · GitHub", "A different repository",
    frozenset(), "lookalike",
)


class Decisions:
    def __init__(self, answer):
        self.answer = answer
        self.calls = []

    async def decide(self, *, model, state, questions):
        self.calls.append(state)
        return InferenceResponse(
            value=DecisionsResponse.model_validate({"answers": {
                "completion": {"type": "choice", "choice": self.answer, "confidence": 0.95},
            }}),
            provider="fake", model=model, latency_seconds=0.01, retries=0, usage=None,
        )


class BrowserUsePages:
    def __init__(self, destination):
        self.page = SEARCH
        self.destination = destination

    async def observe(self):
        return self.page

    async def act(self, action):
        self.page = self.destination
        return {"extracted_content": "clicked"}


class BrowserUseController:
    def __init__(self):
        self.calls = 0

    async def next_actions(self, task, observation, available_actions, memory):
        self.calls += 1
        if self.calls == 1:
            return BrowserDecision(actions=[BrowserAction(
                kind="click", index=7, observation_id="search")])
        return BrowserDecision(completion_status="uncertain")


@dataclass
class JevAgent:
    destination: BrowserObservation

    def __post_init__(self):
        self.browser = SimpleNamespace(observe=self.observe)
        self.closed = False

    def command(self, name, body=None):
        page = self.observe()
        if name == "predict":
            return {"status": "predicted", "page": page}
        return {"status": "done", "page": page, "decisions": [{"choice": "DONE"}]}

    def observe(self, screenshot=False):
        page = self.destination
        return {"url": page.url, "title": page.title, "text": page.dom,
                "fingerprint": page.grounding_fingerprint, "actions": []}

    def close(self):
        self.closed = True


async def execute(backend, goal, destination, answer):
    decisions = Decisions(answer)
    verifier = JevBrowserCompletionVerifier(decisions, model="jev")
    if backend == "browser_use":
        executor = BrowserTaskExecutor(
            BrowserUsePages(destination), BrowserUseController(),
            completion_verifier=verifier,
        )
    else:
        executor = JevUltrafastExecutor(
            verifier, agent_factory=lambda url, task: JevAgent(destination))
    try:
        return await executor.execute(goal), decisions
    finally:
        if backend == "jev_ultrafast":
            await executor.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["browser_use", "jev_ultrafast"])
async def test_canonical_repository_requires_grounded_verifier(backend):
    result, decisions = await execute(backend, GOAL, REPOSITORY, "SATISFIED")
    assert result.completion.status is BrowserCompletionStatus.SATISFIED
    assert result.completion.evidence["url"] == REPOSITORY.url
    assert result.world_changes()["browser.current_url"] == REPOSITORY.url
    assert len(decisions.calls) == 1
    assert decisions.calls[0]["goal"]["target"] == "Browser Use repository"
    assert decisions.calls[0]["observation"]["url"] == REPOSITORY.url


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["browser_use", "jev_ultrafast"])
@pytest.mark.parametrize("answer", ["NOT_SATISFIED", "UNCERTAIN"])
async def test_verifier_rejection_cannot_be_overridden_by_executor(backend, answer):
    result, decisions = await execute(backend, GOAL, REPOSITORY, answer)
    assert result.completion.status is not BrowserCompletionStatus.SATISFIED
    assert result.world_changes()["browser.last_task_status"] != "satisfied"
    assert len(decisions.calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["browser_use", "jev_ultrafast"])
@pytest.mark.parametrize("destination", [SEARCH, LOOKALIKE])
async def test_search_results_and_lookalikes_cannot_be_claimed_complete(backend, destination):
    result, _ = await execute(backend, GOAL, destination, "SATISFIED")
    assert result.completion.status is not BrowserCompletionStatus.SATISFIED
    assert result.world_changes()["browser.last_task_status"] != "satisfied"


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["browser_use", "jev_ultrafast"])
async def test_wrong_requested_site_cannot_be_claimed_complete(backend):
    goal = SemanticGoal("Browser", "FIND", {
        "site": "docs.python.org", "target": "asyncio reference",
    })
    result, decisions = await execute(backend, goal, REPOSITORY, "SATISFIED")
    assert result.completion.status is not BrowserCompletionStatus.SATISFIED
    assert decisions.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["browser_use", "jev_ultrafast"])
async def test_expected_url_is_exact_hard_constraint(backend):
    goal = SemanticGoal("Browser", "FIND", {
        "target": "Browser Use repository",
        "expected_url": "https://github.com/browser-use/browser-use/issues",
    })
    result, decisions = await execute(backend, goal, REPOSITORY, "SATISFIED")
    assert result.completion.status is not BrowserCompletionStatus.SATISFIED
    assert decisions.calls == []
