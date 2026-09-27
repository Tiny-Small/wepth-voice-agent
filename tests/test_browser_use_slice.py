from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from ping_ponder.agentic.browser_use_slice import (
    BrowserAction,
    BrowserActionKind,
    BrowserCompletionStatus,
    BrowserDecision,
    JevBrowserCapability,
    BrowserObservation,
    BrowserTask,
    BrowserTaskExecutor,
    BrowserUseSessionAdapter,
    ModelBrowserController,
    StaleBrowserAction,
)
from ping_ponder.agentic.goals import SemanticGoal
from ping_ponder.agentic.world import WorldState


class FixtureBrowser:
    """Deterministic page surface; navigation, input, and DOM changes are explicit."""

    def __init__(self) -> None:
        self.page = "start"
        self.dom_revision = 0
        self.executed: list[BrowserAction] = []

    async def observe(self) -> BrowserObservation:
        dom = "Search box" if self.page == "start" else "Repository: Browser Use"
        return BrowserObservation(
            observation_id=f"{self.page}-{self.dom_revision}",
            url="https://example.test/search",
            title="Fixture",
            dom=dom,
            interactive_indices=frozenset({3} if self.page == "start" else {8}),
            grounding_fingerprint=f"{self.page}-{self.dom_revision}",
        )

    async def act(self, action: BrowserAction):
        self.executed.append(action)
        if action.kind is BrowserActionKind.CLICK and self.page == "start":
            self.page = "result"
            self.dom_revision += 1
        return {"error": None, "extracted_content": action.kind.value}


class DestinationBrowser(FixtureBrowser):
    async def observe(self) -> BrowserObservation:
        observation = await super().observe()
        if self.page == "result":
            return replace(observation, url="https://www.iana.org/domains")
        return replace(observation, url="https://www.iana.org/domains/reserved")


class QueueController:
    def __init__(self, *decisions: BrowserDecision) -> None:
        self.decisions = list(decisions)
        self.observations: list[BrowserObservation] = []

    async def next_actions(self, task, observation, available_actions, memory):
        self.observations.append(observation)
        return self.decisions.pop(0)


def test_semantic_goal_projects_to_browser_task_without_copying_goal_state():
    goal = SemanticGoal(
        "Browser", "FIND",
        {"target": "IANA Domains page", "expected_url": "https://www.iana.org/domains"},
    )

    task = BrowserTask.from_semantic_goal(goal)

    assert task.semantic_goal is goal
    assert task.target == "IANA Domains page"
    assert task.expected_url == "https://www.iana.org/domains"


def test_indexed_actions_require_the_observation_that_grounded_the_index():
    with pytest.raises(ValueError, match="observation_id"):
        BrowserAction(kind="click", index=3)
    assert BrowserAction(kind="click", index=3, observation_id="current", pages=0).pages == 0


def test_controller_schema_is_valid_for_strict_structured_outputs():
    schema = BrowserDecision.model_json_schema()
    action_schema = schema["$defs"]["BrowserAction"]

    assert set(schema["required"]) == set(schema["properties"])
    assert set(action_schema["required"]) == set(action_schema["properties"])
    assert action_schema["properties"]["pages"].get("exclusiveMinimum") is None
    evidence_schema = schema["$defs"]["BrowserEvidence"]
    assert evidence_schema["additionalProperties"] is False
    assert set(evidence_schema["required"]) == set(evidence_schema["properties"])


@pytest.mark.asyncio
async def test_stale_index_action_is_rejected_before_browser_execution():
    browser = DestinationBrowser()
    stale = BrowserAction(kind="click", index=3, observation_id="older-page-1")
    fresh = BrowserAction(kind="click", index=3, observation_id="start-0")
    controller = QueueController(BrowserDecision(actions=[stale]), BrowserDecision(actions=[fresh]))
    runner = BrowserTaskExecutor(browser, controller)

    result = await runner.execute(SemanticGoal("Browser", "FIND", {
        "target": "IANA Domains page", "expected_url": "https://www.iana.org/domains",
    }))

    assert browser.executed == [fresh]
    assert len(controller.observations) == 2
    assert result.completion.status is BrowserCompletionStatus.SATISFIED
    assert result.controller_calls == 2
    assert [timing["stage"] for timing in result.timings] == [
        "observation", "controller", "observation", "controller",
        "browser_action", "observation", "completion",
    ]
    assert all(timing["latency_s"] >= 0 for timing in result.timings)
    assert result.timing_totals["controller_s"] > 0
    assert result.timing_totals["browser_action_s"] > 0
    assert result.timing_totals["observation_s"] > 0
    assert [timing["number"] for timing in result.timings if timing["stage"] == "observation"] == [1, 2, 3]
    controller_timings = [timing for timing in result.timings if timing["stage"] == "controller"]
    assert [timing["decision"] for timing in controller_timings] == ["click", "click"]
    assert [timing["target"] for timing in controller_timings] == ["[3]", "[3]"]


@pytest.mark.asyncio
async def test_adapter_requires_fresh_observation_after_an_action_or_restart():
    class StubSession:
        async def start(self):
            pass

        async def kill(self):
            pass

    class StubRegistry:
        def __init__(self):
            self.calls = 0

        async def execute_action(self, **kwargs):
            self.calls += 1
            return {"extracted_content": "clicked"}

    registry = StubRegistry()
    adapter = BrowserUseSessionAdapter(
        browser_session=StubSession(), tools=SimpleNamespace(registry=registry)
    )
    await adapter.start()
    adapter._last_observation = await FixtureBrowser().observe()
    adapter._observation_ready = True
    click = BrowserAction(kind="click", index=3, observation_id="start-0")

    await adapter.act(click)
    with pytest.raises(StaleBrowserAction):
        await adapter.act(click)
    await adapter.close(kill_browser=True)
    await adapter.start()
    with pytest.raises(StaleBrowserAction):
        await adapter.act(click)
    assert registry.calls == 1


@pytest.mark.asyncio
async def test_adapter_accepts_grounded_nonindexed_navigation_with_observation_id():
    class StubSession:
        async def start(self):
            pass

        async def kill(self):
            pass

    class StubRegistry:
        def __init__(self):
            self.calls = 0

        async def execute_action(self, **kwargs):
            self.calls += 1
            return {"extracted_content": "navigated"}

    registry = StubRegistry()
    adapter = BrowserUseSessionAdapter(
        browser_session=StubSession(), tools=SimpleNamespace(registry=registry)
    )
    await adapter.start()
    adapter._last_observation = await FixtureBrowser().observe()
    adapter._observation_ready = True

    result = await adapter.act(BrowserAction(
        kind="navigate", url="https://github.com", observation_id="start-0",
    ))

    assert result["extracted_content"] == "navigated"
    assert registry.calls == 1


@pytest.mark.asyncio
async def test_adapter_maps_constrained_enter_action_to_browser_use_send_keys():
    class StubSession:
        async def start(self):
            pass

        async def kill(self):
            pass

    class StubRegistry:
        def __init__(self):
            self.call = None

        async def execute_action(self, **kwargs):
            self.call = kwargs
            return {"extracted_content": "submitted"}

    registry = StubRegistry()
    adapter = BrowserUseSessionAdapter(
        browser_session=StubSession(), tools=SimpleNamespace(registry=registry)
    )
    await adapter.start()
    adapter._last_observation = await FixtureBrowser().observe()
    adapter._observation_ready = True

    await adapter.act(BrowserAction(
        kind="send_keys", observation_id="start-0", key="Enter",
    ))

    assert registry.call["action_name"] == "send_keys"
    assert registry.call["params"] == {"keys": "Enter"}


@pytest.mark.asyncio
async def test_expected_url_completes_after_fresh_observation_without_second_controller_call():
    browser = DestinationBrowser()
    controller = QueueController(BrowserDecision(actions=[
        BrowserAction(kind="click", index=3, observation_id="start-0"),
    ]))
    result = await BrowserTaskExecutor(browser, controller).execute(
        SemanticGoal("Browser", "FIND", {
            "site": "https://www.iana.org/domains/reserved",
            "target": "IANA Domains page",
            "expected_url": "https://www.iana.org/domains",
        })
    )

    assert result.completion.status is BrowserCompletionStatus.SATISFIED
    assert result.completion.evidence["url"] == "https://www.iana.org/domains"
    assert result.controller_calls == 1
    assert len(controller.observations) == 1


@pytest.mark.asyncio
async def test_expected_url_already_current_needs_no_controller_call():
    browser = DestinationBrowser()
    browser.page = "result"
    controller = QueueController()
    result = await BrowserTaskExecutor(browser, controller).execute(
        SemanticGoal("Browser", "FIND", {
            "target": "IANA Domains page", "expected_url": "https://www.iana.org/domains",
        })
    )

    assert result.completion.status is BrowserCompletionStatus.SATISFIED
    assert result.controller_calls == 0


@pytest.mark.asyncio
async def test_controller_timing_records_navigation_target_without_input_text():
    destination = "https://github.com/search?q=browser-use&type=repositories"
    controller = QueueController(
        BrowserDecision(actions=[BrowserAction(kind="navigate", url=destination)]),
        BrowserDecision(actions=[BrowserAction(
            kind="input", index=3, observation_id="start-0", text="private search text",
        )]),
        BrowserDecision(completion_status="uncertain"),
    )
    result = await BrowserTaskExecutor(FixtureBrowser(), controller).execute(
        SemanticGoal("Browser", "FIND", {"target": "repository"})
    )

    controller_timings = [timing for timing in result.timings if timing["stage"] == "controller"]
    assert controller_timings[0]["decision"] == "navigate"
    assert controller_timings[0]["target"] == destination
    assert controller_timings[1]["decision"] == "input"
    assert controller_timings[1]["target"] == "[3]"
    assert controller_timings[2]["decision"] == "complete"
    assert controller_timings[2]["status"] == "uncertain"
    assert "private search text" not in json.dumps(result.timings)


@pytest.mark.asyncio
async def test_controller_cannot_claim_satisfaction_before_expected_url_is_observed():
    controller = QueueController(BrowserDecision(completion_status="satisfied"))
    result = await BrowserTaskExecutor(DestinationBrowser(), controller).execute(
        SemanticGoal("Browser", "FIND", {
            "target": "IANA Domains page", "expected_url": "https://www.iana.org/domains",
        })
    )

    assert result.completion.status is BrowserCompletionStatus.UNCERTAIN
    assert result.controller_calls == 1
    controller_timing = next(timing for timing in result.timings if timing["stage"] == "controller")
    assert controller_timing["decision"] == "complete"
    assert controller_timing["status"] == "satisfied"


@pytest.mark.asyncio
async def test_controller_prompt_separates_semantic_target_and_url_acceptance():
    class RecordingProvider:
        async def infer(self, *, model, messages, response_model):
            self.messages = messages
            return SimpleNamespace(value=BrowserDecision(completion_status="uncertain"))

    provider = RecordingProvider()
    goal = SemanticGoal("Browser", "FIND", {
        "site": "https://www.iana.org/domains/reserved",
        "target": "IANA Domains page",
        "expected_url": "https://www.iana.org/domains",
    })
    observation = await DestinationBrowser().observe()
    await ModelBrowserController(provider, model="test/model").next_actions(
        BrowserTask.from_semantic_goal(goal), observation, tuple(BrowserActionKind), ""
    )
    request = json.loads(provider.messages[1]["content"])

    assert request["intent"] == "FIND"
    assert request["target"] == "IANA Domains page"
    assert request["success_condition"] == {"url_equals": "https://www.iana.org/domains"}
    assert "goal" not in request
    prompt = provider.messages[0]["content"]
    assert "stable search URL" in prompt
    assert "use grounded page controls" in prompt
    assert "navigate to that site before searching" not in prompt


@pytest.mark.asyncio
async def test_site_search_controller_targets_results_page_without_opening_a_result():
    class RecordingProvider:
        async def infer(self, *, model, messages, response_model):
            self.messages = messages
            return SimpleNamespace(value=BrowserDecision(completion_status="uncertain"))

    provider = RecordingProvider()
    goal = SemanticGoal("Browser", "SEARCH_WEBSITE", {"site": "GitHub", "query": "browser-use"})
    observation = await DestinationBrowser().observe()
    await ModelBrowserController(provider, model="test/model").next_actions(
        BrowserTask.from_semantic_goal(goal), observation, tuple(BrowserActionKind), "")

    request = json.loads(provider.messages[1]["content"])
    assert request["success_condition"] == {"site_search_results_for": "browser-use"}
    assert "stop on the search results page" in provider.messages[0]["content"]


@pytest.mark.asyncio
async def test_same_url_dom_rewrite_stops_batch_and_reobserves():
    browser = FixtureBrowser()
    observed = await browser.observe()
    controller = QueueController(
        BrowserDecision(actions=[
            BrowserAction(kind="click", index=3, observation_id=observed.observation_id),
            BrowserAction(kind="click", index=3, observation_id=observed.observation_id),
        ]),
        BrowserDecision(
            completion_status="satisfied",
            evidence={
                "url": "https://untrusted.invalid",
                "title": "untrusted title",
                "matched_text": "Repository: Browser Use",
            },
        ),
    )
    runner = BrowserTaskExecutor(browser, controller)

    result = await runner.execute(SemanticGoal("Browser", "FIND", {"target": "repo"}))

    assert len(browser.executed) == 1
    assert len(controller.observations) == 2
    assert controller.observations[0].url == controller.observations[1].url
    assert controller.observations[0].observation_id != controller.observations[1].observation_id
    assert result.completion.status is BrowserCompletionStatus.UNCERTAIN
    controller_timings = [timing for timing in result.timings if timing["stage"] == "controller"]
    assert controller_timings[0]["decision"] == "batch"
    assert controller_timings[0]["actions"] == ["click", "click"]
    assert controller_timings[1]["decision"] == "complete"
    assert controller_timings[1]["status"] == "satisfied"
    assert "dom" not in result.world_changes()
    updated_world = result.update_world(WorldState())
    assert updated_world.get("browser.last_task_status") == "uncertain"
    assert updated_world.get("browser.current_domain") == "example.test"
    assert updated_world.get("browser.last_task_evidence")["url"] == result.observation.url
    assert updated_world.get("browser.last_task_evidence")["title"] == result.observation.title
    assert "matched_text" not in updated_world.get("browser.last_task_evidence")


@pytest.mark.asyncio
async def test_batch_continues_when_grounding_is_unchanged():
    browser = FixtureBrowser()
    observation = await browser.observe()
    controller = QueueController(
        BrowserDecision(actions=[
            BrowserAction(kind="input", index=3, observation_id=observation.observation_id, text="Browser Use"),
            BrowserAction(kind="click", index=3, observation_id=observation.observation_id),
        ]),
        BrowserDecision(completion_status="satisfied"),
    )
    runner = BrowserTaskExecutor(browser, controller)

    _, updated_world = await JevBrowserCapability(runner).execute(
        SemanticGoal("Browser", "FIND", {"target": "repo"}), WorldState()
    )

    assert [action.kind for action in browser.executed] == [BrowserActionKind.INPUT, BrowserActionKind.CLICK]
    assert updated_world.get("browser.last_task_status") == "uncertain"
    assert updated_world.get("browser.current_url") == "https://example.test/search"
    assert "browser.dom" not in updated_world


@pytest.mark.asyncio
async def test_repeated_action_without_page_progress_stops_controller_loop():
    browser = FixtureBrowser()
    first = await browser.observe()
    after_result = BrowserObservation(
        observation_id="result-1",
        url=first.url,
        title=first.title,
        dom="Repository: Browser Use",
        interactive_indices=frozenset({8}),
        grounding_fingerprint="result-1",
    )
    click_start = BrowserAction(kind="click", index=3, observation_id=first.observation_id)
    click_again = BrowserAction(kind="click", index=8, observation_id="result-1")
    controller = QueueController(
        BrowserDecision(actions=[click_start]),
        BrowserDecision(actions=[click_again]),
        BrowserDecision(actions=[click_again]),
    )

    result = await BrowserTaskExecutor(browser, controller).execute(
        SemanticGoal("Browser", "FIND", {"target": "repo"})
    )

    assert len(browser.executed) == 2
    assert result.completion.status is BrowserCompletionStatus.BLOCKED
    assert "no grounded progress" in result.completion.reason
