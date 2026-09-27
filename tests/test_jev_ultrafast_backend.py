"""The optional Jev policy must obey this project's completion contract."""

import asyncio
import os
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest

from ping_ponder.agentic.browser_use_slice import BrowserCompletionStatus, BrowserTask
from ping_ponder.agentic.goals import SemanticGoal
from ping_ponder.agentic.jev_ultrafast_backend import (
    JevUltrafastExecutor,
    _agent_factory,
    _openrouter_policy_request,
    _text_model_settings,
)

GOAL = SemanticGoal("Browser", "FIND", {"site": "GitHub", "target": "Browser Use repository"})
DESTINATION = {"url": "https://github.com/browser-use/browser-use", "title": "browser-use/browser-use",
               "text": "browser-use repository", "fingerprint": "repo-page", "actions": []}


def test_openrouter_policy_redirects_choice_request_and_restores_source(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    calls = []

    def post_json(url, key, body):
        calls.append((url, key, body))
        return {"answers": {}}

    source = SimpleNamespace(post_json=post_json)
    with _openrouter_policy_request(source, "typesafe/jev-1.13"):
        source.post_json("https://api.typesafe.ai/v1/systemone", "unused", {"model": "jev-latest"})
        assert "TYPESAFE_API_KEY" in os.environ
    assert calls == [("https://openrouter.ai/api/alpha/decisions", "test-openrouter-key",
                      {"model": "typesafe/jev-1.13"})]
    assert source.post_json is post_json
    assert "TYPESAFE_API_KEY" not in os.environ


def test_openrouter_text_helper_gets_matching_endpoint(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    for name in ("TEXT_MODEL_API_KEY", "TEXT_MODEL_BASE_URL", "TEXT_MODEL", "TEXT_MODEL_REASONING"):
        monkeypatch.delenv(name, raising=False)
    with _text_model_settings():
        assert os.environ["TEXT_MODEL_API_KEY"] == "test-openrouter-key"
        assert os.environ["TEXT_MODEL_BASE_URL"] == "https://openrouter.ai/api/v1"
        assert os.environ["TEXT_MODEL"] == "inception/mercury-2.5"
        assert "TEXT_MODEL_REASONING" not in os.environ
    assert "TEXT_MODEL_API_KEY" not in os.environ


def test_explicit_text_model_override_is_preserved(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("TEXT_MODEL", "openai/gpt-4.1")
    with _text_model_settings():
        assert os.environ["TEXT_MODEL"] == "openai/gpt-4.1"
    assert os.environ["TEXT_MODEL"] == "openai/gpt-4.1"


def test_default_mercury_request_keeps_low_reasoning_with_json(monkeypatch):
    model = pytest.importorskip("jev_ultrafast.model")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    for name in ("TEXT_MODEL_API_KEY", "TEXT_MODEL_BASE_URL", "TEXT_MODEL",
                 "TEXT_MODEL_REASONING"):
        monkeypatch.delenv(name, raising=False)
    calls = []

    def post_json(url, key, body):
        calls.append((url, key, body))
        return {"choices": [{"message": {"content": '{"text": "Browser Use"}'}}]}

    monkeypatch.setattr(model, "post_json", post_json)
    context = {"goal": "Find Browser Use on GitHub", "field": {"label": "Search GitHub"},
               "page": {"title": "GitHub", "text": "Search"}, "recent_actions": []}
    with _text_model_settings():
        value, _ = model.field_text(context)
    assert value == "Browser Use"
    assert calls[0][0] == "https://openrouter.ai/api/v1/chat/completions"
    assert calls[0][1] == "test-openrouter-key"
    assert calls[0][2]["model"] == "inception/mercury-2.5"
    assert calls[0][2]["response_format"] == {"type": "json_object"}
    assert calls[0][2]["reasoning"] == {"effort": "low"}


def test_openrouter_text_fallback_never_sends_key_to_other_endpoint(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.setenv("TEXT_MODEL_API_KEY", "")
    monkeypatch.setenv("TEXT_MODEL_BASE_URL", "https://api.deepseek.com/v1")
    with _text_model_settings():
        assert os.environ["TEXT_MODEL_API_KEY"] == "test-openrouter-key"
        assert os.environ["TEXT_MODEL_BASE_URL"] == "https://openrouter.ai/api/v1"
    assert os.environ["TEXT_MODEL_API_KEY"] == ""
    assert os.environ["TEXT_MODEL_BASE_URL"] == "https://api.deepseek.com/v1"


def test_jev_session_requests_reduced_motion_before_first_action(monkeypatch):
    import jev_ultrafast

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")

    class Browser:
        def __init__(self):
            self.media_settings = []

        def call(self, method, **parameters):
            self.media_settings.append((method, parameters))

    class Agent:
        def __init__(self, url, goal):
            self.browser = Browser()

    monkeypatch.setattr(jev_ultrafast, "Agent", Agent)
    agent = _agent_factory("https://github.com/", "Find a repository")
    assert agent.browser.media_settings == [
        ("Emulation.setEmulatedMedia", {
            "media": "",
            "features": [{"name": "prefers-reduced-motion", "value": "reduce"}],
        }),
    ]


class FakeAgent:
    instances: ClassVar[list["FakeAgent"]] = []

    def __init__(self, url, goal, *, status="done", page=None):
        self.start_url, self.goal = url, goal
        self.status = status
        self.page = page or DESTINATION
        self.browser = SimpleNamespace(observe=self.observe)
        self.calls = []
        self.closed = False
        self.instances.append(self)

    def observe(self, screenshot=False):
        self.calls.append("observe")
        return self.page

    def command(self, name, body=None):
        self.calls.append(name)
        if name == "predict":
            return {"status": "predicted", "page": self.page}
        return {"status": self.status, "page": self.page,
                "decisions": [{"choice": "DONE"}], "history": [], "elapsed_ms": 12}

    def close(self):
        self.closed = True


class Verifier:
    def __init__(self, decision):
        self.decision = decision
        self.observations = []

    async def verify(self, task, observation):
        self.observations.append(observation)
        return self.decision


@pytest.mark.asyncio
async def test_done_requires_fresh_grounded_verifier_approval():
    FakeAgent.instances.clear()
    verifier = Verifier(BrowserCompletionStatus.UNSATISFIED)
    executor = JevUltrafastExecutor(verifier, agent_factory=FakeAgent)
    result = await executor.execute(GOAL)
    agent = FakeAgent.instances[-1]
    assert agent.start_url == "https://github.com/"
    assert "Browser Use repository" in agent.goal
    assert agent.calls == ["predict", "act", "observe"]
    assert not agent.closed
    assert verifier.observations[0].url == DESTINATION["url"]
    assert result.completion.status is BrowserCompletionStatus.UNSATISFIED
    assert result.world_changes()["browser.last_task_status"] == "unsatisfied"
    assert result.controller_calls == 1
    await executor.close()
    assert agent.closed


@pytest.mark.asyncio
async def test_website_search_goal_stops_at_results_instead_of_opening_a_result():
    FakeAgent.instances.clear()
    goal = SemanticGoal("Browser", "SEARCH_WEBSITE", {"site": "YouTube", "query": "PersonaPlex"})
    page = {"url": "https://www.youtube.com/results?search_query=PersonaPlex",
            "title": "PersonaPlex - YouTube", "text": "PersonaPlex videos and channels",
            "fingerprint": "search-results", "actions": []}
    verifier = Verifier(BrowserCompletionStatus.SATISFIED)
    executor = JevUltrafastExecutor(
        verifier, agent_factory=lambda url, instruction: FakeAgent(url, instruction, page=page))

    result = await executor.execute(goal)

    assert result.completion.status is BrowserCompletionStatus.SATISFIED
    assert FakeAgent.instances[-1].start_url == "https://www.youtube.com/"
    assert "Search YouTube for PersonaPlex" in FakeAgent.instances[-1].goal
    assert "results page" in FakeAgent.instances[-1].goal
    assert "address contains the submitted query" in FakeAgent.instances[-1].goal
    await executor.close()


def test_website_search_waits_for_results_after_submitting_query():
    goal = SemanticGoal("Browser", "SEARCH_WEBSITE", {"site": "YouTube", "query": "PersonaPlex"})
    pending = {"url": "https://www.youtube.com/results?search_query=personaplex",
               "title": "personaplex - YouTube",
               "text": "Try searching to get started", "fingerprint": "pending", "actions": []}
    loaded = {**pending, "text": "PersonaPlex videos and channels are shown in search results",
              "fingerprint": "loaded"}

    class LoadingAgent(FakeAgent):
        def __init__(self, url, instruction):
            super().__init__(url, instruction, page=pending)
            self.observations = 0

        def observe(self, screenshot=False):
            self.observations += 1
            return pending if self.observations == 1 else loaded

    executor = JevUltrafastExecutor(Verifier(BrowserCompletionStatus.SATISFIED),
                                    agent_factory=LoadingAgent)
    page, _, terminal, _ = executor._run(BrowserTask.from_semantic_goal(goal), threading.Event())

    assert terminal == "done"
    assert "videos and channels" in page["text"]
    executor._agent.close()


@pytest.mark.asyncio
async def test_verified_destination_is_satisfied_even_if_jev_blocks():
    FakeAgent.instances.clear()
    verifier = Verifier(BrowserCompletionStatus.SATISFIED)
    factory = lambda url, goal: FakeAgent(url, goal, status="blocked")
    result = await JevUltrafastExecutor(verifier, agent_factory=factory).execute(GOAL)
    assert result.completion.status is BrowserCompletionStatus.SATISFIED


@pytest.mark.asyncio
async def test_blocked_with_uncertain_evidence_remains_blocked():
    verifier = Verifier(BrowserCompletionStatus.UNCERTAIN)
    factory = lambda url, goal: FakeAgent(url, goal, status="blocked")
    result = await JevUltrafastExecutor(verifier, agent_factory=factory).execute(GOAL)
    assert result.completion.status is BrowserCompletionStatus.BLOCKED


@pytest.mark.asyncio
async def test_policy_error_closes_tab_and_remains_observable():
    class BrokenAgent(FakeAgent):
        def command(self, name, body=None):
            raise RuntimeError("policy service unavailable")

    agent = BrokenAgent("https://github.com/", "goal")
    executor = JevUltrafastExecutor(
        Verifier(BrowserCompletionStatus.SATISFIED), agent_factory=lambda *_: agent)
    with pytest.raises(RuntimeError, match="policy service unavailable"):
        await executor.execute(GOAL)
    assert agent.closed


@pytest.mark.asyncio
async def test_elapsed_budget_is_uncertain_without_fallback():
    verifier = Verifier(BrowserCompletionStatus.UNCERTAIN)
    result = await JevUltrafastExecutor(
        verifier, agent_factory=FakeAgent, max_seconds=0).execute(GOAL)
    assert result.completion.status is BrowserCompletionStatus.UNCERTAIN
    assert result.controller_calls == 0
    assert result.timings[0]["fallback"] is False


@pytest.mark.asyncio
async def test_transient_stale_page_retries_jev_decision():
    class StalePage(ValueError):
        pass

    class StaleAgent(FakeAgent):
        def command(self, name, body=None):
            if name == "act" and "stale" not in self.calls:
                self.calls.append("stale")
                raise StalePage("page changed")
            return super().command(name, body)

    agent = StaleAgent("https://github.com/", "goal")
    result = await JevUltrafastExecutor(
        Verifier(BrowserCompletionStatus.SATISFIED), agent_factory=lambda *_: agent).execute(GOAL)
    assert result.completion.status is BrowserCompletionStatus.SATISFIED
    assert result.controller_calls == 2
    assert agent.calls.count("predict") == 2


def test_transient_stale_prediction_retries_before_any_browser_action():
    class StalePage(ValueError):
        pass

    class NavigatingAgent(FakeAgent):
        def command(self, name, body=None):
            if name == "predict" and "predict" not in self.calls:
                self.calls.append("predict")
                raise StalePage("Document is navigating")
            return super().command(name, body)

    agent = NavigatingAgent("https://github.com/", "goal")
    page, cycles, terminal, _ = JevUltrafastExecutor(
        Verifier(BrowserCompletionStatus.SATISFIED), agent_factory=lambda *_: agent,
        max_seconds=3,
    )._run(BrowserTask.from_semantic_goal(GOAL), threading.Event())
    assert page["url"] == DESTINATION["url"]
    assert terminal == "done"
    assert agent.calls == ["predict", "predict", "act", "observe"]
    assert cycles == 1


def test_persistent_stale_prediction_remains_an_observable_error():
    class StalePage(ValueError):
        pass

    class NavigatingAgent(FakeAgent):
        def command(self, name, body=None):
            self.calls.append(name)
            raise StalePage("Document is navigating")

    agent = NavigatingAgent("https://github.com/", "goal")
    with pytest.raises(StalePage, match="Document is navigating"):
        JevUltrafastExecutor(
            Verifier(BrowserCompletionStatus.SATISFIED), agent_factory=lambda *_: agent,
            max_seconds=2,
        )._run(BrowserTask.from_semantic_goal(GOAL), threading.Event())
    assert agent.closed
    assert agent.calls.count("predict") >= 2
    assert "act" not in agent.calls


def test_search_results_can_appear_before_predicted_blocked_is_acted_on():
    search_url = "https://docs.python.org/3/search.html?q=asyncio+Task"
    loading = {"url": search_url, "title": "Search — Python documentation",
               "text": "Search", "fingerprint": "loading",
               "actions": [{"id": "navigation", "kind": "click", "role": "link", "label": "Search"}]}
    results = {**loading, "text": "Search results: asyncio.Task", "fingerprint": "results",
               "actions": [*loading["actions"],
                           {"id": "task-link", "kind": "click", "role": "link",
                            "label": "asyncio.Task"}]}
    destination = {"url": "https://docs.python.org/3/library/asyncio-task.html#asyncio.Task",
                   "title": "Coroutines and tasks — Python documentation",
                   "text": "asyncio.Task reference", "fingerprint": "destination", "actions": []}

    class SearchAgent:
        def __init__(self):
            self.page = loading
            self.observations = 0
            self.acted_choices = []
            self.browser = SimpleNamespace(observe=self.observe)
            self.state = {"history": [
                {"kind": "fill", "action": "Quick search", "url": "https://docs.python.org/3/"},
                {"kind": "click", "action": "Go", "url": search_url},
                {"kind": "wait", "action": "Wait", "url": search_url},
            ]}

        def observe(self, screenshot=False):
            self.observations += 1
            if self.page is loading and self.observations >= 2:
                self.page = results
            return self.page

        def command(self, name, body=None):
            if name == "predict":
                choice = ("BLOCKED" if self.page is loading else
                          "task-link" if self.page is results else "DONE")
                return {"page": self.page, "decision": {"choice": choice}}
            self.acted_choices.append("BLOCKED" if self.page is loading else
                                      "task-link" if self.page is results else "DONE")
            if self.page is loading:
                return {"status": "blocked", "page": self.page}
            if self.page is results:
                self.page = destination
                return {"status": "ready", "page": self.page}
            return {"status": "done", "page": self.page}

        def close(self):
            pass

    agent = SearchAgent()
    goal = SemanticGoal("Browser", "FIND", {"site": "docs.python.org",
                                                "target": "asyncio Task documentation"})
    page, cycles, terminal, _ = JevUltrafastExecutor(
        Verifier(BrowserCompletionStatus.SATISFIED), agent_factory=lambda *_: agent,
        max_seconds=3,
    )._run(BrowserTask.from_semantic_goal(goal), threading.Event())
    assert page["url"] == destination["url"]
    assert terminal == "done"
    assert cycles == 3
    assert agent.acted_choices == ["task-link", "DONE"]


def test_search_page_still_blocked_when_no_results_appear():
    search = {"url": "https://docs.python.org/3/search.html?q=asyncio+Task",
              "title": "Search — Python documentation", "text": "Search",
              "fingerprint": "loading", "actions": []}

    class EmptyResultsAgent(FakeAgent):
        def __init__(self):
            super().__init__("https://docs.python.org/3/", "goal", status="blocked", page=search)
            self.state = {"history": [{"kind": "click", "action": "Go", "url": search["url"]}]}

        def command(self, name, body=None):
            if name == "predict":
                self.calls.append(name)
                return {"page": self.page, "decision": {"choice": "BLOCKED"}}
            return super().command(name, body)

    agent = EmptyResultsAgent()
    page, cycles, terminal, _ = JevUltrafastExecutor(
        Verifier(BrowserCompletionStatus.SATISFIED), agent_factory=lambda *_: agent,
        max_seconds=0.25,
    )._run(BrowserTask.from_semantic_goal(GOAL), threading.Event())
    assert page["url"] == search["url"]
    assert terminal == "blocked"
    assert cycles == 1
    assert agent.calls.count("act") == 1


def test_cancellation_during_search_settle_prevents_blocked_action():
    search = {"url": "https://docs.python.org/3/search.html?q=asyncio+Task",
              "title": "Search — Python documentation", "text": "Search",
              "fingerprint": "loading", "actions": []}

    class WaitingAgent(FakeAgent):
        def __init__(self):
            super().__init__("https://docs.python.org/3/", "goal", status="blocked", page=search)
            self.state = {"history": [{"kind": "click", "action": "Go", "url": search["url"]}]}

        def command(self, name, body=None):
            if name == "predict":
                self.calls.append(name)
                return {"page": self.page, "decision": {"choice": "BLOCKED"}}
            return super().command(name, body)

    agent = WaitingAgent()
    stop = threading.Event()
    cancel = threading.Timer(0.15, stop.set)
    cancel.start()
    try:
        page, cycles, terminal, _ = JevUltrafastExecutor(
            Verifier(BrowserCompletionStatus.SATISFIED), agent_factory=lambda *_: agent,
            max_seconds=3,
        )._run(BrowserTask.from_semantic_goal(GOAL), stop)
    finally:
        cancel.join()
    assert page == {}
    assert terminal == "cancelled"
    assert cycles == 1
    assert agent.closed
    assert "act" not in agent.calls


def test_expanding_search_waits_for_visible_input_before_next_decision():
    search_button = {"id": "e9", "node": 9, "kind": "click", "role": "button",
                     "label": "Search or jump to",
                     "expanded": "false"}
    expanded_button = {**search_button, "expanded": "true"}
    email = {"id": "e12", "node": 12, "kind": "fill", "label": "Enter your email"}
    unrelated = {"id": "e16", "node": 16, "kind": "click", "label": "Back to top"}
    search_input = {"id": "e15", "node": 15, "kind": "fill", "label": "Search or jump to"}
    base = {"url": "https://github.com/", "title": "GitHub", "text": "GitHub home"}
    before = {**base, "fingerprint": "before", "actions": [search_button, email]}
    hidden = {**base, "fingerprint": "expanded-hidden",
              "actions": [expanded_button, email, unrelated]}
    visible = {**base, "fingerprint": "expanded-visible",
               "actions": [expanded_button, email, unrelated, search_input]}

    class ExpandingAgent:
        def __init__(self, *_):
            self.browser = SimpleNamespace(observe=self.observe)
            self.observations = 0
            self.step = 0
            self.chosen = []

        def observe(self, screenshot=False):
            self.observations += 1
            if self.step == 2:
                return DESTINATION
            return visible if self.observations >= 3 else hidden

        def command(self, name, body=None):
            if name == "predict":
                if self.step == 0:
                    return {"page": before, "decision": {"choice": "e9"}}
                choice = "e15" if self.observations >= 3 else "e12"
                return {"page": visible if choice == "e15" else hidden,
                        "decision": {"choice": choice}}
            self.chosen.append(body["fingerprint"])
            if self.step == 0:
                self.step = 1
                return {"status": "ready", "page": self.observe()}
            if body["fingerprint"] != "expanded-visible":
                raise ValueError("Text helper returned no valid field value; nothing typed.")
            self.step = 2
            return {"status": "done", "page": DESTINATION}

        def close(self):
            pass

    agent = ExpandingAgent()
    page, cycles, terminal, _ = JevUltrafastExecutor(
        Verifier(BrowserCompletionStatus.SATISFIED), agent_factory=lambda *_: agent,
        max_seconds=3,
    )._run(BrowserTask.from_semantic_goal(GOAL), threading.Event())
    assert page["url"] == DESTINATION["url"]
    assert terminal == "done"
    assert cycles == 2
    assert agent.chosen == ["before", "expanded-visible"]


@pytest.mark.asyncio
async def test_cancellation_after_prediction_prevents_action():
    entered = asyncio.Event()

    class SlowAgent(FakeAgent):
        def command(self, name, body=None):
            if name == "predict":
                loop.call_soon_threadsafe(entered.set)
                release_thread.wait()
            return super().command(name, body)

    import threading
    loop = asyncio.get_running_loop()
    release_thread = threading.Event()
    agent = SlowAgent("https://github.com/", "goal")
    work = asyncio.create_task(JevUltrafastExecutor(
        Verifier(BrowserCompletionStatus.UNCERTAIN), agent_factory=lambda *_: agent).execute(GOAL))
    await entered.wait()
    work.cancel()
    with pytest.raises(asyncio.CancelledError):
        await work
    release_thread.set()
    for _ in range(100):
        if agent.closed:
            break
        await asyncio.sleep(0.01)
    assert agent.closed
    assert "act" not in agent.calls


def test_runner_selects_jev_without_building_browser_use_session(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    import scripts.run_voice_agent as runner

    class Provider:
        pass

    monkeypatch.setattr(runner, "OpenRouterDecisionsProvider", Provider)
    monkeypatch.setattr(runner, "OpenRouterProvider", Provider)
    monkeypatch.setattr(runner, "BrowserUseSessionAdapter", lambda **_: pytest.fail("Browser Use was built"))
    backend = SimpleNamespace(browser_controller_model="controller", local_jev_model="jev")
    capability, session, _, verifier_provider = runner.build_browser_find(
        backend, browser_backend="jev_ultrafast", max_seconds=45)
    assert isinstance(capability.executor, JevUltrafastExecutor)
    assert session is capability.executor
    assert capability.executor.completion_verifier.provider is verifier_provider
    assert capability.executor.max_seconds == 45
