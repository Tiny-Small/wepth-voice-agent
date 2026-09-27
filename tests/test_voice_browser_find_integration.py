"""Voice composition and lifecycle checks for the existing Level 2 FIND operator."""

import asyncio
from types import SimpleNamespace

import pytest

from ping_ponder.agentic.adapters.memory import (
    MemoryBrowserAdapter,
    MemorySpotifyAdapter,
)
from ping_ponder.agentic.browser_use_slice import (
    BrowserAction,
    BrowserCompletionStatus,
    BrowserDecision,
    BrowserObservation,
    BrowserTaskExecutor,
    JevBrowserCapability,
)
from ping_ponder.agentic.capabilities.browser import WORLD_SCHEMA
from ping_ponder.agentic.capabilities.spotify import (
    WORLD_SCHEMA as SPOTIFY_WORLD_SCHEMA,
)
from ping_ponder.agentic.execution_config import AdapterBundle
from ping_ponder.agentic.goal_builder import GoalBuilder
from ping_ponder.agentic.goals import SemanticGoal
from ping_ponder.agentic.jev_ultrafast_backend import JevUltrafastExecutor
from ping_ponder.agentic.span import ExtractedSpan
from ping_ponder.agentic.spine import VoiceActionSpine
from ping_ponder.agentic.wiring import (
    RuleBasedGlobalJev,
    RuleBasedLocalJevs,
    build_default_registry,
    build_default_world,
)
from ping_ponder.voice.events import FinalTranscript, SpeechStarted, TranscriptState
from ping_ponder.voice.fast_voice import (
    ParallelVoiceCoordinator,
    ParallelVoiceSettings,
)

FIND = "Find the Browser Use repository on GitHub"


class Extractor:
    name = "fixture"

    async def extract_many(self, utterance, requests):
        values = {"site": "GitHub", "target": "Browser Use repository"}
        return {request.slot: ExtractedSpan(
            slot=request.slot, question=request.question, text=values[request.slot],
            start=utterance.index(values[request.slot]),
            end=utterance.index(values[request.slot]) + len(values[request.slot]),
            confidence=1.0, extractor=self.name,
        ) for request in requests if request.slot in values}


class FastVoice:
    async def stream(self, text, route):
        yield "Sure, let me check."


class Page:
    def __init__(self, url="https://github.com/browser-use/browser-use", gate=None):
        self.url, self.gate = url, gate
        self.started = asyncio.Event()
        self.actions = []
        self.cancelled = False

    async def observe(self):
        self.started.set()
        if self.gate:
            try:
                await self.gate.wait()
            except asyncio.CancelledError:
                self.cancelled = True
                raise
        return BrowserObservation(
            observation_id="page-1", url=self.url, title="browser-use/browser-use",
            dom="browser-use repository", interactive_indices=frozenset(),
            grounding_fingerprint="page-1")

    async def act(self, action):
        self.actions.append(action)
        return {}


class UncertainController:
    async def next_actions(self, task, observation, available_actions, memory):
        return BrowserDecision(completion_status=BrowserCompletionStatus.UNCERTAIN)


class Verifier:
    async def verify(self, task, observation):
        return (BrowserCompletionStatus.SATISFIED
                if observation.url == "https://github.com/browser-use/browser-use"
                else BrowserCompletionStatus.UNCERTAIN)


def spine_for(page, *, browser_open=False, browser_find=None):
    if browser_find is None:
        browser_find = JevBrowserCapability(BrowserTaskExecutor(
            page, UncertainController(), completion_verifier=Verifier()))
    browser_state = {**WORLD_SCHEMA, "browser.running": browser_open}
    registry = build_default_registry(
        local_jevs=RuleBasedLocalJevs(),
        adapters=AdapterBundle(MemorySpotifyAdapter(SPOTIFY_WORLD_SCHEMA),
                               MemoryBrowserAdapter(browser_state)),
        browser_find=browser_find)
    spine = VoiceActionSpine(registry=registry, global_jev=RuleBasedGlobalJev(),
                             goal_builder=GoalBuilder(Extractor()), world=build_default_world())
    return spine, browser_find


def event(turn, text, revision=1):
    return FinalTranscript(TranscriptState(turn, revision, text, True))


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["browser_use", "jev_ultrafast"])
async def test_natural_language_find_reaches_level_two_executor_and_world_state(backend):
    page = Page()
    browser_find = None
    if backend == "jev_ultrafast":
        class Agent:
            def __init__(self, url, task):
                self.browser = SimpleNamespace(observe=self.observe)

            def observe(self, screenshot=False):
                return {"url": page.url, "title": "browser-use/browser-use",
                        "text": "browser-use repository", "fingerprint": "repo", "actions": []}

            def command(self, name, body=None):
                observed = self.observe()
                return ({"status": "predicted", "page": observed} if name == "predict"
                        else {"status": "done", "page": observed, "decisions": [{"choice": "DONE"}]})

            def close(self):
                pass

        browser_find = JevBrowserCapability(JevUltrafastExecutor(
            Verifier(), agent_factory=Agent))
    spine, browser_find = spine_for(page, browser_find=browser_find)
    try:
        outcome = await spine.resolve_final(FIND)
        assert outcome.goal.goal_type == "FIND"
        assert outcome.goal.argument("site") == "GitHub"
        assert outcome.goal.argument("target") == "Browser Use repository"
        assert outcome.report.executed == ("FindWithGroundedBrowser",)
        assert outcome.report.satisfied
        assert browser_find.last_result.completion.status is BrowserCompletionStatus.SATISFIED
        assert spine.world.get("browser.current_url") == page.url
    finally:
        if backend == "jev_ultrafast":
            await browser_find.executor.close()


@pytest.mark.asyncio
async def test_find_from_already_open_browser_skips_invalid_navigation_candidate():
    spine, browser_find = spine_for(Page(), browser_open=True)
    outcome = await spine.resolve_final(FIND)
    assert outcome.report.executed == ("FindWithGroundedBrowser",)
    assert browser_find.last_result.completion.status is BrowserCompletionStatus.SATISFIED


@pytest.mark.asyncio
async def test_find_stays_in_voice_control_until_complete_and_speaks_result():
    gate = asyncio.Event()
    page = Page(gate=gate)
    spine, _ = spine_for(page)
    coordinator = ParallelVoiceCoordinator(
        spine=spine, fast_voice=FastVoice(),
        settings=ParallelVoiceSettings(progress_silence_s=0.01))
    await coordinator.start()
    coordinator.dispatch(event("find", FIND))
    await page.started.wait()
    assert coordinator._current_control_task.running
    await asyncio.sleep(0.03)
    assert coordinator.telemetry["find"].progress_speech_used
    gate.set()
    await coordinator.wait_idle()
    assert coordinator.spoken[-1] == "Found it."
    assert not coordinator._current_control_task.running
    await coordinator.stop()


@pytest.mark.asyncio
async def test_quick_find_suppresses_progress_but_reports_success():
    spine, _ = spine_for(Page())
    coordinator = ParallelVoiceCoordinator(
        spine=spine, fast_voice=FastVoice(),
        settings=ParallelVoiceSettings(progress_silence_s=0.1))
    await coordinator.start()
    coordinator.dispatch(event("quick", FIND))
    await coordinator.wait_idle()
    assert not coordinator.telemetry["quick"].progress_speech_used
    assert coordinator.spoken[-1] == "Found it."
    await coordinator.stop()


@pytest.mark.asyncio
async def test_uncertain_find_is_reported_as_failure_to_voice():
    spine, browser_find = spine_for(Page(url="https://github.com/"))
    coordinator = ParallelVoiceCoordinator(spine=spine, fast_voice=FastVoice())
    await coordinator.start()
    coordinator.dispatch(event("uncertain", FIND))
    await coordinator.wait_idle()
    assert browser_find.last_result.completion.status is BrowserCompletionStatus.UNCERTAIN
    assert spine.world.get("browser.last_task_status") == "uncertain"
    assert coordinator.spoken[-1] == "I couldn't complete that in Browser."
    await coordinator.stop()


@pytest.mark.asyncio
async def test_authoritative_new_turn_cancels_level_two_and_runs_next_command():
    gate = asyncio.Event()
    page = Page(gate=gate)
    spine, _ = spine_for(page)
    coordinator = ParallelVoiceCoordinator(
        spine=spine, fast_voice=FastVoice(),
        settings=ParallelVoiceSettings(progress_silence_s=0.1))
    await coordinator.start()
    coordinator.dispatch(event("old", FIND))
    await page.started.wait()
    coordinator.dispatch(SpeechStarted("new"))
    coordinator.dispatch(event("new", "Open the browser"))
    await coordinator.wait_idle()
    assert page.cancelled
    assert page.actions == []
    assert spine.world.get("browser.running") is True
    assert "Found it." not in coordinator.spoken
    assert not coordinator.telemetry["old"].progress_speech_used
    await coordinator.stop()


@pytest.mark.asyncio
async def test_cancelling_controller_request_prevents_further_browser_actions():
    page = Page(url="https://github.com/")
    entered = asyncio.Event()
    release = asyncio.Event()

    class Controller:
        async def next_actions(self, task, observation, available_actions, memory):
            entered.set()
            await release.wait()
            return BrowserDecision(actions=[BrowserAction(
                kind="navigate", url="https://github.com/browser-use/browser-use")])

    executor = BrowserTaskExecutor(page, Controller(), completion_verifier=Verifier())
    goal = SemanticGoal(capability="Browser", goal_type="FIND",
                        arguments={"site": "GitHub", "target": "Browser Use repository"})
    work = asyncio.create_task(executor.execute(goal))
    await entered.wait()
    work.cancel()
    with pytest.raises(asyncio.CancelledError):
        await work
    release.set()
    await asyncio.sleep(0)
    assert page.actions == []


@pytest.mark.asyncio
async def test_authoritative_revision_cancels_old_find():
    gate = asyncio.Event()
    page = Page(gate=gate)
    spine, _ = spine_for(page)
    coordinator = ParallelVoiceCoordinator(spine=spine, fast_voice=FastVoice())
    await coordinator.start()
    coordinator.dispatch(event("same", FIND))
    await page.started.wait()
    coordinator.dispatch(event("same", "Open the browser", revision=2))
    await coordinator.wait_idle()
    assert page.cancelled
    assert coordinator.telemetry["same"].revision == 2
    assert "Found it." not in coordinator.spoken
    await coordinator.stop()


def test_canonical_runner_builds_existing_level_two_components(monkeypatch):
    import scripts.run_voice_agent as runner

    class Provider:
        pass

    class Session:
        pass

    monkeypatch.setattr(runner, "OpenRouterDecisionsProvider", Provider)
    monkeypatch.setattr(runner, "OpenRouterProvider", Provider)
    monkeypatch.setattr(runner, "BrowserUseSessionAdapter", Session)
    backend = SimpleNamespace(browser_controller_model="controller", local_jev_model="jev")
    capability, session, browser_provider, jev_provider = runner.build_browser_find(backend)
    assert isinstance(capability, JevBrowserCapability)
    assert isinstance(capability.executor, BrowserTaskExecutor)
    assert capability.executor.browser is session
    assert capability.executor.controller.provider is browser_provider
    assert capability.executor.completion_verifier.provider is jev_provider
