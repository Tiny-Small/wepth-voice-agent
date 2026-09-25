"""Browser.FIND wiring from natural language through the existing Jev spine."""

from __future__ import annotations

import pytest

from ping_ponder.agentic.adapters.memory import MemoryBrowserAdapter, MemorySpotifyAdapter
from ping_ponder.agentic.capabilities.browser import WORLD_SCHEMA, browser_goal_schemas
from ping_ponder.agentic.capabilities.spotify import WORLD_SCHEMA as SPOTIFY_WORLD_SCHEMA
from ping_ponder.agentic.execution_config import AdapterBundle, ExecutionSettings
from ping_ponder.agentic.span import ExtractedSpan, SlotRequest
from ping_ponder.agentic.browser_use_slice import (
    BrowserCompletionStatus,
    BrowserDecision,
    BrowserEvidence,
    BrowserObservation,
    BrowserTaskExecutor,
    JevBrowserCapability,
)
from ping_ponder.agentic.backend_config import BackendSettings
from ping_ponder.agentic.wiring import build_chat_session, build_default_registry, build_default_spine


class BrowserFindExtractor:
    """Deterministic NuExtract-shaped fixture: only requested slots are returned."""

    name = "browser-find-fixture"

    async def extract_many(self, utterance: str, requests: tuple[SlotRequest, ...]):
        values = {"site": "GitHub", "target": "Browser Use repository"}
        spans = {}
        for request in requests:
            value = values.get(request.slot)
            if value is None:
                continue
            start = utterance.index(value)
            spans[request.slot] = ExtractedSpan(
                slot=request.slot,
                question=request.question,
                text=value,
                start=start,
                end=start + len(value),
                confidence=0.99,
                extractor=self.name,
            )
        self.requests = requests
        return spans

    async def extract(self, utterance, question, *, slot, confidence_threshold=0.0):
        raise AssertionError("Browser.FIND should use the multi-slot extraction profile")


class RecordingBrowserFind:
    def __init__(self, status=BrowserCompletionStatus.SATISFIED):
        self.goals = []
        self.controller = RecordingController(status)
        class CompletionVerifier:
            async def verify(inner_self, task, observation):
                self.completion_task = task
                return status

        self.runner = JevBrowserCapability(BrowserTaskExecutor(
            GroundedPage(), self.controller, completion_verifier=CompletionVerifier(),
        ))

    async def execute(self, goal, world):
        self.goals.append(goal)
        result, updated = await self.runner.execute(goal, world)
        self.result = result
        return result, updated


class GroundedPage:
    async def observe(self):
        return BrowserObservation(
            observation_id="github-page-1",
            url="https://github.com/browser-use/browser-use",
            title="GitHub - browser-use/browser-use",
            dom="browser-use/browser-use repository",
            interactive_indices=frozenset(),
            grounding_fingerprint="github-page",
        )

    async def act(self, action):
        raise AssertionError("mock controller should finish from supplied evidence")


class RecordingController:
    def __init__(self, status=BrowserCompletionStatus.SATISFIED):
        self.status = status

    async def next_actions(self, task, observation, available_actions, memory):
        self.task = task
        return BrowserDecision(
            completion_status=self.status,
            evidence=BrowserEvidence(
                url="https://github.com/browser-use/browser-use",
                title="browser-use/browser-use",
                matched_text="Browser Use",
            ),
        )


@pytest.mark.asyncio
async def test_browser_find_routes_extracts_builds_and_hands_off_authoritative_goal():
    extractor = BrowserFindExtractor()
    browser_find = RecordingBrowserFind()
    browser = MemoryBrowserAdapter(WORLD_SCHEMA)
    registry = build_default_registry(
        adapters=AdapterBundle(
            spotify=MemorySpotifyAdapter(SPOTIFY_WORLD_SCHEMA), browser=browser,
        ),
        browser_find=browser_find,
    )
    service = build_default_spine(extractor=extractor, registry=registry)

    outcome = await service.resolve_final("Find the Browser Use repository on GitHub.")

    assert outcome.global_capability == "Browser"
    assert outcome.local_goal_type == "FIND"
    assert outcome.goal.describe() == "Browser.FIND(site='GitHub', target='Browser Use repository')"
    assert [request.slot for request in extractor.requests] == ["site", "target"]
    assert outcome.report is not None and outcome.report.satisfied
    assert [goal.describe() for goal in browser_find.goals] == [outcome.goal.describe()]
    assert browser_find.completion_task.semantic_goal.argument("site") == "GitHub"
    assert browser_find.completion_task.target == "Browser Use repository"
    assert browser_find.result.completion.status is BrowserCompletionStatus.SATISFIED
    assert service.world.get("browser.last_task_evidence") == {
        "url": "https://github.com/browser-use/browser-use",
        "title": "GitHub - browser-use/browser-use",
        "matched_text": "GitHub - browser-use/browser-use",
    }
    assert "browser.dom" not in service.world.as_dict()
    assert "browser.selector_map" not in service.world.as_dict()


@pytest.mark.asyncio
async def test_browser_controller_blocked_result_is_not_global_goal_satisfaction():
    browser_find = RecordingBrowserFind(BrowserCompletionStatus.BLOCKED)
    registry = build_default_registry(
        adapters=AdapterBundle(
            spotify=MemorySpotifyAdapter(SPOTIFY_WORLD_SCHEMA),
            browser=MemoryBrowserAdapter(WORLD_SCHEMA),
        ),
        browser_find=browser_find,
    )
    service = build_default_spine(registry=registry)

    outcome = await service.resolve_final("Find the Browser Use repository on GitHub.")

    assert outcome.report is not None
    assert not outcome.report.satisfied
    assert outcome.report.outcome.value == "BLOCKED", outcome.report.reason
    assert service.world.get("browser.last_task_status") == "blocked"
    assert service.world.get("browser.last_task_evidence")["matched_text"] == "Browser Use"


def test_browser_find_schema_is_the_minimal_extraction_profile():
    schema = browser_goal_schemas()["FIND"]

    assert [(slot.name, slot.required) for slot in schema.slots] == [
        ("site", False),
        ("target", True),
    ]


def test_chat_session_accepts_the_existing_level_two_browser_capability():
    browser_find = RecordingBrowserFind()

    session = build_chat_session(
        settings=BackendSettings(),
        execution=ExecutionSettings(backend="simulated"),
        browser_find=browser_find,
    )

    assert "FindWithGroundedBrowser" in session.spine.registry.get("Browser").operator_names()


@pytest.mark.asyncio
async def test_existing_browser_navigate_stays_on_deterministic_operator_path():
    browser_find = RecordingBrowserFind()
    browser = MemoryBrowserAdapter(WORLD_SCHEMA)
    registry = build_default_registry(
        adapters=AdapterBundle(
            spotify=MemorySpotifyAdapter(SPOTIFY_WORLD_SCHEMA), browser=browser,
        ),
        browser_find=browser_find,
    )
    service = build_default_spine(registry=registry)

    outcome = await service.resolve_final("Go to github.com.")

    assert outcome.global_capability == "Browser"
    assert outcome.local_goal_type == "NAVIGATE"
    assert outcome.report is not None
    assert outcome.report.executed[-1] == "NavigateURL"
    assert "FindWithGroundedBrowser" not in outcome.report.executed
    assert browser_find.goals == []
