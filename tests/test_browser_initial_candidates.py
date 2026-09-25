"""Phase 4 candidate generation stays concrete and preserves the browser loop."""

from dataclasses import replace

import pytest

from ping_ponder.agentic.browser_use_slice import (
    BrowserAction, BrowserActionKind, BrowserCompletionStatus, BrowserDecision,
    BrowserObservation, BrowserTask, BrowserTaskExecutor, StaleBrowserAction,
    generate_action_candidates,
)
from ping_ponder.agentic.goals import SemanticGoal


BLANK = BrowserObservation("blank-1", "about:blank", "Blank", "", frozenset(), "blank-1")
HOME = BrowserObservation("home-2", "https://github.com/", "GitHub", "Search GitHub",
                          frozenset(), "home-2")
GOAL = SemanticGoal("Browser", "FIND", {"site": "GitHub", "target": "Browser Use repository"})


def test_phase2_control_does_not_turn_site_homepage_into_a_weak_candidate():
    candidates = generate_action_candidates(BrowserTask.from_semantic_goal(GOAL), BLANK,
                                            tuple(BrowserActionKind), native_site_jev=False)
    navigations = [candidate.action for candidate in candidates if candidate.action and
                   candidate.action.kind is BrowserActionKind.NAVIGATE]
    assert navigations == []
    assert [candidate.option for candidate in candidates] == ["ESCALATE_TO_LUNA"]
    assert "search?q=" not in repr(candidates)


def test_explicit_target_url_is_bounded_and_conflicting_site_is_not_offered():
    goal = SemanticGoal("Browser", "FIND", {"site": "https://www.iana.org",
                                             "target": "https://www.iana.org/domains/reserved"})
    candidates = generate_action_candidates(BrowserTask.from_semantic_goal(goal), BLANK,
                                            tuple(BrowserActionKind))
    assert [candidate.action.url for candidate in candidates if candidate.action] == [
        "https://www.iana.org/domains/reserved",
    ]
    conflicting = goal.with_arguments(site="GitHub")
    candidates = generate_action_candidates(BrowserTask.from_semantic_goal(conflicting), BLANK,
                                            tuple(BrowserActionKind))
    assert all(candidate.action is None or candidate.action.url != goal.argument("target")
               for candidate in candidates)


def test_unknown_site_and_human_target_escalate_without_guessing_domain():
    goal = SemanticGoal("Browser", "FIND", {"site": "Example", "target": "Blue Atlas guide"})
    candidates = generate_action_candidates(BrowserTask.from_semantic_goal(goal), BLANK,
                                            tuple(BrowserActionKind))
    assert [candidate.option for candidate in candidates] == ["ESCALATE_TO_LUNA"]


class Pages:
    def __init__(self):
        self.page = BLANK
        self.actions = []

    async def observe(self):
        return self.page

    async def act(self, action):
        if action.observation_id and action.observation_id != self.page.observation_id:
            raise StaleBrowserAction("observation changed before execution")
        self.actions.append(action)
        self.page = HOME
        return {"extracted_content": "navigated"}


class Controller:
    def __init__(self):
        self.calls = 0

    async def next_actions(self, task, observation, available_actions, memory):
        self.calls += 1
        if observation.observation_id != "blank-1":
            return BrowserDecision(completion_status="uncertain")
        return BrowserDecision(actions=[BrowserAction(kind="navigate", url="https://github.com/")])


class Chooser:
    def __init__(self, option):
        self.option = option
        self.candidates = []

    async def choose(self, task, observation, candidates):
        self.candidates.append(candidates)
        return self.option


@pytest.mark.asyncio
async def test_selected_initial_candidate_executes_once_and_regenerates_after_observation():
    browser, controller, chooser = Pages(), Controller(), Chooser("NAVIGATE_1")
    goal = SemanticGoal("Browser", "FIND", {"site": "GitHub", "target": "https://github.com/"})
    result = await BrowserTaskExecutor(browser, controller, action_chooser=chooser,
                                       initial_action_candidates=True, max_decisions=2).execute(goal)
    assert [(action.kind, action.url) for action in browser.actions] == [
        (BrowserActionKind.NAVIGATE, "https://github.com/")]
    assert controller.calls == 1
    assert len(chooser.candidates) == 2
    assert all(candidate.action is None or candidate.action.kind is not BrowserActionKind.NAVIGATE
               for candidate in chooser.candidates[1])
    assert result.timing_totals["action_candidate_generation_calls"] == 2
    assert result.timing_totals["jev_action_choice_calls"] == 2


@pytest.mark.asyncio
async def test_unknown_or_mutated_choice_escalates_and_control_disables_only_initial_generation():
    goal = SemanticGoal("Browser", "FIND", {"site": "GitHub", "target": "https://github.com/"})
    for enabled, option in [(True, "NAVIGATE_999"), (False, "NAVIGATE_1")]:
        browser, controller, chooser = Pages(), Controller(), Chooser(option)
        result = await BrowserTaskExecutor(browser, controller, action_chooser=chooser,
                                           initial_action_candidates=enabled,
                                           max_decisions=1).execute(goal)
        assert [action.url for action in browser.actions] == ["https://github.com/"]
        assert controller.calls == 1
        assert result.timing_totals["action_candidate_generation_calls"] == (1 if enabled else 0)
        assert result.timing_totals["jev_action_choice_calls"] == (1 if enabled else 0)


@pytest.mark.asyncio
async def test_candidate_navigation_rejects_stale_observation():
    browser, controller = Pages(), Controller()
    goal = SemanticGoal("Browser", "FIND", {"site": "GitHub", "target": "https://github.com/"})
    class StaleChooser:
        async def choose(self, task, observation, candidates):
            browser.page = replace(HOME, observation_id="changed-before-act")
            return "NAVIGATE_1"
    result = await BrowserTaskExecutor(browser, controller, action_chooser=StaleChooser(),
                                       initial_action_candidates=True, max_decisions=1).execute(goal)
    assert browser.actions == []
    assert result.completion.status is BrowserCompletionStatus.UNCERTAIN


@pytest.mark.asyncio
async def test_phase4_escalation_executes_only_first_luna_action_then_rebuilds_candidates():
    class BatchController(Controller):
        async def next_actions(self, task, observation, available_actions, memory):
            if observation.observation_id == "blank-1":
                self.calls += 1
                return BrowserDecision(actions=[
                    BrowserAction(kind="navigate", url="https://github.com/"),
                    BrowserAction(kind="click", index=7, observation_id="blank-1"),
                ])
            return await super().next_actions(task, observation, available_actions, memory)

    goal = SemanticGoal("Browser", "FIND", {"site": "GitHub", "target": "https://github.com/"})
    browser, controller, chooser = Pages(), BatchController(), Chooser("ESCALATE_TO_LUNA")
    result = await BrowserTaskExecutor(browser, controller, action_chooser=chooser,
                                       max_decisions=2).execute(goal)
    assert [(action.kind, action.url) for action in browser.actions] == [
        (BrowserActionKind.NAVIGATE, "https://github.com/")]
    assert result.timing_totals["jev_action_escalations"] == 2
    assert result.timing_totals["action_candidate_generation_calls"] == 2
    assert len(chooser.candidates) == 2
    assert all(candidate.action is None or candidate.action.kind is not BrowserActionKind.NAVIGATE
               for candidate in chooser.candidates[1])
