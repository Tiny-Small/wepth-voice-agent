"""Native-site choices use fresh, fixed actions from the semantic goal."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from ping_ponder.agentic.browser_use_slice import (
    BrowserAction, BrowserActionKind, BrowserCompletionStatus, BrowserDecision,
    BrowserElement, BrowserObservation, BrowserTask, BrowserTaskExecutor,
    BrowserUseSessionAdapter, StaleBrowserAction, generate_action_candidates,
)
from ping_ponder.agentic.goals import SemanticGoal


GOAL = SemanticGoal("Browser", "FIND", {"site": "GitHub", "target": "Browser Use repository"})
BLANK = BrowserObservation("blank", "about:blank", "Blank", "", frozenset(), "blank")
HOME = BrowserObservation("home", "https://github.com/", "GitHub", "Search", frozenset({4, 5}), "home", elements=(
    BrowserElement(4, "button", "Search GitHub"),
    BrowserElement(5, "input", "Search GitHub", role="searchbox", input_type="text", placeholder="Search GitHub"),
))
TYPED = replace(HOME, observation_id="typed", grounding_fingerprint="typed", elements=(
    BrowserElement(4, "button", "Search GitHub"),
    BrowserElement(5, "input", "Browser Use repository", role="searchbox", input_type="text",
                   placeholder="Search GitHub", value="Browser Use repository"),
))


def options(page, *, native=True, previous=None, goal=GOAL):
    return generate_action_candidates(BrowserTask.from_semantic_goal(goal), page,
                                      tuple(BrowserActionKind), native_site_jev=native,
                                      previous_action=previous)


def test_trusted_homepage_is_fixed_and_control_restores_prior_candidates():
    candidates = options(BLANK)
    assert candidates[0].option == "NAVIGATE_1"
    assert candidates[0].action.url == "https://github.com/"
    assert candidates[0].action.observation_id == "blank"
    assert [item.option for item in options(BLANK, native=False)] == ["ESCALATE_TO_LUNA"]
    assert [item.option for item in options(BLANK, goal=GOAL.with_arguments(site="Unknown Site"))] == [
        "ESCALATE_TO_LUNA"]


def test_viewport_scroll_candidates_are_bound_to_current_observation():
    scrolls = [item.action for item in options(HOME) if item.action and
               item.action.kind is BrowserActionKind.SCROLL]
    assert scrolls and all(action.observation_id == HOME.observation_id for action in scrolls)


def test_search_input_is_fixed_to_goal_and_provenance():
    candidates = options(HOME)
    selected = next(item.action for item in candidates if item.option == "INPUT_5")
    assert (selected.index, selected.text, selected.observation_id) == (5, "Browser Use repository", "home")
    assert "INPUT_5" not in [item.option for item in options(HOME, native=False)]
    assert "INPUT_5" not in [item.option for item in options(TYPED)]
    assert "SEND_KEYS_ENTER" in [item.option for item in options(TYPED, previous=selected)]
    assert "SEND_KEYS_ENTER" not in [item.option for item in options(TYPED)]
    with pytest.raises(ValueError, match="Enter"):
        BrowserAction(kind="send_keys", observation_id="typed", key="Escape")


def test_nonsearch_or_sensitive_editables_are_never_bounded_inputs():
    page = replace(HOME, elements=(
        BrowserElement(4, "input", "Password", input_type="password", placeholder="Search password"),
        BrowserElement(5, "input", "First name", input_type="text", placeholder="First name"),
    ))
    assert not any(item.option.startswith("INPUT_") for item in options(page))


def test_native_input_is_only_offered_on_the_resolved_site():
    unrelated = replace(HOME, url="https://unrelated.example/search")
    assert "INPUT_5" not in [item.option for item in options(unrelated)]


def test_input_does_not_require_navigate_to_be_available():
    candidates = generate_action_candidates(BrowserTask.from_semantic_goal(GOAL), HOME,
                                            (BrowserActionKind.INPUT, BrowserActionKind.CLICK))
    assert "INPUT_5" in [item.option for item in candidates]


def test_trusted_site_subdomain_does_not_offer_homepage_again():
    goal = GOAL.with_arguments(site="Wikipedia", target="Python programming language article")
    page = replace(HOME, url="https://en.wikipedia.org/wiki/Python_(programming_language)")
    assert not any(item.action and item.action.kind is BrowserActionKind.NAVIGATE
                   for item in options(page, goal=goal))


def test_search_control_survives_a_link_heavy_homepage():
    elements = tuple(BrowserElement(index, "a", f"Unrelated link {index}") for index in range(10, 40))
    page = replace(HOME, elements=HOME.elements + elements,
                   interactive_indices=frozenset({4, 5, *range(10, 40)}))
    assert "CLICK_4" in [item.option for item in options(page)]
    assert "INPUT_5" in [item.option for item in options(page)]


@pytest.mark.asyncio
async def test_adapter_projects_search_attributes_from_selector_map():
    node = SimpleNamespace(index=5, tag_name="input", session_id="session", backend_node_id=1,
                           attributes={"type": "search", "placeholder": "Search this site", "name": "q", "value": ""},
                           get_meaningful_text_for_llm=lambda: "Search this site")
    state = SimpleNamespace(url="https://github.com/", title="GitHub", tabs=(),
                            dom_state=SimpleNamespace(selector_map={5: node},
                                                      llm_representation=lambda: "[5]<input>"))

    class Session:
        agent_focus_target_id = "tab"

        async def start(self):
            pass

        async def get_browser_state_summary(self, **kwargs):
            return state

    adapter = BrowserUseSessionAdapter(browser_session=Session(), tools=SimpleNamespace())
    observation = await adapter.observe()
    assert observation.elements[0].input_type == "search"
    assert observation.elements[0].placeholder == "Search this site"
    assert observation.elements[0].name == "q"
    assert next(item.action for item in options(observation) if item.option == "INPUT_5").text == GOAL.argument("target")


class Pages:
    def __init__(self):
        self.page = BLANK
        self.actions = []

    async def observe(self):
        return self.page

    async def act(self, action):
        if action.observation_id != self.page.observation_id:
            raise StaleBrowserAction("stale candidate")
        self.actions.append(action)
        if action.kind is BrowserActionKind.NAVIGATE:
            self.page = HOME
        elif action.kind is BrowserActionKind.INPUT:
            self.page = TYPED
        elif action.kind is BrowserActionKind.SEND_KEYS:
            self.page = replace(TYPED, observation_id="results", url="https://github.com/search",
                                title="Search results")
        return {"extracted_content": action.kind.value}


class Chooser:
    def __init__(self, *options):
        self.options = iter(options)
        self.seen = []

    async def choose(self, task, observation, candidates):
        self.seen.append((observation.observation_id, [item.option for item in candidates]))
        return next(self.options)


class Controller:
    def __init__(self):
        self.calls = 0

    async def next_actions(self, task, observation, available_actions, memory):
        self.calls += 1
        return BrowserDecision(completion_status="uncertain")


@pytest.mark.asyncio
async def test_native_choices_execute_one_action_and_rebuild_after_each_observation():
    browser, chooser, controller = Pages(), Chooser("NAVIGATE_1", "INPUT_5", "SEND_KEYS_ENTER"), Controller()
    result = await BrowserTaskExecutor(browser, controller, action_chooser=chooser,
                                       max_decisions=3).execute(GOAL)
    assert [action.kind for action in browser.actions] == [BrowserActionKind.NAVIGATE,
                                                           BrowserActionKind.INPUT, BrowserActionKind.SEND_KEYS]
    assert [action.observation_id for action in browser.actions] == ["blank", "home", "typed"]
    assert [seen[0] for seen in chooser.seen] == ["blank", "home", "typed"]
    assert result.timing_totals["action_candidate_generation_calls"] == 3
    assert controller.calls == 0


@pytest.mark.asyncio
async def test_choice_identifier_cannot_mutate_url_or_input_text():
    browser, controller = Pages(), Controller()
    result = await BrowserTaskExecutor(browser, controller, action_chooser=Chooser("NAVIGATE_1:https://evil.test"),
                                       max_decisions=1).execute(GOAL)
    assert browser.actions == []
    assert result.timing_totals["jev_action_escalations"] == 1
    browser.page = HOME
    result = await BrowserTaskExecutor(browser, controller, action_chooser=Chooser("INPUT_5:other text"),
                                       max_decisions=1).execute(GOAL)
    assert browser.actions == []
    assert result.completion.status is BrowserCompletionStatus.UNCERTAIN


@pytest.mark.asyncio
async def test_stale_input_candidate_is_rejected_before_execution():
    browser, controller = Pages(), Controller()
    browser.page = HOME

    class ReplacingChooser:
        async def choose(self, task, observation, candidates):
            browser.page = replace(HOME, observation_id="replacement")
            return "INPUT_5"

    result = await BrowserTaskExecutor(browser, controller, action_chooser=ReplacingChooser(),
                                       max_decisions=1).execute(GOAL)
    assert browser.actions == []
    assert result.completion.status is BrowserCompletionStatus.UNCERTAIN


@pytest.mark.asyncio
async def test_adapter_maps_fixed_grounded_input_to_browser_use_tool():
    class Session:
        async def start(self):
            pass

    class Registry:
        def __init__(self):
            self.call = None

        async def execute_action(self, **kwargs):
            self.call = kwargs
            return {"extracted_content": "typed"}

    registry = Registry()
    adapter = BrowserUseSessionAdapter(browser_session=Session(), tools=SimpleNamespace(registry=registry))
    await adapter.start()
    adapter._last_observation = HOME
    adapter._observation_ready = True
    action = next(item.action for item in options(HOME) if item.option == "INPUT_5")
    await adapter.act(action)
    assert registry.call["action_name"] == "input"
    assert registry.call["params"] == {"index": 5, "text": "Browser Use repository", "clear": True}
