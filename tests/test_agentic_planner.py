"""Planner and executor: state-dependent plans, replanning, speculation policy."""

import pytest

from ping_ponder.agentic.adapters.memory import MemoryBrowserAdapter
from ping_ponder.agentic.capabilities import build_browser_descriptor, build_spotify_descriptor
from ping_ponder.agentic.capabilities.browser import WORLD_SCHEMA as BROWSER_WORLD_SCHEMA
from ping_ponder.agentic.executor import ExecutionLimits, ExecutionOutcome, Executor
from ping_ponder.agentic.goals import SemanticGoal
from ping_ponder.agentic.operators import ActionOperator, OperatorMetadata
from ping_ponder.agentic.planner import DeterministicPlanner, PlanningFailed
from ping_ponder.agentic.world import Compare, Condition, Effect, WorldState

SPOTIFY = build_spotify_descriptor(None)
BROWSER = build_browser_descriptor(None)
CLOSED = WorldState.from_schema(SPOTIFY.world_schema | BROWSER.world_schema)
PLAY_JAZZ = SemanticGoal("Spotify", "PLAY", {"query": "Jazz"})


def test_same_goal_plans_differently_under_different_world_states():
    planner = DeterministicPlanner()
    schema = SPOTIFY.schema("PLAY")

    closed = planner.plan(PLAY_JAZZ, schema, CLOSED, SPOTIFY.operators)
    assert [step.operator.name for step in closed.steps] == ["OpenSpotify", "SearchSpotify", "PlayResult"]

    opened = CLOSED.updated({"spotify.running": True})
    assert [step.operator.name for step in planner.plan(PLAY_JAZZ, schema, opened, SPOTIFY.operators).steps] == \
        ["SearchSpotify", "PlayResult"]

    searched = opened.updated({"spotify.search_results": "Jazz"})
    assert [step.operator.name for step in planner.plan(PLAY_JAZZ, schema, searched, SPOTIFY.operators).steps] == \
        ["PlayResult"]

    playing = searched.updated({"media.playing": True, "media.query": "Jazz",
                                "spotify.current_track": "Jazz - top result"})
    assert planner.plan(PLAY_JAZZ, schema, playing, SPOTIFY.operators).empty


def test_planner_never_invents_an_unreachable_plan():
    planner = DeterministicPlanner()
    with pytest.raises(PlanningFailed):
        planner.plan(SemanticGoal("Browser", "BACK"), BROWSER.schema("BACK"), CLOSED, BROWSER.operators)


@pytest.mark.asyncio
@pytest.mark.parametrize(("alias", "expected_url"), [
    ("YouTube", "https://www.youtube.com/"),
    ("Wikipedia", "https://www.wikipedia.org/"),
])
async def test_navigate_alias_opens_browser_and_reaches_resolved_url(alias, expected_url):
    adapter = MemoryBrowserAdapter(BROWSER_WORLD_SCHEMA)
    descriptor = build_browser_descriptor(None, adapter=adapter)
    goal = SemanticGoal("Browser", "NAVIGATE", {"target": alias})

    report = await Executor(DeterministicPlanner()).execute(
        goal, descriptor.schema("NAVIGATE"), CLOSED, descriptor,
    )

    assert report.outcome is ExecutionOutcome.SATISFIED
    assert (await adapter.observe())["browser.current_url"] == expected_url


def test_command_goals_resolve_to_their_own_operator():
    planner = DeterministicPlanner()
    history = CLOSED.updated({"browser.running": True, "browser.history": ("a", "b"),
                              "browser.history_index": 1, "browser.current_url": "b"})
    back = planner.plan(SemanticGoal("Browser", "BACK"), BROWSER.schema("BACK"), history, BROWSER.operators)
    assert [step.operator.name for step in back.steps] == ["Back"]
    forward = planner.plan(SemanticGoal("Browser", "FORWARD"), BROWSER.schema("FORWARD"),
                           history.updated({"browser.history_index": 0}), BROWSER.operators)
    assert [step.operator.name for step in forward.steps] == ["Forward"]


def test_operators_without_bound_arguments_are_skipped():
    planner = DeterministicPlanner()
    unbound = SemanticGoal("Spotify", "PLAY")
    with pytest.raises(PlanningFailed):
        planner.plan(unbound, SPOTIFY.schema("PLAY"), CLOSED, SPOTIFY.operators)


def test_speculative_closure_finds_prerequisites_not_goal_satisfiers():
    planner = DeterministicPlanner()
    steps = planner.speculative_steps(SemanticGoal("Spotify", "PLAY"), SPOTIFY.schema("PLAY"),
                                      CLOSED, SPOTIFY.operators)
    # OpenSpotify is speculative-safe and produces a prerequisite; PlayResult must not run.
    assert [step.operator.name for step in steps] == ["OpenSpotify"]


def test_speculative_closure_excludes_requires_final_operators():
    planner = DeterministicPlanner()
    steps = planner.speculative_steps(PLAY_JAZZ, SPOTIFY.schema("PLAY"), CLOSED, SPOTIFY.operators)
    assert "PlayResult" not in [step.operator.name for step in steps]


@pytest.mark.asyncio
async def test_executor_replans_when_the_world_changes_under_it():
    """A second step whose preconditions vanished is replanned, not forced."""
    calls: list[str] = []

    async def open_thing(world, args):
        calls.append("open")
        return {"thing.open": True}

    async def flaky(world, args):
        calls.append("flaky")
        raise RuntimeError("transient failure")

    async def finish(world, args):
        calls.append("finish")
        return {"thing.done": True}

    operators = (
        ActionOperator("OpenThing", (), (Condition("thing.open", Compare.FALSY),), (Effect("thing.open", True),),
                       executor=open_thing, metadata=OperatorMetadata(speculative_safe=True)),
        ActionOperator("Flaky", (), (Condition("thing.open"),), (Effect("thing.done", True),),
                       executor=flaky, metadata=OperatorMetadata(speculative_safe=False)),
        ActionOperator("Finish", (), (Condition("thing.open"),), (Effect("thing.done", True),),
                       executor=finish, metadata=OperatorMetadata(speculative_safe=False)),
    )
    from ping_ponder.agentic.goals import GoalSchema
    from ping_ponder.agentic.registry import CapabilityDescriptor

    descriptor = CapabilityDescriptor("Thing", "test", SPOTIFY.local_jev,
                                      {"DO": GoalSchema("DO", satisfied_when=(Condition("thing.done"),))},
                                      operators)
    goal = SemanticGoal("Thing", "DO")
    report = await Executor(DeterministicPlanner()).execute(
        goal, descriptor.schema("DO"), WorldState({"thing.open": False}), descriptor)
    assert report.outcome is ExecutionOutcome.SATISFIED
    assert report.replans >= 1 and "Finish" in report.executed


@pytest.mark.asyncio
async def test_incomplete_partial_search_can_speculate_open_without_query():
    adapter = MemoryBrowserAdapter(BROWSER_WORLD_SCHEMA)
    descriptor = build_browser_descriptor(None, adapter=adapter)
    report = await Executor(DeterministicPlanner()).execute(
        SemanticGoal("Browser", "SEARCH"), descriptor.schema("SEARCH"), CLOSED, descriptor,
        final=False, speculation_only=True,
    )

    assert report.executed == ("OpenBrowser",)


@pytest.mark.asyncio
async def test_executor_observes_undeclared_effects_as_a_failure():
    async def lying(world, args):
        return {"thing.done": True, "thing.sneaky": True}

    operators = (ActionOperator("Lie", (), (), (Effect("thing.done", True),), executor=lying),)
    from ping_ponder.agentic.goals import GoalSchema
    from ping_ponder.agentic.registry import CapabilityDescriptor

    descriptor = CapabilityDescriptor("Thing", "test", SPOTIFY.local_jev,
                                      {"DO": GoalSchema("DO", satisfied_when=(Condition("thing.done"),))},
                                      operators)
    report = await Executor(DeterministicPlanner()).execute(
        SemanticGoal("Thing", "DO"), descriptor.schema("DO"), WorldState(), descriptor)
    assert report.outcome is ExecutionOutcome.FAILED
    assert "undeclared effects" in report.reason


@pytest.mark.asyncio
async def test_executor_reports_blocked_when_no_operator_applies():
    report = await Executor(DeterministicPlanner()).execute(
        SemanticGoal("Browser", "BACK"), BROWSER.schema("BACK"), CLOSED, BROWSER)
    assert report.outcome is ExecutionOutcome.BLOCKED and not report.records


@pytest.mark.asyncio
async def test_executor_respects_its_step_budget():
    async def bump(world, args):
        return {"counter": int(world.get("counter") or 0) + 1}

    from ping_ponder.agentic.goals import GoalSchema
    from ping_ponder.agentic.registry import CapabilityDescriptor
    from ping_ponder.agentic.planner import PlanningLimits

    def at_least(limit):
        return lambda world: int(world.get("counter") or 0) >= limit

    operators = (ActionOperator("Bump", (), (), (Effect("counter", derive=lambda w, a: int(w.get("counter") or 0) + 1),),
                               executor=bump),)
    descriptor = CapabilityDescriptor("Thing", "test", SPOTIFY.local_jev,
                                      {"COUNT": GoalSchema("COUNT", satisfied_when=(
                                          Condition("counter", predicate=at_least(100)),))},
                                      operators)
    # The executor sizes its search from the capability, so a single-operator
    # capability gets a shallow plan and the step budget is what stops the loop.
    executor = Executor(DeterministicPlanner(), limits=ExecutionLimits(max_steps=3))
    report = await executor.execute(SemanticGoal("Thing", "COUNT"), descriptor.schema("COUNT"),
                                    WorldState({"counter": 0}), descriptor)
    assert report.outcome is ExecutionOutcome.BLOCKED
    assert len(report.records) <= 3
    assert report.world_after.get("counter") == len([r for r in report.records if r.succeeded])


@pytest.mark.asyncio
async def test_executor_replans_when_observed_effects_differ_from_projection():
    """The projection says done, the executor observes otherwise: replan, do not trust the plan."""
    async def optimistic(world, args):
        return {"counter": int(world.get("counter") or 0) + 1}

    async def real(world, args):
        return {"done": True}

    from ping_ponder.agentic.goals import GoalSchema
    from ping_ponder.agentic.registry import CapabilityDescriptor
    from ping_ponder.agentic.planner import PlanningLimits

    def done(world):
        return bool(world.get("done"))

    operators = (
        # Declares a counter effect but only ever moves it by one: the planner may
        # believe a 3-step plan reaches 3, while execution stays at 1.
        ActionOperator("Optimistic", (), (), (Effect("counter", derive=lambda w, a: int(w.get("counter") or 0) + 1),),
                       executor=optimistic),
        ActionOperator("Real", (), (), (Effect("done", True),), executor=real),
        ActionOperator("FakeSatisfy", (), (), (Effect("done", True),), executor=None),
    )
    descriptor = CapabilityDescriptor("Thing", "test", SPOTIFY.local_jev,
                                      {"GO": GoalSchema("GO", satisfied_when=(Condition("done", predicate=done),))},
                                      operators)
    executor = Executor(DeterministicPlanner(limits=PlanningLimits(max_depth=4)))
    report = await executor.execute(SemanticGoal("Thing", "GO"), descriptor.schema("GO"),
                                    WorldState({"done": False, "counter": 0}), descriptor)
    assert report.outcome is ExecutionOutcome.SATISFIED
    assert "Real" in report.executed


@pytest.mark.asyncio
async def test_executor_observes_before_planning_and_skips_satisfied_prerequisite():
    calls: list[str] = []

    async def open_thing(world, args):
        calls.append("open")
        return {}

    state = {"thing.open": True, "thing.done": False}

    async def observing():
        return dict(state)

    async def observed_finish(world, args):
        calls.append("finish")
        state["thing.done"] = True
        return {"thing.done": True}

    from ping_ponder.agentic.goals import GoalSchema
    from ping_ponder.agentic.registry import CapabilityDescriptor

    descriptor = CapabilityDescriptor(
        "Thing", "test", SPOTIFY.local_jev,
        {"DO": GoalSchema("DO", satisfied_when=(Condition("thing.done"),))},
        (
            ActionOperator("OpenThing", (), (Condition("thing.open", Compare.FALSY),),
                           (Effect("thing.open", True),), executor=open_thing),
            ActionOperator("Finish", (), (Condition("thing.open"),),
                           (Effect("thing.done", True),), executor=observed_finish),
        ),
        world_schema={"thing.open": False, "thing.done": False},
        observer=observing,
    )

    report = await Executor(DeterministicPlanner()).execute(
        SemanticGoal("Thing", "DO"), descriptor.schema("DO"),
        WorldState(descriptor.world_schema), descriptor)

    assert report.outcome is ExecutionOutcome.SATISFIED
    assert calls == ["finish"]
    assert report.world_after.get("thing.open") is True


@pytest.mark.asyncio
async def test_executor_rejects_observer_keys_outside_capability_schema():
    async def observe():
        return {"thing.open": False, "spotify.running": True}

    async def finish(world, args):
        return {"thing.done": True}

    from ping_ponder.agentic.goals import GoalSchema
    from ping_ponder.agentic.registry import CapabilityDescriptor

    descriptor = CapabilityDescriptor(
        "Thing", "test", SPOTIFY.local_jev,
        {"DO": GoalSchema("DO", satisfied_when=(Condition("thing.done"),))},
        (ActionOperator("Finish", (), (), (Effect("thing.done", True),), executor=finish),),
        world_schema={"thing.open": False, "thing.done": False},
        observer=observe,
    )
    report = await Executor(DeterministicPlanner()).execute(
        SemanticGoal("Thing", "DO"), descriptor.schema("DO"),
        WorldState(descriptor.world_schema), descriptor)

    assert report.outcome is ExecutionOutcome.FAILED
    assert not report.records
    assert "outside world schema" in (report.reason or "")


@pytest.mark.asyncio
async def test_executor_excludes_step_that_makes_no_observed_progress():
    attempts = 0

    async def unchanged():
        return {"thing.done": False}

    async def pretend(world, args):
        nonlocal attempts
        attempts += 1
        return {"thing.done": True}

    from ping_ponder.agentic.goals import GoalSchema
    from ping_ponder.agentic.registry import CapabilityDescriptor

    descriptor = CapabilityDescriptor(
        "Thing", "test", SPOTIFY.local_jev,
        {"DO": GoalSchema("DO", satisfied_when=(Condition("thing.done"),))},
        (ActionOperator("Pretend", (), (), (Effect("thing.done", True),), executor=pretend),),
        world_schema={"thing.done": False}, observer=unchanged,
    )
    report = await Executor(DeterministicPlanner()).execute(
        SemanticGoal("Thing", "DO"), descriptor.schema("DO"),
        WorldState(descriptor.world_schema), descriptor)

    assert attempts == 1
    assert report.outcome in {ExecutionOutcome.FAILED, ExecutionOutcome.BLOCKED}
    assert "no observed progress" in (report.reason or "")
