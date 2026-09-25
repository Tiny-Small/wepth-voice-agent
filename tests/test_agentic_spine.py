"""End-to-end spine: vertical slice, partial speculation, arbitration, concurrency."""

import asyncio

import pytest

from ping_ponder.agentic.jev import GlobalRoute, LocalRoute
from ping_ponder.agentic.registry import CapabilityDescriptor
from ping_ponder.agentic.spine import VoiceActionSpine
from ping_ponder.agentic.wiring import build_default_spine


def spine() -> VoiceActionSpine:
    return build_default_spine()


@pytest.mark.asyncio
async def test_vertical_slice_play_some_jazz():
    """"Play some Jazz" -> Spotify -> PLAY -> "Jazz" -> Open -> Search -> Play."""
    service = spine()
    outcome = await service.resolve_final("Play some Jazz")

    assert outcome.global_capability == "Spotify"
    assert outcome.local_goal_type == "PLAY"
    assert outcome.goal.describe() == "Spotify.PLAY(query='Jazz')"
    assert outcome.report.executed == ("OpenSpotify", "SearchSpotify", "PlayResult")
    assert outcome.satisfied
    assert service.world.get("spotify.running") is True
    assert service.world.get("media.playing") is True
    assert service.world.get("spotify.current_track") == "Jazz - top result"


@pytest.mark.asyncio
async def test_semantic_goal_is_identical_regardless_of_world_state():
    """Jev must infer PLAY(query=Jazz) whether or not Spotify is already open."""
    closed = spine()
    first = await closed.resolve_final("Play some Jazz")

    opened = spine()
    await opened.resolve_final("Play some Jazz")  # Spotify is now running and has results
    second = await opened.resolve_final("Play some Jazz")

    assert first.goal.describe() == second.goal.describe() == "Spotify.PLAY(query='Jazz')"
    # Only the plan differs.
    assert first.report.executed == ("OpenSpotify", "SearchSpotify", "PlayResult")
    assert second.report.executed == ()


@pytest.mark.asyncio
async def test_planner_opens_prerequisite_but_goal_stays_play_not_open():
    service = spine()
    outcome = await service.resolve_final("Play some Jazz")
    assert outcome.local_goal_type == "PLAY"
    assert outcome.local_goal_type != "OPEN"
    assert "OpenSpotify" in outcome.report.executed  # a prerequisite, not the goal


@pytest.mark.asyncio
async def test_active_local_is_set_and_used_as_a_prior():
    service = spine()
    await service.resolve_final("Play some Jazz")
    assert service.active_local == "Spotify"

    # "pause it" has no capability keyword; the active local resolves it.
    outcome = await service.resolve_final("Pause it")
    assert outcome.global_capability == "Spotify"
    assert outcome.goal.describe() == "Spotify.PAUSE()"
    assert outcome.report.executed == ("PauseSpotify",)
    assert service.world.get("media.playing") is False


@pytest.mark.asyncio
async def test_active_local_goal_survives_global_none_route():
    """An ambiguous global route must not drop a valid active-local follow-up."""

    class NoCapabilityGlobal:
        async def route(self, utterance, *, active_local=None, context=None):
            return GlobalRoute(capability=None, confidence=0.2)

    service = spine()
    await service.resolve_final("Play some Jazz")
    service.global_jev = NoCapabilityGlobal()

    outcome = await service.resolve_final("Stop it")

    assert outcome.global_capability == "Spotify"
    assert outcome.local_goal_type == "PAUSE"
    assert outcome.goal.describe() == "Spotify.PAUSE()"
    assert outcome.report.executed == ("PauseSpotify",)
    assert outcome.discarded_speculation is None


@pytest.mark.asyncio
async def test_global_jev_can_switch_away_from_active_local():
    service = spine()
    await service.resolve_final("Play some Jazz")
    assert service.active_local == "Spotify"

    outcome = await service.resolve_final("Search Google for Miles Davis")
    assert outcome.global_capability == "Browser"
    assert outcome.goal.describe() == "Browser.SEARCH(query='Miles Davis')"
    assert outcome.discarded_speculation == "switched_to_Browser"
    assert service.active_local == "Browser"


@pytest.mark.asyncio
async def test_partial_speculation_opens_prerequisite_without_authoritative_goal():
    service = spine()
    partial = await service.observe_partial("Open Spotify and play some")

    assert partial.accepted and partial.acted
    assert partial.hypothesis.is_final is False
    assert partial.hypothesis.goal_type == "PLAY"
    # Only the speculative-safe prerequisite ran.
    assert partial.speculative_report.executed == ("OpenSpotify",)
    assert partial.speculative_report.speculative
    assert not partial.speculative_report.satisfied
    # Speculation is not authoritative: no active_local, no goal committed.
    assert service.active_local is None
    assert service.world.get("spotify.running") is True

    final = await service.resolve_final("Open Spotify and play Jazz")
    assert final.goal.describe() == "Spotify.PLAY(query='Jazz')"
    # The planner benefits from the speculative prerequisite already being satisfied.
    assert final.report.executed == ("SearchSpotify", "PlayResult")
    assert final.preserved_partial_speculation


@pytest.mark.asyncio
async def test_partial_never_runs_a_requires_final_operator():
    service = spine()
    # Spotify already open, so the only ready operator is the non-speculative PlayResult.
    service.observe_world({"spotify.running": True, "spotify.focused": True})
    partial = await service.observe_partial("play some Jazz")
    assert partial.acted
    executed = partial.speculative_report.executed
    assert "PlayResult" not in executed
    assert service.world.get("media.playing") is False


@pytest.mark.asyncio
async def test_partial_capability_mismatch_is_discarded_not_executed():
    service = spine()
    await service.resolve_final("Play some Jazz")
    service.observe_world({"media.playing": False})
    partial = await service.observe_partial("search Google for tour dates")
    assert not partial.accepted
    assert partial.discard_reason == "capability_not_active"
    assert partial.global_capability == "Browser"
    assert not partial.acted


@pytest.mark.asyncio
async def test_world_state_never_reaches_jev():
    """The core contamination guard: Jev must not see execution state."""
    seen: list[dict] = []

    class RecordingGlobal:
        async def route(self, utterance, *, active_local=None, context=None):
            seen.append(dict(context or {}))
            return GlobalRoute(capability="Spotify", confidence=0.9)

    class RecordingLocal:
        async def interpret(self, utterance, *, descriptor):
            return LocalRoute(capability=descriptor.name, goal_type="PLAY", confidence=0.9)

    service = spine()
    service.global_jev = RecordingGlobal()
    descriptor = service.registry.get("Spotify")
    service.registry.register(CapabilityDescriptor(descriptor.name, descriptor.description,
                                                   RecordingLocal(), descriptor.goal_schemas,
                                                   descriptor.operators, descriptor.world_schema))
    # A hostile world: Spotify closed, nothing playing.
    service.observe_world({"spotify.running": False, "media.playing": False})
    outcome = await service.resolve_final("Play some Jazz")

    assert outcome.local_goal_type == "PLAY"
    assert outcome.goal.describe() == "Spotify.PLAY(query='Jazz')"
    assert all(not any(k.startswith(("spotify.", "media.", "browser.")) for k in ctx) for ctx in seen)


@pytest.mark.asyncio
async def test_concurrent_global_and_active_local_overlap_instead_of_serialize():
    global_delay = local_delay = 0.10

    class SlowGlobal:
        async def route(self, utterance, *, active_local=None, context=None):
            await asyncio.sleep(global_delay)
            return GlobalRoute(capability=active_local, confidence=0.9)

    class SlowLocal:
        async def interpret(self, utterance, *, descriptor):
            await asyncio.sleep(local_delay)
            return LocalRoute(capability=descriptor.name, goal_type="PAUSE", confidence=0.9)

    service = spine()
    await service.resolve_final("Play some Jazz")
    service.global_jev = SlowGlobal()
    descriptor = service.registry.get("Spotify")
    service.registry.register(CapabilityDescriptor(descriptor.name, descriptor.description,
                                                   SlowLocal(), descriptor.goal_schemas,
                                                   descriptor.operators, descriptor.world_schema))

    started = asyncio.get_running_loop().time()
    outcome = await service.resolve_final("Pause it")
    elapsed = asyncio.get_running_loop().time() - started

    assert outcome.goal.describe() == "Spotify.PAUSE()"
    # max(T_global, T_local), not T_global + T_local.
    assert elapsed < global_delay + local_delay
    assert outcome.serial_equivalent_latency > outcome.effective_semantic_latency
    assert outcome.latency_saved > 0


@pytest.mark.asyncio
async def test_partial_semantics_overlap_global_and_active_local():
    """Partial inference keeps the same max(global, local) active-context latency."""
    delay = 0.10

    class SlowGlobal:
        async def route(self, utterance, *, active_local=None, context=None):
            await asyncio.sleep(delay)
            return GlobalRoute(capability=active_local, confidence=0.9)

    class SlowLocal:
        async def interpret(self, utterance, *, descriptor):
            await asyncio.sleep(delay)
            return LocalRoute(capability=descriptor.name, goal_type="PAUSE", confidence=0.9)

    service = spine()
    await service.resolve_final("Play some Jazz")
    service.global_jev = SlowGlobal()
    descriptor = service.registry.get("Spotify")
    service.registry.register(CapabilityDescriptor(
        descriptor.name, descriptor.description, SlowLocal(), descriptor.goal_schemas,
        descriptor.operators, descriptor.world_schema,
    ))

    started = asyncio.get_running_loop().time()
    evaluation = await service.evaluate_semantics("Pause it", final=False)
    elapsed = asyncio.get_running_loop().time() - started

    assert evaluation.local_goal_type == "PAUSE"
    assert elapsed < delay * 1.8
    assert evaluation.serial_equivalent_latency > evaluation.effective_semantic_latency


@pytest.mark.asyncio
async def test_active_local_extraction_starts_before_global_arbitration_finishes():
    global_started = asyncio.Event()
    release_global = asyncio.Event()
    extraction_completed = asyncio.Event()

    class GatedGlobal:
        async def route(self, utterance, *, active_local=None, context=None):
            global_started.set()
            await release_global.wait()
            return GlobalRoute(capability=active_local, confidence=0.9)

    class RecordingExtractor:
        name = "gated-test-extractor"

        async def extract_many(self, utterance, requests):
            spans = {}
            values = {"site": "GitHub", "target": "Browser Use repository"}
            for request in requests:
                text = values[request.slot]
                start = utterance.index(text)
                spans[request.slot] = ExtractedSpan(
                    request.slot, request.question, text, start, start + len(text), 0.99, self.name)
            extraction_completed.set()
            return spans

        async def extract(self, utterance, question, *, slot, confidence_threshold=0.0):
            raise AssertionError("the active Local path should use one multi-slot extraction")

    from ping_ponder.agentic.span import ExtractedSpan
    from ping_ponder.agentic.goal_builder import GoalBuilder

    service = spine()
    service.active_local = "Browser"
    service.global_jev = GatedGlobal()
    service.goal_builder = GoalBuilder(RecordingExtractor())

    evaluation_task = asyncio.create_task(
        service.evaluate_semantics("Find the Browser Use repository on GitHub."))
    await global_started.wait()
    await asyncio.wait_for(extraction_completed.wait(), timeout=0.5)
    assert not release_global.is_set()

    release_global.set()
    evaluation = await evaluation_task

    assert evaluation.capability == "Browser"
    assert evaluation.local_goal_type == "FIND"
    assert evaluation.build.goal.argument("site") == "GitHub"
    assert evaluation.build.goal.argument("target") == "Browser Use repository"


@pytest.mark.asyncio
async def test_semantic_evaluation_reports_local_and_extraction_stage_timings():
    service = spine()
    service.active_local = "Browser"
    from ping_ponder.agentic.goal_builder import GoalBuilder
    from ping_ponder.agentic.span import ExtractedSpan

    class SlowMultiSlotExtractor:
        name = "slow-multislot-fixture"

        async def extract_many(self, utterance, requests):
            await asyncio.sleep(0.02)
            values = {"site": "GitHub", "target": "Browser Use repository"}
            result = {}
            for request in requests:
                text = values[request.slot]
                start = utterance.index(text)
                result[request.slot] = ExtractedSpan(
                    request.slot, request.question, text, start, start + len(text), 0.99, self.name)
            return result

        async def extract(self, *args, **kwargs):
            raise AssertionError("multi-slot extraction expected")

    service.goal_builder = GoalBuilder(SlowMultiSlotExtractor())
    evaluation = await service.evaluate_semantics(
        "Find the Browser Use repository on GitHub.", final=True)

    assert evaluation.timings["global_s"] >= 0
    assert evaluation.timings["local_s"] >= 0
    assert evaluation.timings["extraction_s"] >= 0
    assert evaluation.timings["extraction_s"] >= 0.02
    assert evaluation.timings["goal_builder_s"] < evaluation.timings["extraction_s"]
    assert evaluation.timings["goal_build_wall_s"] >= evaluation.timings["extraction_s"]
    assert evaluation.timings["semantic_total_s"] >= 0


@pytest.mark.asyncio
async def test_semantic_evaluation_does_not_mutate_world_until_applied():
    service = spine()

    evaluation = await service.evaluate_semantics(
        "Open Spotify and play some", final=False,
    )

    assert service.world.get("spotify.running") is False
    assert service.active_local is None

    outcome = await service.apply_partial(evaluation)

    assert outcome.speculative_report.executed == ("OpenSpotify",)
    assert service.world.get("spotify.running") is True


@pytest.mark.asyncio
async def test_active_local_does_not_flap_on_speculative_local_failure():
    class BrokenLocal:
        async def interpret(self, utterance, *, descriptor):
            raise RuntimeError("local model unreachable")

    service = spine()
    await service.resolve_final("Play some Jazz")
    assert service.active_local == "Spotify"
    descriptor = service.registry.get("Spotify")
    service.registry.register(CapabilityDescriptor(descriptor.name, descriptor.description,
                                                   BrokenLocal(), descriptor.goal_schemas,
                                                   descriptor.operators, descriptor.world_schema))
    outcome = await service.resolve_final("Pause it")
    assert outcome.discarded_speculation == "local_failed"
    assert outcome.report is None  # no plan from a broken Local Jev
    assert service.stats.speculative_discarded >= 1


@pytest.mark.asyncio
async def test_incomplete_final_goal_is_not_planned():
    service = spine()
    outcome = await service.resolve_final("Play some")
    assert outcome.goal is not None and not outcome.build.complete
    assert outcome.build.missing_slots == ("query",)
    assert outcome.report is None
    assert service.world.get("media.playing") is False


@pytest.mark.asyncio
async def test_browser_capability_plans_without_touching_global_jev():
    service = spine()
    outcome = await service.resolve_final("Search Google for PersonaPlex turn detection")
    assert outcome.goal.describe() == "Browser.SEARCH(query='PersonaPlex turn detection')"
    assert outcome.report.executed == ("OpenBrowser", "SearchWeb")


@pytest.mark.asyncio
async def test_none_capability_produces_no_goal():
    service = spine()
    outcome = await service.resolve_final("thanks, that's all for now")
    assert outcome.global_capability is None
    assert outcome.goal is None and outcome.report is None
    assert service.active_local is None


@pytest.mark.asyncio
async def test_switch_still_routes_when_active_local_jev_is_broken():
    """A broken active Local Jev must not block routing to a different capability."""

    class BrokenLocal:
        async def interpret(self, utterance, *, descriptor):
            raise RuntimeError("local model unreachable")

    service = spine()
    await service.resolve_final("Play some Jazz")
    descriptor = service.registry.get("Spotify")
    service.registry.register(CapabilityDescriptor(descriptor.name, descriptor.description,
                                                   BrokenLocal(), descriptor.goal_schemas,
                                                   descriptor.operators, descriptor.world_schema))
    outcome = await service.resolve_final("Search Google for Miles Davis")
    assert outcome.global_capability == "Browser"
    assert outcome.goal.describe() == "Browser.SEARCH(query='Miles Davis')"
    assert outcome.report.executed == ("OpenBrowser", "SearchWeb")
    assert service.active_local == "Browser"
