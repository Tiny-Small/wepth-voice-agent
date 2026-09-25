"""Regressions from the reported chat session: extraction, skip, and play-after-search."""

import pytest

from ping_ponder.agentic.capabilities.spotify import build_spotify_descriptor
from ping_ponder.agentic.goals import SemanticGoal
from ping_ponder.agentic.planner import DeterministicPlanner, PlanningFailed
from ping_ponder.agentic.span import HeuristicSpanExtractor, NoAnswer
from ping_ponder.agentic.world import WorldState
from ping_ponder.agentic.wiring import build_chat_session

SPOTIFY = build_spotify_descriptor(None)


def world(**overrides):
    """World state builder. Overrides are world keys, so they must be qualified
    (`spotify.running`); an unqualified name would create an unused extra key and the
    predicate under test would silently read the default."""
    unknown = sorted(key for key in overrides if "." not in key)
    if unknown:
        raise ValueError(f"world keys must be qualified (e.g. spotify.running): {unknown}")
    return WorldState.from_schema(SPOTIFY.world_schema).updated(overrides)


def plan(goal_type, arguments=None, **overrides):
    goal = SemanticGoal("Spotify", goal_type, arguments or {})
    return DeterministicPlanner().plan(goal, SPOTIFY.schema(goal_type), world(**overrides),
                                       SPOTIFY.operators, limits=SPOTIFY.planning_limits())


# --- extraction: "search <target> for <query>" ---------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize("utterance,expected", [
    ("Search Spotify for Miles Davis", "Miles Davis"),
    ("Search Amazon for shoes", "shoes"),
    ("Search for Miles Davis", "Miles Davis"),
    ("search the web for cheap flights to Tokyo", "cheap flights to Tokyo"),
])
async def test_search_target_is_stripped_from_the_query(utterance, expected):
    """"Search Spotify for X" used to return the whole utterance as the query."""
    question = SPOTIFY.schema("SEARCH").slot("query").question
    span = await HeuristicSpanExtractor().extract(utterance, question, slot="query")
    assert span.text == expected
    assert utterance[span.start:span.end] == expected


@pytest.mark.asyncio
async def test_word_clauses_are_kept_but_numeric_clauses_are_not():
    """The old trailing-clause rule ate real query text; it now only drops amounts."""
    question = "What music does the user want to search for?"
    span = await HeuristicSpanExtractor().extract("play music for running", question, slot="query")
    assert span.text == "music for running"


# --- skip: declared effects and the playback presupposition --------------------------

def test_skip_declares_every_effect_it_observes():
    """SkipSpotify writes three keys; declaring one made the consistency guard fail it."""
    operator = SPOTIFY.operator("SkipSpotify")
    assert {effect.key for effect in operator.effects} == {
        "spotify.skipped", "media.playing", "spotify.paused"}


def test_skip_is_plannable_when_a_track_is_loaded():
    steps = [step.operator.name for step in
             plan("SKIP", **{"spotify.running": True,
                             "spotify.current_track": "Jazz - top result"}).steps]
    assert steps == ["SkipSpotify"]


def test_skip_presupposes_playback_and_cannot_start_it():
    """The planner must not satisfy SKIP by first starting playback."""
    with pytest.raises(PlanningFailed) as error:
        plan("SKIP", **{"spotify.running": True, "spotify.search_results": "Miles Davis"})
    assert "presupposes" in str(error.value)


# --- play after search ---------------------------------------------------------------

def test_play_with_a_query_does_not_reuse_unrelated_results():
    """PLAY(query='Jazz') while Miles Davis results are showing must re-search."""
    steps = [step.operator.name for step in
             plan("PLAY", {"query": "Jazz"}, **{"spotify.running": True,
                                                   "spotify.search_results": "Miles Davis"}).steps]
    assert steps == ["SearchSpotify", "PlayResult"]


def test_play_with_a_query_is_satisfied_by_matching_results():
    steps = [step.operator.name for step in
             plan("PLAY", {"query": "Jazz"}, **{"spotify.running": True, "spotify.search_results": "Jazz"}).steps]
    assert steps == ["PlayResult"]


def test_play_query_requires_matching_media_provenance():
    steps = [step.operator.name for step in
             plan("PLAY", {"query": "Jazz"}, **{"spotify.running": True,
                                                   "spotify.search_results": "Jazz",
                                                   "media.playing": True,
                                                   "spotify.current_track": "Jazz - top result",
                                                   "media.query": "Miles Davis"}).steps]
    assert steps == ["PlayResult"]


def test_spotify_open_is_a_semantic_goal():
    descriptor = build_spotify_descriptor(None)
    assert "OPEN" in descriptor.goal_schemas
    assert [step.operator.name for step in
            plan("OPEN", **{"spotify.running": False}).steps] == ["OpenSpotify"]


def test_play_result_requires_an_antecedent():
    with pytest.raises(PlanningFailed) as error:
        plan("PLAY_RESULT", **{"spotify.running": True})
    assert "presupposes" in str(error.value)


def test_play_result_plays_the_rooted_results():
    assert [step.operator.name for step in
            plan("PLAY_RESULT", **{"spotify.running": True,
                                                        "spotify.search_results": "Miles Davis"}).steps] == ["PlayResult"]


def test_play_query_is_required_so_a_truncated_utterance_produces_no_goal():
    """"Play some" has no referent. It must not act on whatever is on screen."""
    assert SPOTIFY.schema("PLAY").slot("query").required is True


# --- end-to-end over the chat session ------------------------------------------------

@pytest.mark.asyncio
async def test_reported_session_search_then_skip_explains_itself():
    service = build_chat_session()
    await service.say("Search Spotify for Miles Davis")
    trace = await service.say("Skip this track")
    assert trace.goal == "Spotify.SKIP()"
    assert trace.outcome == "BLOCKED"
    # The reason must name the missing precondition, not just a search bound.
    assert "presupposes" in trace.reason and "spotify.current_track" in trace.reason


@pytest.mark.asyncio
async def test_play_it_after_a_search_plays_the_results():
    service = build_chat_session()
    await service.say("Search Spotify for Miles Davis")
    trace = await service.say("Play it")
    assert trace.goal == "Spotify.PLAY_RESULT()"
    assert trace.outcome == "SATISFIED"
    assert service.spine.world.get("spotify.current_track") == "Miles Davis - top result"


@pytest.mark.asyncio
async def test_skip_after_playing_a_track_succeeds():
    service = build_chat_session()
    await service.say("Play some Jazz")
    trace = await service.say("Skip this track")
    assert trace.outcome == "SATISFIED"
    assert [step["operator"] for step in trace.as_dict()["executed"]] == ["SkipSpotify"]


@pytest.mark.asyncio
async def test_each_skip_utterance_executes_exactly_one_skip():
    service = build_chat_session()
    await service.say("Play some Jazz")
    first = await service.say("Skip this track")
    second = await service.say("Skip this track")
    assert [step["operator"] for step in first.as_dict()["executed"]] == ["SkipSpotify"]
    assert [step["operator"] for step in second.as_dict()["executed"]] == ["SkipSpotify"]


@pytest.mark.asyncio
async def test_truncated_play_does_not_act_on_stale_results():
    service = build_chat_session()
    await service.say("Search Spotify for Miles Davis")
    trace = await service.say("Play some")
    assert trace.outcome is None
    assert service.spine.world.get("spotify.current_track") is None
