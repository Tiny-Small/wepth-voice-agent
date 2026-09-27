"""Spotify capability: Local Jev goals, goal schemas, operators, and world effects.

Local Jev answers PLAY/SEARCH/PAUSE/RESUME/SKIP, never OPEN_SPOTIFY. If Spotify is
closed, the planner adds OpenSpotify as a prerequisite; the semantic goal is
unchanged by execution state.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping

from ..goals import GoalSchema, SlotSpec
from ..operators import ActionOperator, OperatorMetadata
from ..registry import CapabilityDescriptor
from ..world import Arg, Compare, Condition, Effect, WorldState
from ..adapters import MemorySpotifyAdapter, SpotifyAdapter

logger = logging.getLogger(__name__)

WORLD_SCHEMA: Mapping[str, Any] = {
    "spotify.running": False,
    "spotify.focused": False,
    "spotify.search_results": None,
    "spotify.search_query": None,
    "spotify.search_ready": False,
    "spotify.current_track": None,
    "spotify.current_artist": None,
    "spotify.paused": False,
    # One-shot completion pulse for SKIP. Adapters clear it on the next
    # observation so a later "Skip it" utterance executes again.
    "spotify.skipped": False,
    "media.playing": False,
    "media.query": None,
}
SPOTIFY_WORLD_SCHEMA = WORLD_SCHEMA


def _track_for(world: WorldState, arguments: Mapping[str, Any]) -> str:
    query = arguments.get("query") or world.get("spotify.search_results")
    return f"{query} - top result"


def spotify_goal_types() -> tuple[str, ...]:
    """Goal types are declared once, in `spotify_goal_schemas`; this is a convenience view."""
    return tuple(spotify_goal_schemas())


def _results_match_query(world: WorldState, arguments: Mapping[str, Any]) -> bool:
    """Search results exist and correspond to the requested query.

    Buttons: when the goal names a query, playing the *wrong* results is a
    correctness bug, so the results must equal that query. When no query was given
    ("play it"), any existing results are what the user meant.
    """
    results = world.get("spotify.search_results")
    if not results:
        return False
    query = arguments.get("query")
    return not query or results == query


CONDITION_RESULTS_FOR_QUERY = Condition("spotify.search_results", bound_predicate=_results_match_query)


def _playing_matches_query(world: WorldState, arguments: Mapping[str, Any]) -> bool:
    query = arguments.get("query")
    if not query:
        return bool(world.get("media.playing"))
    media_query = world.get("media.query")
    return media_query == query


CONDITION_PLAYING_FOR_QUERY = Condition("media.query", bound_predicate=_playing_matches_query)


def spotify_operators(adapter: SpotifyAdapter | None = None) -> tuple[ActionOperator, ...]:
    """Operators with explicit parameters, preconditions, effects, cost, and policy."""
    adapter = adapter or MemorySpotifyAdapter(WORLD_SCHEMA)

    async def open_adapter(world, arguments):
        await adapter.open()
        return {}

    async def search_adapter(world, arguments):
        await adapter.search(arguments["query"])
        return {}

    async def play_adapter(world, arguments):
        await adapter.play_result(arguments.get("query"))
        return {}

    async def pause_adapter(world, arguments):
        await adapter.pause()
        return {}

    async def resume_adapter(world, arguments):
        await adapter.resume()
        return {}

    async def skip_adapter(world, arguments):
        await adapter.skip()
        return {}

    return (
        ActionOperator(
            name="OpenSpotify",
            preconditions=(Condition("spotify.running", Compare.FALSY),),
            effects=(Effect("spotify.running", True), Effect("spotify.focused", True)),
            cost=1.0,
            executor=open_adapter,
            satisfies_goals=("OPEN",),
            metadata=OperatorMetadata(speculative_safe=True, reversible=True, idempotent=True),
        ),
        ActionOperator(
            name="SearchSpotify",
            parameters=("query",),
            preconditions=(Condition("spotify.running"),),
            effects=(Effect("spotify.search_results", Arg("query")),
                     Effect("spotify.search_query", Arg("query")),
                     Effect("spotify.search_ready", True)),
            cost=1.0,
            executor=search_adapter,
            metadata=OperatorMetadata(speculative_safe=True, reversible=True, idempotent=True),
        ),
        ActionOperator(
            name="PlayResult",
            # No required parameters: with a query it plays that query's results, and
            # without one it plays whatever search is already showing.
            optional_parameters=("query",),
            preconditions=(Condition("spotify.running"), CONDITION_RESULTS_FOR_QUERY),
            # Declared effects must match what the executor observes, or the planner's
            # projection and the real world state diverge. The track is the requested
            # query when there is one, otherwise whatever search is already showing.
            effects=(Effect("media.playing", True), Effect("spotify.paused", False),
                     Effect("spotify.current_track",
                            derive=lambda world, args: _track_for(world, args)),
                     Effect("media.query", derive=lambda world, args:
                            args.get("query") or world.get("spotify.search_query") or world.get("spotify.search_results"))),
            cost=1.0,
            executor=play_adapter,
            metadata=OperatorMetadata(speculative_safe=False, reversible=False, idempotent=False),
        ),
        ActionOperator(
            name="PauseSpotify",
            preconditions=(Condition("spotify.running"), Condition("media.playing")),
            effects=(Effect("spotify.paused", True), Effect("media.playing", False)),
            cost=1.0,
            executor=pause_adapter,
            metadata=OperatorMetadata(speculative_safe=False, reversible=True, idempotent=True),
        ),
        ActionOperator(
            name="ResumeSpotify",
            preconditions=(Condition("spotify.running"), Condition("spotify.current_track")),
            effects=(Effect("spotify.paused", False), Effect("media.playing", True)),
            cost=1.0,
            executor=resume_adapter,
            metadata=OperatorMetadata(speculative_safe=False, reversible=True, idempotent=True),
        ),
        ActionOperator(
            name="SkipSpotify",
            preconditions=(Condition("spotify.running"), Condition("spotify.current_track")),
            # Declared effects must match exactly what the executor observes. Skipping
            # also resumes playback, which is what makes SKIP satisfiable afterwards.
            effects=(Effect("spotify.skipped", True),
                     Effect("media.playing", True),
                     Effect("spotify.paused", False)),
            cost=1.0,
            executor=skip_adapter,
            metadata=OperatorMetadata(speculative_safe=False, reversible=False, idempotent=False),
        ),
    )


def spotify_goal_schemas() -> Mapping[str, GoalSchema]:
    """Goal schemas with their open-ended slots and world-state satisfaction tests."""
    return {
        "OPEN": GoalSchema("OPEN", satisfied_when=(Condition("spotify.running"),)),
        "PLAY": GoalSchema("PLAY", slots=(SlotSpec("query", "What music does the user want to play?"),),
                           satisfied_when=(Condition("media.playing"), CONDITION_PLAYING_FOR_QUERY,
                                           Condition("spotify.current_track", Compare.TRUTHY))),
        "PLAY_RESULT": GoalSchema(
            "PLAY_RESULT",
            satisfied_when=(Condition("media.playing"), Condition("spotify.current_track", Compare.TRUTHY)),
            presupposes=(Condition("spotify.search_results", Compare.TRUTHY),),
        ),
        "SEARCH": GoalSchema("SEARCH", slots=(SlotSpec("query", "What music does the user want to search for?"),),
                             satisfied_when=(Condition("spotify.search_results", Compare.EQ, Arg("query")),)),
        "PAUSE": GoalSchema("PAUSE", satisfied_when=(Condition("media.playing", Compare.FALSY),
                                                     Condition("spotify.paused"),)),
        "RESUME": GoalSchema("RESUME", satisfied_when=(Condition("media.playing"),)),
        "SKIP": GoalSchema("SKIP",
                           satisfied_when=(Condition("spotify.skipped", Compare.TRUTHY),),
                           presupposes=(Condition("spotify.current_track", Compare.TRUTHY),)),
    }


class SpotifyCapability:
    """Bundles Spotify's Local Jev goals, schemas, and operators."""

    name = "Spotify"
    description = ("Music playback through Spotify: play a song, artist, album, or genre, search the "
                   "catalog, pause, resume, or skip the current track.")
    world_schema = WORLD_SCHEMA

    def __init__(self, local_jev: Any, adapter: SpotifyAdapter | None = None) -> None:
        self.local_jev = local_jev
        self.adapter = adapter or MemorySpotifyAdapter(WORLD_SCHEMA)

    def descriptor(self) -> CapabilityDescriptor:
        return CapabilityDescriptor(name=self.name, description=self.description,
                                    local_jev=self.local_jev,
                                    goal_schemas=spotify_goal_schemas(),
                                    operators=spotify_operators(self.adapter),
                                    world_schema=WORLD_SCHEMA,
                                    observer=self.adapter.observe)


def build_spotify_descriptor(local_jev: Any, *, adapter: SpotifyAdapter | None = None) -> CapabilityDescriptor:
    return SpotifyCapability(local_jev, adapter=adapter).descriptor()
