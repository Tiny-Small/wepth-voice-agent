import asyncio

import pytest

from ping_ponder.agentic.adapters.memory import MemoryBrowserAdapter, MemorySpotifyAdapter
from ping_ponder.agentic.capabilities.browser import WORLD_SCHEMA as BROWSER_WORLD_SCHEMA
from ping_ponder.agentic.capabilities.spotify import WORLD_SCHEMA as SPOTIFY_WORLD_SCHEMA
from ping_ponder.agentic.execution_config import AdapterBundle
from ping_ponder.agentic.goal_builder import GoalBuilder
from ping_ponder.agentic.registry import CapabilityDescriptor
from ping_ponder.agentic.span import HeuristicSpanExtractor
from ping_ponder.agentic.spine import VoiceActionSpine
from ping_ponder.agentic.wiring import (
    RuleBasedGlobalJev,
    build_default_registry,
    build_default_world,
)
from ping_ponder.voice.coordinator import TurnCoordinator
from ping_ponder.voice.events import FinalTranscript, PartialTranscript, TranscriptState
from ping_ponder.voice.tts import RecordingSpeechSynthesizer
from ping_ponder.voice.wiring import VoiceRuntime


class GatedSpotifyAdapter(MemorySpotifyAdapter):
    def __init__(self):
        super().__init__(SPOTIFY_WORLD_SCHEMA)
        self.opened = asyncio.Event()
        self.search_started = asyncio.Event()
        self.release_search = asyncio.Event()
        self.actions = []

    async def open(self):
        await super().open()
        self.actions.append("OpenSpotify")
        self.opened.set()

    async def search(self, query):
        self.search_started.set()
        await self.release_search.wait()
        await super().search(query)
        self.actions.append("SearchSpotify")

    async def play_result(self, query):
        await super().play_result(query)
        self.actions.append("PlayResult")


class CountingGlobal:
    def __init__(self, inner):
        self.inner = inner
        self.calls = 0

    async def route(self, *args, **kwargs):
        self.calls += 1
        return await self.inner.route(*args, **kwargs)


class CountingLocal:
    def __init__(self, inner):
        self.inner = inner
        self.calls = 0

    async def interpret(self, *args, **kwargs):
        self.calls += 1
        return await self.inner.interpret(*args, **kwargs)


class CountingExtractor:
    name = "counting-heuristic"

    def __init__(self):
        self.inner = HeuristicSpanExtractor()
        self.calls = 0

    async def extract(self, *args, **kwargs):
        self.calls += 1
        return await self.inner.extract(*args, **kwargs)


class RecordingOutput:
    def __init__(self):
        self.ack_started = asyncio.Event()

    async def play(self, frames):
        self.ack_started.set()
        async for _ in frames:
            pass

    async def stop_playback(self):
        pass


@pytest.mark.asyncio
async def test_play_jazz_reuses_partial_semantics_and_overlaps_ack_with_final_execution():
    spotify = GatedSpotifyAdapter()
    registry = build_default_registry(adapters=AdapterBundle(
        spotify=spotify,
        browser=MemoryBrowserAdapter(BROWSER_WORLD_SCHEMA),
    ))
    spotify_descriptor = registry.get("Spotify")
    local = CountingLocal(spotify_descriptor.local_jev)
    registry.register(CapabilityDescriptor(
        name=spotify_descriptor.name,
        description=spotify_descriptor.description,
        local_jev=local,
        goal_schemas=spotify_descriptor.goal_schemas,
        operators=spotify_descriptor.operators,
        world_schema=spotify_descriptor.world_schema,
        confirmation_subject=spotify_descriptor.confirmation_subject,
        observer=spotify_descriptor.observer,
    ))
    global_jev = CountingGlobal(RuleBasedGlobalJev())
    extractor = CountingExtractor()
    spine = VoiceActionSpine(
        registry=registry,
        global_jev=global_jev,
        goal_builder=GoalBuilder(extractor),
        world=build_default_world(),
    )
    synthesizer = RecordingSpeechSynthesizer()
    output = RecordingOutput()
    coordinator = TurnCoordinator(
        spine=spine, synthesizer=synthesizer, audio_output=output,
    )
    runtime = VoiceRuntime(coordinator=coordinator)
    await runtime.start()

    runtime.dispatch(PartialTranscript(TranscriptState(
        "t1", 1, "Play some jazz", False,
    )))
    await spotify.opened.wait()
    await spotify.search_started.wait()
    runtime.dispatch(FinalTranscript(TranscriptState(
        "t1", 2, "Play some jazz", True,
    )))

    await output.ack_started.wait()
    assert coordinator.final_control_running
    assert spotify.actions == ["OpenSpotify"]

    spotify.release_search.set()
    await runtime.wait_idle()

    assert global_jev.calls == 1
    assert local.calls == 1
    assert extractor.calls == 1
    assert spotify.actions == ["OpenSpotify", "SearchSpotify", "PlayResult"]
    assert synthesizer.utterances == ["Sure"]
    assert spine.world.get("media.playing") is True
    await runtime.stop()
