"""Opt-in native application acceptance tests.

These tests are skipped unless the caller explicitly opts into application-changing
desktop automation with AGENTIC_DESKTOP_TESTS=1.
"""

import os

import pytest

from ping_ponder.agentic.execution_config import DESKTOP, ExecutionSettings
from ping_ponder.agentic.wiring import build_chat_session


pytestmark = [pytest.mark.integration,
              pytest.mark.skipif(os.environ.get("AGENTIC_DESKTOP_TESTS") != "1",
                                 reason="set AGENTIC_DESKTOP_TESTS=1 to control desktop applications")]


@pytest.mark.asyncio
async def test_spotify_desktop_vertical_slice_observes_requested_playback():
    session = build_chat_session(execution=ExecutionSettings(DESKTOP))
    trace = await session.say("Play some Jazz")
    assert trace.outcome == "SATISFIED", trace.reason
    assert trace.world_after["media.playing"] is True
    assert trace.world_after["media.query"] == "Jazz"


@pytest.mark.asyncio
async def test_browser_desktop_search_observes_requested_query():
    session = build_chat_session(execution=ExecutionSettings(DESKTOP))
    trace = await session.say("Search for Jev documentation")
    assert trace.outcome == "SATISFIED", trace.reason
    assert trace.world_after["browser.search_query"] == "Jev documentation"
