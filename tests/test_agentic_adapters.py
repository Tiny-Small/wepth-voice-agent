"""Application adapters keep UI control out of goals, planners, and Jevs."""

import json
import sys
from pathlib import Path

import pytest

from ping_ponder.agentic.capabilities.spotify import SPOTIFY_WORLD_SCHEMA, build_spotify_descriptor
from ping_ponder.agentic.executor import ExecutionOutcome, Executor
from ping_ponder.agentic.goals import SemanticGoal
from ping_ponder.agentic.adapters.memory import MemorySpotifyAdapter
from ping_ponder.agentic.adapters.memory import MemoryBrowserAdapter
from ping_ponder.agentic.capabilities.browser import BROWSER_WORLD_SCHEMA, build_browser_descriptor
from ping_ponder.agentic.adapters.powershell import (
    PowerShellOutputError, PowerShellProcessError, PowerShellRunner, PowerShellTimeout,
)
from ping_ponder.agentic.execution_config import DESKTOP, SIMULATED, ExecutionSettings
from ping_ponder.agentic.adapters.windows_spotify import WindowsSpotifyAdapter
from ping_ponder.agentic.adapters.windows_browser import WindowsBrowserAdapter
from ping_ponder.agentic.adapters.base import AdapterError
from ping_ponder.agentic.planner import DeterministicPlanner
from ping_ponder.agentic.world import WorldState


@pytest.mark.asyncio
async def test_spotify_capability_uses_adapter_observation_as_world_truth():
    adapter = MemorySpotifyAdapter(SPOTIFY_WORLD_SCHEMA)
    descriptor = build_spotify_descriptor(None, adapter=adapter)
    report = await Executor(DeterministicPlanner()).execute(
        SemanticGoal("Spotify", "PLAY", {"query": "Jazz"}),
        descriptor.schema("PLAY"), WorldState(descriptor.world_schema), descriptor)

    assert report.outcome is ExecutionOutcome.SATISFIED
    assert report.executed == ("OpenSpotify", "SearchSpotify", "PlayResult")
    assert report.world_after.get("spotify.search_query") == "Jazz"
    assert report.world_after.get("media.query") == "Jazz"
    assert report.world_after.get("media.playing") is True


@pytest.mark.asyncio
async def test_memory_spotify_preserves_literal_query():
    query = 'Björk\'s "Jóga" & strings'
    adapter = MemorySpotifyAdapter(SPOTIFY_WORLD_SCHEMA)
    descriptor = build_spotify_descriptor(None, adapter=adapter)
    report = await Executor(DeterministicPlanner()).execute(
        SemanticGoal("Spotify", "PLAY", {"query": query}),
        descriptor.schema("PLAY"), WorldState(descriptor.world_schema), descriptor)

    assert report.outcome is ExecutionOutcome.SATISFIED
    assert report.world_after.get("spotify.search_query") == query
    assert report.world_after.get("media.query") == query


def _fixture_script(tmp_path, body: str):
    path = tmp_path / "fixture.py"
    path.write_text(body, encoding="utf-8")
    return path


@pytest.mark.asyncio
async def test_powershell_runner_preserves_literal_argv(tmp_path):
    path = _fixture_script(tmp_path, """
import json, sys
print(json.dumps({'operation': sys.argv[1], 'arguments': sys.argv[2:]}))
""")
    runner = PowerShellRunner(executable=sys.executable, prelude=(), script_flag=None, timeout=2)
    value = await runner.run(str(path), "search", 'Björk\'s "Jóga" & strings')
    assert value == {"operation": "search", "arguments": ['Björk\'s "Jóga" & strings']}


@pytest.mark.asyncio
async def test_powershell_runner_rejects_malformed_json(tmp_path):
    path = _fixture_script(tmp_path, "print('not-json')")
    runner = PowerShellRunner(executable=sys.executable, prelude=(), script_flag=None, timeout=2)
    with pytest.raises(PowerShellOutputError):
        await runner.run(str(path), "observe")


@pytest.mark.asyncio
async def test_powershell_runner_reports_nonzero_exit(tmp_path):
    path = _fixture_script(tmp_path, "import sys; print('bad'); sys.exit(3)")
    runner = PowerShellRunner(executable=sys.executable, prelude=(), script_flag=None, timeout=2)
    with pytest.raises(PowerShellProcessError):
        await runner.run(str(path), "observe")


@pytest.mark.asyncio
async def test_powershell_runner_times_out(tmp_path):
    path = _fixture_script(tmp_path, "import time; time.sleep(2)")
    runner = PowerShellRunner(executable=sys.executable, prelude=(), script_flag=None, timeout=0.01)
    with pytest.raises(PowerShellTimeout):
        await runner.run(str(path), "observe")


def test_execution_settings_default_to_simulated_and_accept_desktop():
    assert ExecutionSettings.from_env({}).backend == SIMULATED
    assert ExecutionSettings.from_env({"AGENTIC_EXECUTION_BACKEND": "desktop"}).backend == DESKTOP


class _SpotifyRunner:
    calls = 0
    async def run(self, script, operation, *arguments):
        if operation == "observe":
            self.calls += 1
            if self.calls == 1:
                return {"running": True, "focused": True, "status": "Paused",
                        "source": "Spotify", "title": "Old", "artist": "Other",
                        "search_query": 'Björk\'s "Jóga" & strings', "search_ready": True}
            return {"running": True, "focused": True, "status": "Playing",
                    "source": "Spotify", "title": "Jóga", "artist": "Björk",
                    "search_query": 'Björk\'s "Jóga" & strings', "search_ready": True}
        return {}


@pytest.mark.asyncio
async def test_windows_spotify_observation_requires_verified_playing_metadata():
    adapter = WindowsSpotifyAdapter(_SpotifyRunner(), script="fake.ps1")
    await adapter.play_result('Björk\'s "Jóga" & strings')
    state = await adapter.observe()
    assert state["media.playing"] is True
    assert state["media.query"] == 'Björk\'s "Jóga" & strings'
    assert state["spotify.current_track"] == "Björk - Jóga"


@pytest.mark.asyncio
async def test_windows_spotify_play_accepts_resuming_the_already_selected_paused_track():
    class ResumeRunner:
        def __init__(self):
            self.observes = 0

        async def run(self, script, operation, *arguments):
            if operation == "observe":
                self.observes += 1
                return {"running": True,
                        "status": "Paused" if self.observes == 1 else "Playing",
                        "source": "Spotify", "title": "Soul Eyes", "artist": "Carl Winther",
                        "search_query": "Jazz", "search_ready": True}
            return {}

    adapter = WindowsSpotifyAdapter(ResumeRunner(), script="fake.ps1", settle_timeout=0.2)
    await adapter.play_result("Jazz")
    state = await adapter.observe()
    assert state["media.query"] == "Jazz"


@pytest.mark.asyncio
async def test_browser_capability_reuses_memory_adapter_and_planner():
    adapter = MemoryBrowserAdapter(BROWSER_WORLD_SCHEMA)
    descriptor = build_browser_descriptor(None, adapter=adapter)
    report = await Executor(DeterministicPlanner()).execute(
        SemanticGoal("Browser", "SEARCH", {"query": "Jev documentation"}),
        descriptor.schema("SEARCH"), WorldState(descriptor.world_schema), descriptor)
    assert report.outcome is ExecutionOutcome.SATISFIED
    assert report.executed == ("OpenBrowser", "SearchWeb")
    assert report.world_after.get("browser.search_query") == "Jev documentation"


class _BrowserRunner:
    async def run(self, script, operation, *arguments):
        if operation == "observe":
            return {"running": True, "current_url": "https://example.com",
                    "navigation_target": "https://example.com/a?x=Jóga&y=1"}
        return {}


@pytest.mark.asyncio
async def test_windows_browser_preserves_navigation_target():
    adapter = WindowsBrowserAdapter(_BrowserRunner(), script="fake.ps1")
    await adapter.navigate("https://example.com/a?x=Jóga&y=1")
    state = await adapter.observe()
    assert state["browser.running"] is True
    assert state["browser.current_url"] == "https://example.com"
    assert state["browser.navigation_target"] == "https://example.com/a?x=Jóga&y=1"


@pytest.mark.asyncio
async def test_windows_browser_derives_search_query_from_observed_url():
    class UrlRunner:
        async def run(self, script, operation, *arguments):
            return {"running": True, "current_url": "https://www.google.com/search?q=Jev+documentation"}
    state = await WindowsBrowserAdapter(UrlRunner(), script="fake.ps1").observe()
    assert state["browser.search_query"] == "Jev documentation"


class _NoOpNativeRunner:
    async def run(self, script, operation, *arguments):
        return {"running": True} if operation == "observe" else {}


@pytest.mark.asyncio
async def test_native_observation_cannot_claim_browser_search_without_ui_evidence():
    adapter = WindowsBrowserAdapter(_NoOpNativeRunner(), script="fake.ps1")
    descriptor = build_browser_descriptor(None, adapter=adapter)
    report = await Executor(DeterministicPlanner()).execute(
        SemanticGoal("Browser", "SEARCH", {"query": "Jev documentation"}),
        descriptor.schema("SEARCH"), WorldState(descriptor.world_schema), descriptor)
    assert report.outcome is not ExecutionOutcome.SATISFIED


@pytest.mark.asyncio
async def test_native_spotify_does_not_assign_query_to_unrelated_playback():
    class UnrelatedRunner:
        async def run(self, script, operation, *arguments):
            if operation == "observe":
                return {"running": True, "status": "Playing", "source": "Spotify",
                        "title": "Kind of Blue", "artist": "Miles Davis",
                        "search_query": "Miles Davis", "search_ready": True}
            return {}
    adapter = WindowsSpotifyAdapter(UnrelatedRunner(), script="fake.ps1", settle_timeout=0.01)
    with pytest.raises(AdapterError):
        await adapter.play_result("Jazz")


@pytest.mark.asyncio
async def test_native_spotify_observation_never_infers_query_without_play_invocation():
    class ExistingMusicRunner:
        async def run(self, script, operation, *arguments):
            if operation == "observe":
                return {"running": True, "status": "Playing", "source": "Spotify",
                        "title": "Kind of Blue", "artist": "Miles Davis",
                        "search_query": "Jazz", "search_ready": True}
            return {}
    state = await WindowsSpotifyAdapter(ExistingMusicRunner(), script="fake.ps1").observe()
    assert state["media.query"] is None


@pytest.mark.asyncio
async def test_windows_spotify_search_tracks_verified_native_search_without_edit_value():
    class SearchRunner:
        async def run(self, script, operation, *arguments):
            if operation == "search":
                return {"running": True, "search_query": "Jazz",
                        "search_results": "Jazz", "search_ready": True}
            if operation == "observe":
                return {"running": True, "status": "Paused", "source": "Spotify",
                        "search_query": None, "search_results": None, "search_ready": True}
            return {}

    adapter = WindowsSpotifyAdapter(SearchRunner(), script="fake.ps1")
    await adapter.search("Jazz")
    state = await adapter.observe()
    assert state["spotify.search_query"] == "Jazz"
    assert state["spotify.search_results"] == "Jazz"
    assert state["spotify.search_ready"] is True


@pytest.mark.asyncio
async def test_windows_spotify_retries_one_transient_search_timeout():
    class RetryingSearchRunner:
        def __init__(self):
            self.search_attempts = 0

        async def run(self, script, operation, *arguments):
            if operation == "search":
                self.search_attempts += 1
                if self.search_attempts == 1:
                    raise PowerShellTimeout("PowerShell operation 'search' timed out")
                return {"running": True, "search_query": "Jazz",
                        "search_results": "Jazz", "search_ready": True}
            return {}

    runner = RetryingSearchRunner()
    await WindowsSpotifyAdapter(runner, script="fake.ps1").search("Jazz")
    assert runner.search_attempts == 2


@pytest.mark.asyncio
async def test_windows_spotify_play_uses_verified_query_when_observer_cannot_read_search_text():
    class SearchThenPlayRunner:
        def __init__(self):
            self.observes = 0

        async def run(self, script, operation, *arguments):
            if operation == "search":
                return {"running": True, "search_query": "Jazz",
                        "search_results": "Jazz", "search_ready": True}
            if operation == "observe":
                self.observes += 1
                if self.observes == 1:
                    return {"running": True, "status": "Paused", "source": "Spotify",
                            "title": "Old", "artist": "Other", "search_ready": True}
                return {"running": True, "status": "Playing", "source": "Spotify",
                        "title": "New", "artist": "Artist", "search_ready": False}
            return {}

    adapter = WindowsSpotifyAdapter(SearchThenPlayRunner(), script="fake.ps1", settle_timeout=0.2)
    await adapter.search("Jazz")
    await adapter.play_result("Jazz")


@pytest.mark.asyncio
async def test_windows_spotify_skip_reports_one_completed_skip():
    class SkipRunner:
        async def run(self, script, operation, *arguments):
            if operation == "observe":
                return {"running": True, "status": "Playing", "source": "Spotify",
                        "title": "Next track", "artist": "Artist", "skipped": 0}
            if operation == "skip":
                return {"running": True}
            return {}

    adapter = WindowsSpotifyAdapter(SkipRunner(), script="fake.ps1")
    await adapter.skip()
    state = await adapter.observe()
    assert state["spotify.skipped"] is True


@pytest.mark.asyncio
async def test_native_skip_goal_executes_once_and_satisfies():
    class SkipRunner:
        async def run(self, script, operation, *arguments):
            if operation == "observe":
                return {"running": True, "status": "Playing", "source": "Spotify",
                        "title": "Current track", "artist": "Artist", "skipped": 0}
            return {"running": True}

    adapter = WindowsSpotifyAdapter(SkipRunner(), script="fake.ps1")
    descriptor = build_spotify_descriptor(None, adapter=adapter)
    report = await Executor(DeterministicPlanner()).execute(
        SemanticGoal("Spotify", "SKIP", {}), descriptor.schema("SKIP"),
        WorldState(descriptor.world_schema), descriptor)
    assert report.outcome is ExecutionOutcome.SATISFIED
    assert report.executed == ("SkipSpotify",)


def test_spotify_script_uses_explicit_winrt_result_types():
    script = (Path(__file__).parents[1] / "src/ping_ponder/agentic/adapters/scripts/spotify.ps1").read_text()
    assert "function Await-WinRT($operation, [Type]$resultType)" in script
    assert "MakeGenericMethod($resultType)" in script
    assert "ParameterType.Name -eq 'IAsyncOperation`1'" in script


def test_spotify_search_uses_keyboard_input_not_read_only_omnibox():
    script = (Path(__file__).parents[1] / "src/ping_ponder/agentic/adapters/scripts/spotify.ps1").read_text()
    assert "function Send-SpotifySearchQuery($text)" in script
    assert "Clipboard]::SetText($text)" in script
    assert "SendKeys]::SendWait(\"^v\")" in script
    assert "ValuePattern]::Pattern).SetValue($text)" not in script


def test_spotify_native_search_uses_search_button_and_result_grid():
    script = (Path(__file__).parents[1] / "src/ping_ponder/agentic/adapters/scripts/spotify.ps1").read_text()
    assert 'NameProperty, "Search"' in script
    assert 'ControlType.ProgrammaticName -ne "ControlType.DataGrid"' in script
    assert 'NameProperty, "Search results"' in script


def test_browser_script_finds_address_bar_with_uia_names_and_keyboard_fallback():
    script = (Path(__file__).parents[1] / "src/ping_ponder/agentic/adapters/scripts/browser.ps1").read_text()
    assert '"Address and search bar"' in script
    assert '"Address bar"' in script
    assert 'AutomationElement]::NameProperty' in script
    assert 'SendKeys]::SendWait("^l")' in script


def test_browser_search_reuses_an_existing_chrome_window():
    script = (Path(__file__).parents[1] / "src/ping_ponder/agentic/adapters/scripts/browser.ps1").read_text()
    search_block = script.split('    "search" {', 1)[1].split('    "navigate" {', 1)[0]
    assert "Ensure-ChromeWindow" in search_block
    assert 'Start-Process "chrome.exe"' not in search_block
