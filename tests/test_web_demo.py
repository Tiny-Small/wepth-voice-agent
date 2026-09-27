"""Hosted adapter boundaries and one real semantic/control path."""

import json
import logging
import wave
from types import SimpleNamespace

import assemblyai
import assemblyai.sync.v1
import pytest

from ping_ponder.agentic.browser_use_slice import (
    BrowserCompletionStatus, BrowserDecision, BrowserObservation,
    BrowserTaskExecutor, JevBrowserCapability,
)
from ping_ponder.agentic.span import ExtractedSpan
from ping_ponder.agentic.wiring import build_chat_session
from ping_ponder.agentic.backend_config import BackendSettings
from ping_ponder.agentic.execution_config import ExecutionSettings
from ping_ponder.agentic.reply import SilentReplyComposer
from ping_ponder.voice.web_demo import HostedDemo, HostedRuntime, transcribe_recording

FIND = "Find the Browser Use repository on GitHub"
URL = "https://github.com/browser-use/browser-use"


class Extractor:
    name = "fixture"

    async def extract_many(self, utterance, requests):
        values = {"site": "GitHub", "target": "Browser Use repository"}
        return {request.slot: ExtractedSpan(
            slot=request.slot, question=request.question, text=values[request.slot],
            start=utterance.index(values[request.slot]),
            end=utterance.index(values[request.slot]) + len(values[request.slot]),
            confidence=1.0, extractor=self.name,
        ) for request in requests if request.slot in values and values[request.slot] in utterance}


class Page:
    def __init__(self, url=URL):
        self.url = url
        self.observations = 0

    async def observe(self):
        self.observations += 1
        return BrowserObservation("page", self.url, "browser-use/browser-use", "repository", frozenset(), "page")

    async def act(self, action):
        return {}


class Controller:
    async def next_actions(self, task, observation, available_actions, memory):
        return BrowserDecision(completion_status=BrowserCompletionStatus.UNCERTAIN)


class Verifier:
    def __init__(self, expected_url=URL):
        self.expected_url = expected_url

    async def verify(self, task, observation):
        return (BrowserCompletionStatus.SATISFIED if observation.url == self.expected_url
                else BrowserCompletionStatus.UNCERTAIN)


def demo(page=None, transcribe=None, extractor=None, expected_url=URL):
    page = page or Page()
    browser = JevBrowserCapability(BrowserTaskExecutor(
        page, Controller(), completion_verifier=Verifier(expected_url)))
    session = build_chat_session(settings=BackendSettings(), execution=ExecutionSettings(),
                                 extractor=extractor or Extractor(), browser_find=browser,
                                 replies=SilentReplyComposer())
    return HostedDemo(session=session, browser_find=browser, transcribe=transcribe), page


@pytest.mark.asyncio
async def test_text_find_uses_existing_level_two_and_returns_url():
    app, page = demo()
    result = await app.submit(text=FIND)
    assert result.transcript == FIND
    assert result.goal.startswith("Browser.FIND")
    assert result.capability == "Browser"
    assert result.goal_type == "FIND"
    assert result.site == "GitHub"
    assert result.target == "Browser Use repository"
    assert result.planner == "FindWithGroundedBrowser"
    assert result.status == "SATISFIED"
    assert result.final_url == URL
    assert page.observations > 0


@pytest.mark.asyncio
async def test_text_site_search_reaches_results_page_through_semantic_pipeline():
    command = "Search GitHub for browser-use"
    results_url = "https://github.com/search?q=browser-use"

    class SearchExtractor:
        name = "fixture"

        async def extract_many(self, utterance, requests):
            values = {"site": "GitHub", "query": "browser-use"}
            return {request.slot: ExtractedSpan(
                slot=request.slot, question=request.question, text=values[request.slot],
                start=utterance.index(values[request.slot]),
                end=utterance.index(values[request.slot]) + len(values[request.slot]),
                confidence=1.0, extractor=self.name,
            ) for request in requests if request.slot in values}

    app, page = demo(Page(results_url), extractor=SearchExtractor(), expected_url=results_url)
    result = await app.submit(text=command)

    assert result.goal == "Browser.SEARCH_WEBSITE(query='browser-use', site='GitHub')"
    assert result.goal_type == "SEARCH_WEBSITE"
    assert result.site == "GitHub"
    assert result.target == "browser-use"
    assert result.planner == "SearchWebsiteWithGroundedBrowser"
    assert result.status == "SATISFIED"
    assert result.final_url == results_url
    assert page.observations > 0


@pytest.mark.asyncio
async def test_public_result_hides_internal_execution_error(monkeypatch):
    app, _ = demo()

    async def broken(evaluation):
        raise RuntimeError("secret-api-key provider debug chain-of-thought")

    monkeypatch.setattr(app.session.spine, "reconcile_final", broken)
    result = await app.submit(text=FIND)
    assert result.status == "EXECUTION_ERROR"
    assert result.capability == "Browser"
    assert result.goal_type == "FIND"
    assert result.site == "GitHub"
    assert result.target == "Browser Use repository"
    assert result.planner == ""
    assert "secret-api-key" not in repr(result)
    assert "provider debug" not in repr(result)
    assert "chain-of-thought" not in repr(result)


@pytest.mark.asyncio
async def test_hosted_runtime_closes_jev_backend_without_browser_use_provider(monkeypatch):
    closed = []

    async def close_jevs(*args):
        closed.append("semantic")

    class BrowserSession:
        async def close(self, *, kill_browser):
            assert kill_browser is True
            closed.append("browser")

    class Provider:
        async def aclose(self):
            closed.append("verifier")

    monkeypatch.setattr("ping_ponder.voice.web_demo.aclose_jevs", close_jevs)
    runtime = HostedRuntime.__new__(HostedRuntime)
    runtime.session = SimpleNamespace(spine=SimpleNamespace(global_jev=None), slot_extractor=None)
    runtime.browser_session = BrowserSession()
    runtime.browser_provider = None
    runtime.browser_jev_provider = Provider()

    await runtime.close()

    assert closed == ["semantic", "browser", "verifier"]


@pytest.mark.asyncio
async def test_audio_uses_same_semantic_path():
    calls = []

    async def transcribe(path):
        calls.append(path)
        return FIND

    app, page = demo(transcribe=transcribe)
    result = await app.submit(audio_path="sample.wav")
    assert calls == ["sample.wav"]
    assert result.status == "SATISFIED"
    assert result.final_url == URL
    assert page.observations > 0


@pytest.mark.asyncio
async def test_spotify_is_blocked_before_execution():
    app, page = demo()
    result = await app.submit(text="Open Spotify")
    assert result.status == "UNSUPPORTED"
    assert "local desktop" in result.message
    assert page.observations == 0
    assert app.session.spine.world.get("spotify.running") is False


@pytest.mark.asyncio
async def test_uncertain_find_is_readable():
    app, _ = demo(Page("https://github.com/search?q=browser-use"))
    result = await app.submit(text=FIND)
    assert result.status != "SATISFIED"
    assert result.message


@pytest.mark.asyncio
async def test_missing_audio_and_transcription_failure_are_results():
    app, _ = demo()
    assert (await app.submit()).status == "INPUT_ERROR"

    async def broken(path):
        raise RuntimeError("bad audio")

    app, _ = demo(transcribe=broken)
    result = await app.submit(audio_path="bad.wav")
    assert result.status == "TRANSCRIPTION_ERROR"
    assert "AssemblyAI" in result.message


@pytest.mark.asyncio
async def test_recording_adapter_rejects_invalid_file(tmp_path):
    invalid = tmp_path / "empty.wav"
    invalid.write_bytes(b"")
    with pytest.raises(ValueError, match="empty"):
        await transcribe_recording(str(invalid), api_key="test", sync_transcriber_factory=lambda key: None)


@pytest.mark.asyncio
async def test_recorded_wav_uses_one_sync_request_without_polling(tmp_path, monkeypatch, caplog):
    recording = tmp_path / "command.wav"
    with wave.open(str(recording), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(b"\0\0" * 16000)
    paths = []

    async def run_inline(function, *args):
        return function(*args)

    monkeypatch.setattr("ping_ponder.voice.web_demo.asyncio.to_thread", run_inline)

    class Transcriber:
        def transcribe(self, path):
            paths.append(path)
            return SimpleNamespace(text=FIND, audio_duration_ms=1000)

    with caplog.at_level(logging.WARNING):
        text = await transcribe_recording(str(recording), api_key="test",
                                          sync_transcriber_factory=lambda key: Transcriber())
    assert text == FIND
    assert paths == [str(recording)]
    timing = json.loads(next(record.message.split("stt_latency ", 1)[1]
                             for record in caplog.records if "stt_latency " in record.message))
    assert timing["audio_duration_s"] == 1.0
    assert timing["conversion_latency_s"] == 0.0
    assert timing["request_latency_s"] >= 0
    assert timing["total_latency_s"] >= timing["request_latency_s"]
    assert all(event in timing["events"] for event in (
        "audio_received", "conversion_start", "conversion_complete",
        "sync_request_start", "sync_response_received", "transcript_available"))


@pytest.mark.asyncio
async def test_default_recording_adapter_selects_sync_sdk(tmp_path, monkeypatch):
    recording = tmp_path / "command.wav"
    with wave.open(str(recording), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(b"\0\0" * 16000)

    async def run_inline(function, *args):
        return function(*args)

    monkeypatch.setattr("ping_ponder.voice.web_demo.asyncio.to_thread", run_inline)
    monkeypatch.setattr(assemblyai, "Transcriber", lambda **kwargs: pytest.fail("async SDK was used"))

    class SyncTranscriber:
        def __init__(self, *, api_key):
            assert api_key == "test-key"

        def transcribe(self, path):
            assert path == str(recording)
            return SimpleNamespace(text=FIND)

    monkeypatch.setattr(assemblyai.sync.v1, "SyncTranscriber", SyncTranscriber)
    assert await transcribe_recording(str(recording), api_key="test-key") == FIND


@pytest.mark.asyncio
async def test_sync_provider_failure_is_readable_and_does_not_leak_key(tmp_path, monkeypatch, caplog):
    recording = tmp_path / "command.wav"
    with wave.open(str(recording), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(b"\0\0" * 16000)

    async def run_inline(function, *args):
        return function(*args)

    monkeypatch.setattr("ping_ponder.voice.web_demo.asyncio.to_thread", run_inline)

    class Transcriber:
        def transcribe(self, path):
            raise RuntimeError("provider rejected secret-test-key")

    async def transcribe(path):
        return await transcribe_recording(path, api_key="secret-test-key",
                                         sync_transcriber_factory=lambda key: Transcriber())

    app, _ = demo(transcribe=transcribe)
    with caplog.at_level(logging.WARNING):
        result = await app.submit(audio_path=str(recording))
    assert result.status == "TRANSCRIPTION_ERROR"
    assert "AssemblyAI transcription failed" in result.message
    assert "secret-test-key" not in result.message
    assert "secret-test-key" not in caplog.text
    assert "provider rejected" in caplog.text
