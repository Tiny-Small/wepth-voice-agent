"""Recorded-audio and text adapter around the existing semantic/control spine."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Awaitable

from ping_ponder.agentic.backend_config import BackendSettings
from scripts.run_voice_agent import build_browser_find
from ping_ponder.agentic.execution_config import ExecutionSettings, SIMULATED
from ping_ponder.agentic.reply import SilentReplyComposer
from ping_ponder.agentic.wiring import aclose_jevs, build_chat_session

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DemoResult:
    transcript: str = ""
    goal: str = ""
    status: str = ""
    message: str = ""
    final_url: str = ""
    page_title: str = ""
    capability: str = ""
    goal_type: str = ""
    site: str = ""
    target: str = ""
    planner: str = ""


async def transcribe_recording(path: str, *, api_key: str | None = None,
                               sync_transcriber_factory: Callable | None = None) -> str:
    """Send a completed WAV recording to AssemblyAI's one-request Sync STT API."""
    received = time.perf_counter()
    events = {"audio_received": round(time.time(), 3)}
    audio = Path(path)
    if not audio.is_file():
        raise ValueError("audio upload was not found")
    if audio.stat().st_size == 0:
        raise ValueError("audio upload is empty")
    if audio.stat().st_size > 25 * 1024 * 1024:
        raise ValueError("audio upload exceeds 25 MB")
    key = api_key or os.environ.get("ASSEMBLYAI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("ASSEMBLYAI_API_KEY is required for audio input")
    events["conversion_start"] = round(time.time(), 3)
    try:
        with wave.open(str(audio), "rb") as recording:
            if recording.getsampwidth() != 2:
                raise ValueError("audio must be 16-bit PCM WAV")
            duration = recording.getnframes() / recording.getframerate()
            sample_rate = recording.getframerate()
            channels = recording.getnchannels()
    except (wave.Error, EOFError) as error:
        raise ValueError("Sync STT requires a WAV PCM recording or upload") from error
    events["conversion_complete"] = round(time.time(), 3)
    if sync_transcriber_factory is None:
        from assemblyai.sync.v1 import SyncTranscriber
        sync_transcriber_factory = lambda token: SyncTranscriber(api_key=token)
    events["sync_request_start"] = round(time.time(), 3)
    request_start = time.perf_counter()
    try:
        response = await asyncio.to_thread(sync_transcriber_factory(key).transcribe, str(audio))
        events["sync_response_received"] = round(time.time(), 3)
        response_received = time.perf_counter()
        text = str(getattr(response, "text", "") or "").strip()
        if not text:
            raise RuntimeError("AssemblyAI returned no speech")
        events["transcript_available"] = round(time.time(), 3)
        return text
    except Exception as error:
        events["sync_response_received"] = round(time.time(), 3)
        response_received = time.perf_counter()
        detail = str(error).replace(key, "[REDACTED]")
        logger.error("sync_stt_error type=%s status=%s code=%s detail=%s",
                     type(error).__name__, getattr(error, "status_code", None),
                     getattr(error, "error_code", None), detail)
        raise
    finally:
        logger.warning("stt_latency %s", json.dumps({
            "events": events, "audio_size_bytes": audio.stat().st_size,
            "audio_format": "wav_pcm16", "sample_rate": sample_rate,
            "channels": channels, "audio_duration_s": round(duration, 3),
            "conversion_latency_s": 0.0,
            "request_latency_s": round(response_received - request_start, 3),
            "total_latency_s": round(time.perf_counter() - received, 3),
        }))


class HostedDemo:
    """One serialized hosted session; only Browser.FIND may reach execution."""

    def __init__(self, *, session, browser_find, transcribe: Callable[[str], Awaitable[str]] | None = None):
        self.session = session
        self.browser_find = browser_find
        self.transcribe = transcribe or transcribe_recording
        self._lock = asyncio.Lock()

    async def submit(self, *, text: str = "", audio_path: str | None = None) -> DemoResult:
        async with self._lock:
            utterance = text.strip()
            if not utterance:
                if not audio_path:
                    return DemoResult(status="INPUT_ERROR", message="Enter text or record audio first.")
                try:
                    utterance = (await self.transcribe(audio_path)).strip()
                    if not utterance:
                        raise ValueError("no speech was detected")
                except Exception:
                    return DemoResult(status="TRANSCRIPTION_ERROR",
                                      message="AssemblyAI transcription failed. Please try a WAV recording again.")
            try:
                evaluation = await self.session.spine.evaluate_semantics(utterance, final=True)
            except Exception:
                return DemoResult(transcript=utterance, status="SEMANTIC_ERROR",
                                  message="Semantic service failed. Please try again.")
            goal = evaluation.build.goal.describe() if evaluation.build else ""
            goal_fields = {}
            if evaluation.build:
                semantic_goal = evaluation.build.goal
                goal_fields = {
                    "capability": semantic_goal.capability,
                    "goal_type": semantic_goal.goal_type,
                    "site": str(semantic_goal.argument("site") or ""),
                    "target": str(semantic_goal.argument("target") or ""),
                }
            if evaluation.capability != "Browser" or evaluation.local_goal_type != "FIND":
                if evaluation.capability:
                    return DemoResult(utterance, goal, "UNSUPPORTED",
                                      "This action is available in the local desktop agent but is disabled in the hosted demo.",
                                      **goal_fields)
                return DemoResult(utterance, goal, "NOT_UNDERSTOOD", "Could not identify a supported Browser.FIND goal.")
            if not evaluation.build or not evaluation.build.complete:
                return DemoResult(utterance, goal, "INCOMPLETE", "Browser.FIND needs a target; try naming a site too.",
                                  **goal_fields)
            try:
                outcome = await self.session.spine.reconcile_final(evaluation)
            except Exception:
                return DemoResult(utterance, goal, "EXECUTION_ERROR", "Browser execution failed. Please try again.",
                                  **goal_fields)
            world = self.session.spine.world
            browser_status = world.get("browser.last_task_status")
            evidence = world.get("browser.last_task_evidence") or {}
            status = "SATISFIED" if outcome.satisfied else str(browser_status or (outcome.report.outcome.value if outcome.report else "UNCERTAIN")).upper()
            message = "Found it." if outcome.satisfied else "Browser.FIND could not verify a result."
            planner = " → ".join(record.step.operator.name for record in outcome.report.records) if outcome.report else ""
            return DemoResult(utterance, goal, status, message,
                              str(evidence.get("url") or ""), str(evidence.get("title") or ""),
                              planner=planner, **goal_fields)


class HostedRuntime:
    """Own the persistent browser and provider clients used by a Gradio process."""

    def __init__(self):
        backend = BackendSettings.from_env()
        if not os.environ.get("OPENROUTER_API_KEY", "").strip():
            raise RuntimeError("OPENROUTER_API_KEY is required for Browser.FIND")
        self.browser_find, self.browser_session, self.browser_provider, self.browser_jev_provider = build_browser_find(backend, headless=True)
        self.session = build_chat_session(settings=backend, execution=ExecutionSettings(SIMULATED),
                                          replies=SilentReplyComposer(), browser_find=self.browser_find)
        self.demo = HostedDemo(session=self.session, browser_find=self.browser_find)

    async def close(self):
        await aclose_jevs(self.session.spine.global_jev, self.session.slot_extractor)
        await self.browser_session.close(kill_browser=True)
        await self.browser_provider.aclose()
        await self.browser_jev_provider.aclose()
