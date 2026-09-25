#!/usr/bin/env python3
"""Run the local microphone → AssemblyAI → concurrent control voice slice."""

from __future__ import annotations

import asyncio
import logging
import os
import sys

import httpx

from ping_ponder.agentic.reply import SilentReplyComposer
from ping_ponder.agentic.browser_use_slice import (
    BrowserTaskExecutor, BrowserUseSessionAdapter, JevBrowserActionChooser,
    JevBrowserCapability, JevBrowserCompletionVerifier, ModelBrowserController,
)
from ping_ponder.agentic.execution_config import ExecutionSettings
from ping_ponder.agentic.wiring import aclose_jevs, build_chat_session
from ping_ponder.providers.openrouter import OpenRouterProvider
from ping_ponder.providers.openrouter_decisions import OpenRouterDecisionsProvider
from ping_ponder.voice.assemblyai_stt import (
    AssemblyAIRecognizerSettings,
    AssemblyAIStreamingRecognizer,
)
from ping_ponder.voice.audio import SoundDeviceAudioInput, SoundDeviceAudioOutput
from ping_ponder.voice.cli import (
    TranscriptConsoleObserver,
    configure_control_logging,
    parser,
    resolve_backend_settings,
)
from ping_ponder.voice.coordinator import CoordinatorSettings
from ping_ponder.voice.fast_voice import OpenAICompatibleFastVoice, ParallelVoiceSettings
from ping_ponder.voice.fish_audio import FISH_AUDIO_ENDPOINT, FishAudioSynthesizer
from ping_ponder.voice.events import FinalTranscript, TranscriptState
from ping_ponder.voice.wiring import build_voice_runtime


class TextOutputSynthesizer:
    """Print speech text when --text-output is selected."""

    async def synthesize_stream(self, text):
        utterance = "".join([chunk async for chunk in text]).strip()
        if utterance:
            print(f"Agent: {utterance}", flush=True)
        if False:  # keep this an async generator while intentionally yielding no PCM
            yield None

    async def stop(self):
        return None


def build_browser_find(backend):
    """Compose the same Level 2 executor and browser surface as the FIND demo."""
    jev_provider = OpenRouterDecisionsProvider()
    browser_provider = OpenRouterProvider()
    browser_session = BrowserUseSessionAdapter()
    capability = JevBrowserCapability(BrowserTaskExecutor(
        browser_session,
        ModelBrowserController(browser_provider, model=backend.browser_controller_model),
        max_decisions=8,
        completion_verifier=JevBrowserCompletionVerifier(
            jev_provider, model=backend.local_jev_model),
        action_chooser=JevBrowserActionChooser(
            jev_provider, model=backend.local_jev_model, formulation="baseline"),
        initial_action_candidates=True,
        native_site_jev=True,
    ))
    return capability, browser_session, browser_provider, jev_provider


async def run(args) -> None:
    configure_control_logging(args.verbose_control)
    api_key = os.environ.get("ASSEMBLYAI_API_KEY", "").strip()
    if not api_key and not args.text_input:
        raise SystemExit("ASSEMBLYAI_API_KEY is required")
    openrouter_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    fast_endpoint = args.fast_voice_endpoint or os.environ.get(
        "FAST_VOICE_ENDPOINT", "https://openrouter.ai/api/v1/chat/completions")
    tts_endpoint = args.tts_endpoint or os.environ.get("FISH_AUDIO_ENDPOINT", FISH_AUDIO_ENDPOINT)
    if (fast_endpoint.startswith("https://openrouter.ai/") or
            tts_endpoint.startswith("https://openrouter.ai/")) and not openrouter_key:
        raise SystemExit("OPENROUTER_API_KEY is required for the configured OpenRouter endpoint")
    audio_input = None if args.text_input else SoundDeviceAudioInput(
        sample_rate=args.sample_rate, device=args.input_device,
    )
    recognizer = None if args.text_input else AssemblyAIStreamingRecognizer(
        api_key=api_key,
        settings=AssemblyAIRecognizerSettings(
            sample_rate=args.sample_rate,
            min_turn_silence=args.min_turn_silence,
            max_turn_silence=args.max_turn_silence,
        ),
        on_ready=lambda: print(
            "Ready — listening. Press Ctrl+C to stop.", flush=True),
    )
    backend = resolve_backend_settings(args)
    execution = ExecutionSettings.from_env()
    browser_find, browser_session, browser_provider, browser_jev_provider = build_browser_find(backend)
    session = build_chat_session(
        settings=backend, execution=execution, replies=SilentReplyComposer(),
        browser_find=browser_find)
    if args.verbose_control:
        logging.getLogger("ping_ponder.voice.cli").info(
            "voice_agent_configuration jev=%s extractor=%s device=%s execution=%s",
            backend.jev_backend, backend.extractor, backend.extractor_device,
            execution.backend,
        )
    dialogue_client = httpx.AsyncClient()
    fast_voice = OpenAICompatibleFastVoice(
        endpoint=fast_endpoint,
        client=dialogue_client,
        api_key=openrouter_key,
    )
    synthesizer = TextOutputSynthesizer() if args.text_output else FishAudioSynthesizer(
        client=dialogue_client,
        api_key=openrouter_key,
        endpoint=tts_endpoint,
        model=args.tts_model,
        voice=args.fish_audio_voice or os.environ.get("FISH_AUDIO_VOICE"),
    )
    audio_output = None if args.text_output else SoundDeviceAudioOutput(
        sample_rate=None, device=args.output_device)
    runtime = build_voice_runtime(
        spine=session.spine,
        audio_input=audio_input,
        recognizer=recognizer,
        audio_output=audio_output,
        fast_voice=fast_voice,
        synthesizer=synthesizer,
        event_observer=TranscriptConsoleObserver() if args.show_transcripts else None,
        settings=CoordinatorSettings(
            partial_debounce_ms=args.partial_debounce_ms),
        parallel_settings=ParallelVoiceSettings(
            progress_silence_s=args.progress_silence_seconds),
    )
    try:
        if args.text_input:
            await runtime.start()
            try:
                for number, utterance in enumerate(args.text_input, start=1):
                    runtime.dispatch(FinalTranscript(TranscriptState(
                        f"text-{number}", 1, utterance, True)))
                    await runtime.wait_idle()
            finally:
                await runtime.stop()
        else:
            await runtime.run()
    finally:
        await dialogue_client.aclose()
        await aclose_jevs(session.spine.global_jev, session.slot_extractor)
        await browser_session.close(kill_browser=True)
        await browser_provider.aclose()
        await browser_jev_provider.aclose()


def main() -> int:
    args = parser().parse_args()
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
