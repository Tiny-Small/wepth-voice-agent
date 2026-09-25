"""Configuration helpers for the local voice command-line runtime."""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import replace
from typing import Mapping

import httpx

from ping_ponder.agentic.backend_config import (
    VALID_EXTRACTORS,
    VALID_JEV_BACKENDS,
    BackendSettings,
)

from .dialogue import OpenAICompatibleDialogueModel
from .events import FinalTranscript, PartialTranscript, SpeechEvent


def configure_control_logging(enabled: bool) -> None:
    """Show structured application diagnostics without enabling library chatter."""
    if not enabled:
        return
    logger = logging.getLogger("ping_ponder")
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())
    logger.propagate = False


class TranscriptConsoleObserver:
    """Print provider-neutral transcript checkpoints after queue dispatch."""

    def __call__(self, event: SpeechEvent) -> None:
        if isinstance(event, PartialTranscript):
            print(f"Partial: {event.state.text}", flush=True)
        elif isinstance(event, FinalTranscript):
            print(f"Final: {event.state.text}", flush=True)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Local AssemblyAI realtime voice-control architecture demo",
    )
    result.add_argument(
        "--text-output", action="store_true",
        help="print speech text instead of playing Fish Audio through the speaker",
    )
    result.add_argument(
        "--text-input", action="append", default=[],
        help="submit a final transcript without a microphone; repeat for sequential turns",
    )
    result.add_argument(
        "--show-transcripts", action="store_true",
        help="print each partial and final AssemblyAI transcript checkpoint",
    )
    result.add_argument(
        "--verbose-control", action="store_true",
        help="show structured Jev, extraction, planning, and executor diagnostics",
    )
    result.add_argument("--input-device", help="sounddevice microphone identifier")
    result.add_argument("--output-device", help="sounddevice speaker identifier")
    result.add_argument("--sample-rate", type=int, default=16_000)
    result.add_argument("--partial-debounce-ms", type=int, default=0)
    result.add_argument("--min-turn-silence", type=int)
    result.add_argument("--max-turn-silence", type=int)
    result.add_argument("--jev-backend", choices=VALID_JEV_BACKENDS)
    result.add_argument("--global-jev-model")
    result.add_argument("--local-jev-model")
    result.add_argument("--extractor", choices=VALID_EXTRACTORS)
    result.add_argument("--dialogue-endpoint")
    result.add_argument("--dialogue-model")
    result.add_argument("--dialogue-api-key-env", default="DIALOGUE_API_KEY")
    result.add_argument("--fast-voice-endpoint", default=None,
                        help="OpenAI-compatible Llama chat-completions endpoint")
    result.add_argument("--progress-silence-seconds", type=float, default=2.5)
    result.add_argument("--tts-model", default="fish-audio/s2.1-pro-free:free")
    result.add_argument("--tts-endpoint", default=None,
                        help="OpenAI-compatible TTS endpoint; defaults to OpenRouter")
    result.add_argument("--fish-audio-voice", default=None,
                        help="optional Fish Audio voice identifier")
    return result


def resolve_backend_settings(args: argparse.Namespace, *,
                             environ: Mapping[str, str] | None = None) -> BackendSettings:
    env = dict(os.environ if environ is None else environ)
    settings = BackendSettings.from_env(env)
    return replace(
        settings,
        jev_backend=args.jev_backend or settings.jev_backend,
        global_jev_model=args.global_jev_model or settings.global_jev_model,
        local_jev_model=args.local_jev_model or settings.local_jev_model,
        extractor=args.extractor or settings.extractor,
    )


def build_dialogue_model(args: argparse.Namespace, *, client: httpx.AsyncClient,
                         environ: Mapping[str, str] | None = None):
    env = os.environ if environ is None else environ
    endpoint = args.dialogue_endpoint or env.get("VOICE_DIALOGUE_ENDPOINT")
    model = args.dialogue_model or env.get("VOICE_DIALOGUE_MODEL")
    if not endpoint and not model:
        return None
    if not endpoint or not model:
        raise ValueError("dialogue endpoint and model must be configured together")
    return OpenAICompatibleDialogueModel(
        endpoint=endpoint,
        model=model,
        client=client,
        api_key=env.get(args.dialogue_api_key_env),
    )
