import subprocess
import sys

import httpx

from ping_ponder.voice.cli import (
    TranscriptConsoleObserver,
    build_dialogue_model,
    parser,
    resolve_backend_settings,
)
from ping_ponder.voice.events import FinalTranscript, PartialTranscript, TranscriptState


def test_voice_cli_help_is_available_without_credentials_or_audio_devices():
    result = subprocess.run(
        [sys.executable, "scripts/run_voice_agent.py", "--help"],
        text=True, capture_output=True, check=False,
    )

    assert result.returncode == 0
    assert "--partial-debounce-ms" in result.stdout
    assert "--text-output" in result.stdout
    assert "--show-transcripts" in result.stdout


def test_voice_cli_wires_configured_jev_and_dialogue_backends():
    args = parser().parse_args([
        "--text-output",
        "--jev-backend", "live",
        "--global-jev-model", "global-model",
        "--local-jev-model", "local-model",
        "--dialogue-endpoint", "https://example.test/v1/chat/completions",
        "--dialogue-model", "dialogue-model",
    ])
    backend = resolve_backend_settings(args, environ={})
    client = httpx.AsyncClient()
    try:
        dialogue = build_dialogue_model(args, client=client, environ={"DIALOGUE_API_KEY": "key"})
    finally:
        import asyncio
        asyncio.run(client.aclose())

    assert backend.live
    assert backend.global_jev_model == "global-model"
    assert backend.local_jev_model == "local-model"
    assert dialogue.model == "dialogue-model"


def test_show_transcripts_flag_and_console_observer(capsys):
    args = parser().parse_args(["--text-output", "--show-transcripts"])
    observer = TranscriptConsoleObserver()

    observer(PartialTranscript(TranscriptState("t1", 1, "Play some", False)))
    observer(FinalTranscript(TranscriptState("t1", 2, "Play some jazz", True)))

    assert args.show_transcripts is True
    assert capsys.readouterr().out.splitlines() == [
        "Partial: Play some",
        "Final: Play some jazz",
    ]
