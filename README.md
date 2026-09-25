# wepth-voice-agent

Real-time local voice agent. AssemblyAI streams microphone speech to a voice coordinator. Global and Local Jev choose a capability and goal schema; the configured extractor fills that schema's slots, and GoalBuilder validates the result. The deterministic planner and executor update WorldState. Browser navigation uses normal browser operators; Browser.FIND uses the same Level 2 `JevBrowserCapability` → `BrowserTaskExecutor` → `BrowserUseSessionAdapter` path. Fast Voice arbitrates acknowledgements, progress, and results before Fish Audio TTS plays them.

## Setup

Use Python 3.12 or newer. From this repository:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[voice,test]'
```

The `voice` extra installs AssemblyAI, sounddevice, and the pinned Browser Use version. Browser Use needs a browser available on the machine. On systems where it has not been installed yet, run `playwright install chromium`. Native browser and Spotify operators use the included PowerShell adapters on Windows; the default simulated backend is suitable for tests and control diagnostics.

Copy `.env.example` to `.env`, enter your keys, and load it into the shell. The script reads environment variables; it does not load `.env` automatically. `OPENROUTER_API_KEY` is needed for the default OpenRouter voice/TTS and live Jev, Llama extraction, and Browser.FIND controller. `ASSEMBLYAI_API_KEY` is needed for microphone input. Set `AGENTIC_JEV_BACKEND=live` and `AGENTIC_EXTRACTOR=llama` to use the intended live semantic path. `AGENTIC_LLAMA_MODEL` defaults to `meta-llama/llama-3.1-8b-instruct`. The browser controller defaults to `openai/gpt-4.1`; override it with `AGENTIC_LUNA_MODEL` or select `AGENTIC_BROWSER_CONTROLLER=llama`. Set `AGENTIC_EXECUTION_BACKEND=desktop` for native direct browser actions on Windows; the default is `simulated`.

For example, load a local `.env` with `set -a; source .env; set +a` in a compatible shell. Keep credentials out of Git.

## Run

```bash
python scripts/run_voice_agent.py --show-transcripts --verbose-control
```

Use `--text-output` to print replies instead of playing TTS. Use `--text-input 'Open GitHub' --text-output --verbose-control` to diagnose semantics and control without a microphone. A second `--text-input` sends another final turn sequentially. `--progress-silence-seconds` defaults to `2.5`. `--help` lists audio device, STT silence, model, and endpoint options.

With `--show-transcripts`, expect `Final: Open GitHub`; with `--verbose-control`, inspect the extracted goal and planner/executor events for `Browser.NAVIGATE`. A quick acknowledged navigation should not produce stale progress or a generic Done. For `Find the Browser Use repository on GitHub`, expect a Browser.FIND goal, Level 2 browser decisions, and a final result such as `Found it.` when completion is verified. The browser target is `https://github.com/browser-use/browser-use`. If a newer turn or revision arrives, old FIND work and its pending speech are cancelled. Logs can expose spoken text and target URLs; do not share them without review.

## Tests

```bash
python -m pytest -q tests
```

Desktop acceptance tests are skipped unless `AGENTIC_DESKTOP_TESTS=1`. Most runtime tests use mocks and do not need a microphone or credentials.
