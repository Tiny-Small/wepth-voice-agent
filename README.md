# wepth-voice-agent

`wepth-voice-agent` is a voice-driven action agent that turns speech into an explicit, validated `SemanticGoal` before choosing how to act. AssemblyAI supplies speech-to-text; Global and Local Jev select the capability and goal schema, structured extraction fills its slots, and GoalBuilder validates the result. `DeterministicPlanner` then selects capability-specific operators. Straightforward commands use direct operators; open-ended `Browser.FIND` tasks use the Level 2 browser path.

The intermediate goal separates **what the user wants** from **how the system executes it**. The planner uses WorldState to choose an action, and execution updates that state so the outcome can be observed. The local runtime also provides realtime streaming voice interaction and spoken acknowledgements, progress updates, and results.

## Architecture

```text
Voice / text command
        ↓
AssemblyAI STT (voice input)
        ↓
Global Jev → Local Jev → GoalSchema
        ↓
Structured slot extraction → GoalBuilder validation
        ↓
SemanticGoal → DeterministicPlanner
        ↓
Direct capability operator OR Level 2 Browser.FIND operator
        ↓
Executor / WorldState
        ↓
Result
```

The planner does not act on raw transcript text. For example:

| Command | Semantic goal | Planner operator / path |
| --- | --- | --- |
| “Open GitHub” | `Browser.NAVIGATE` | `OpenBrowser` → `NavigateURL` from a closed browser; `NavigateURL` when already open |
| “Find the Browser Use repository on GitHub” | `Browser.FIND(site='GitHub', target='Browser Use repository')` | `FindWithGroundedBrowser` (Level 2 browser execution) |

`Browser.FIND` extends the same goal and planning flow with `JevBrowserCapability` → `BrowserTaskExecutor` → `BrowserUseSessionAdapter` when the task needs grounded browser interaction.

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

With `--show-transcripts`, expect `Final: Open GitHub`; with `--verbose-control`, inspect the constructed goal and planner/executor events for `Browser.NAVIGATE`. For `Find the Browser Use repository on GitHub`, expect a `Browser.FIND` goal, Level 2 browser decisions, and a final result such as `Found it.` when completion is verified. The browser target is `https://github.com/browser-use/browser-use`. If a newer turn or revision arrives, old FIND work and its pending speech are cancelled. Logs can expose spoken text and target URLs; review them before sharing.

## Tests

```bash
python -m pytest -q tests
```

Desktop acceptance tests are skipped unless `AGENTIC_DESKTOP_TESTS=1`. Most runtime tests use mocks and do not need a microphone or credentials.

## Hosted web demo

Install with `python -m pip install -e '.[voice,web]'`, then install Chromium with
`playwright install chromium` (and its Linux system libraries if the image needs them).
Load `.env` into the shell as above, then run:

```bash
python app.py --host 0.0.0.0 --port 7860
```

The port defaults to 7860 and can also be set with `GRADIO_SERVER_PORT`.
Set `OPENROUTER_API_KEY`, `AGENTIC_JEV_BACKEND=live`, and
`AGENTIC_EXTRACTOR=llama` for the intended semantic path. Set
`ASSEMBLYAI_API_KEY` to use recorded microphone or uploaded audio; typed text
bypasses transcription. The browser runs headless on the server. This judge-facing
demo shows the transcript, structured goal, selected planner operator, status,
and result URL for `Browser.FIND`. It serializes submissions through one browser
session. The local runtime additionally supports realtime streaming voice
interaction, spoken acknowledgements/progress/results, and desktop-only
capabilities such as Spotify and native browser actions.

## Inspiration and acknowledgements

The initial idea of using Jev for low-latency, bounded intent and action selection
came from [Evolving AI Instagram post](https://www.instagram.com/reel/DdjIOcstTtn/)
that featured a video credited as “Source: X / Andy Gao.” `wepth-voice-agent`
adds structured goal schemas, slot extraction, GoalBuilder validation, deterministic
planning, WorldState, selective Level 2 browser execution, and realtime voice
and control orchestration.

Browser automation for the Level 2 `Browser.FIND` path builds on the external
open-source [Browser Use](https://github.com/browser-use/browser-use) project,
consumed here as the pinned `browser-use==0.13.10` dependency.

### Experimental Jev Ultrafast Level 2 backend

`BROWSER_BACKEND=browser_use` remains the default. To evaluate the alternative,
install `pip install '.[voice,jev]'` and set `BROWSER_BACKEND=jev_ultrafast`.
The existing `OPENROUTER_API_KEY` sends Jev's policy choices through OpenRouter
Decisions using `AGENTIC_LOCAL_JEV_MODEL`, and also supplies its `TYPE_TEXT`
helper through OpenRouter's chat completions endpoint. If `TYPESAFE_API_KEY` is
set, policy choices use upstream TypeSafe directly instead; `TYPESAFE_MODEL`
then controls that provider. The default `TYPE_TEXT` model is
`inception/mercury-2.5`, using Jev's low reasoning setting for its JSON
response. Optional text overrides are `TEXT_MODEL_API_KEY`,
`TEXT_MODEL_BASE_URL`, `TEXT_MODEL`, and `TEXT_MODEL_REASONING`. The `jev` extra
installs the Jev package from the pinned `Tiny-Small/jev-ultrafast` fork commit
`98d45b6e91565468f343d9e3bdf5de0cf7632b15`. That branch adds an observed
`PRESS_ENTER` action for populated search fields so sites without a visible
submit control can run a search. The fork also supplies Jev's Browser Harness
and HTTP/2 dependencies; this project does not copy Jev's core source.

The `FIND` operator and grounded completion verifier remain shared. Jev's
`DONE` only stops its action loop. The adapter observes the final tab and asks
the existing verifier; a `NOT_SATISFIED` or `UNCERTAIN` answer cannot become
success. No automatic fallback to Browser Use occurs. Jev owns a Browser
Harness tab in an existing Chrome profile and does not attach to the Browser
Use session; its final URL is recorded in the task result and the tab remains
open until the runtime closes the backend. Its synchronous requests run in a
worker thread, so cancellation stops before the next action, but an already
running browser action cannot be interrupted. Live comparison requires Chrome
with Browser Harness connected and an OpenRouter key. The policy redirect is
scoped to each Jev prediction and restores the package transport afterward.
