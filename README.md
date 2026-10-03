# wepth-voice-agent

`wepth-voice-agent` is a voice-driven action agent that turns natural-language commands into an explicit, validated `SemanticGoal` before deciding how to act.

AssemblyAI provides speech-to-text. Global and Local Jev select the capability and goal schema, a structured extractor fills the required slots, and `GoalBuilder` validates the result. A deterministic planner then chooses the execution path.

The key idea is to separate **what the user wants** from **how the system executes it**.

Straightforward commands use deterministic operators. More open-ended browser tasks such as `Browser.FIND` and `Browser.SEARCH_WEBSITE` use a grounded Level 2 browser path.

## Architecture

```text
Voice / text command
        ↓
AssemblyAI STT
        ↓
Global Jev → Local Jev → GoalSchema
        ↓
Structured slot extraction
        ↓
GoalBuilder validation
        ↓
SemanticGoal
        ↓
DeterministicPlanner
        ↓
Direct capability operator
        OR
Grounded Browser.FIND / SEARCH_WEBSITE
        ↓
Executor / WorldState
        ↓
Result
```

The planner acts on the validated `SemanticGoal`, not directly on raw transcript text.

### Examples

| Command | Semantic goal | Execution path |
| --- | --- | --- |
| “Open GitHub” | `Browser.NAVIGATE` | Deterministic browser navigation |
| “Find the Browser Use repository on GitHub” | `Browser.FIND(site='GitHub', target='Browser Use repository')` | `FindWithGroundedBrowser` |
| “Search GitHub for browser-use” | `Browser.SEARCH_WEBSITE(site='GitHub', query='browser-use')` | `SearchWebsiteWithGroundedBrowser` |

The grounded browser path uses the existing `JevBrowserCapability` execution layer with either Browser Use or the optional Jev Ultrafast backend.

## Hosted demo

The judge-facing Gradio demo shows:

- the transcript;
- the constructed `SemanticGoal`;
- the planner operator;
- execution status;
- the final result URL.

The hosted demo supports voice or text input and executes `Browser.FIND` and `Browser.SEARCH_WEBSITE` in a server-side headless browser.

Example:

```text
Transcript:
Find the Browser Use repository on GitHub

Semantic Goal:
Capability: Browser
Goal: FIND
Site: GitHub
Target: Browser Use repository

Planner:
FindWithGroundedBrowser

Status:
SATISFIED

Result:
Found it.

Final URL:
https://github.com/browser-use/browser-use
```

The local runtime additionally supports realtime streaming microphone input, spoken acknowledgements and progress updates, TTS, Spotify control, and native browser actions.

## Setup

Requires Python 3.12+.

```bash
python -m venv .venv
source .venv/bin/activate

python -m pip install -e '.[voice,web,jev,test]'
playwright install chromium
```

Copy `.env.example` to `.env` and add your credentials.
The scripts read environment variables; they do not load `.env` automatically.
In a compatible shell, load it with `set -a; source .env; set +a` before running
the agent.

At minimum, the intended live path uses:

```env
ASSEMBLYAI_API_KEY=...
OPENROUTER_API_KEY=...

AGENTIC_JEV_BACKEND=live
AGENTIC_EXTRACTOR=llama
```

The default structured-extraction model is:

```text
meta-llama/llama-3.1-8b-instruct
```

Keep credentials out of Git.

## Run locally

### Realtime voice agent

```bash
python scripts/run_voice_agent.py --show-transcripts --verbose-control
```

Use `--text-output` to print responses instead of playing TTS.
For real local Spotify playback, set `AGENTIC_EXECUTION_BACKEND=desktop`;
`simulated` only updates in-memory state.

For a text-only control-path check:

```bash
python scripts/run_voice_agent.py \
  --text-input 'Find the Browser Use repository on GitHub' \
  --text-output \
  --verbose-control
```

### Gradio demo

```bash
python app.py --host 0.0.0.0 --port 7860
```

Then open:

```text
http://localhost:7860
```

Typed input bypasses transcription. Recorded microphone input uses AssemblyAI speech-to-text before entering the same semantic/control pipeline.

## Browser backends

`browser_use` is the default grounded browser backend.

An experimental `jev_ultrafast` backend is also available. See
[docs/jev-backend.md](docs/jev-backend.md) for setup and troubleshooting.

## Tests

```bash
python -m pytest -q tests
```

Most tests use mocks and do not require a microphone or live API credentials.

Desktop acceptance tests are skipped unless explicitly enabled.

## Evaluation

We evaluated the grounded browser path on destination-finding and site-search tasks across several public websites.

See [docs/evaluation.md](docs/evaluation.md) for the full methodology and results.
The task list is in `eval/find_tasks.json`. Generated JSON reports and logs go in
`eval/results/`, which Git ignores; keep the summary in `docs/evaluation.md`.

## Inspiration and acknowledgements

The initial idea of using Jev for low-latency, bounded intent and action selection was inspired by an Evolving AI repost credited to Andy Gao on X.

`wepth-voice-agent` extends that idea with:

- structured goal schemas;
- structured slot extraction;
- `GoalBuilder` validation;
- deterministic planning;
- `WorldState`;
- selective Level 2 browser execution;
- realtime voice/control orchestration.

Browser automation for the Level 2 `Browser.FIND` path builds on the open-source [Browser Use](https://github.com/browser-use/browser-use) project.

The optional browser backend builds on the external open-source [Jev Ultrafast](https://github.com/browser-use/jev-ultrafast) project. This repository evaluates a modified [Tiny-Small fork](https://github.com/Tiny-Small/jev-ultrafast) pinned in `pyproject.toml`.

## Current limits

- The hosted demo exposes grounded browser search/finding rather than desktop Spotify or native local-app control.
- Hosted microphone input is push-to-talk; the local CLI supports realtime streaming speech.
- Browser behavior depends on third-party websites whose markup and search results may change.
- This is a bounded action agent, not a general-purpose web-search service.
- A semantically correct destination can still be marked `UNCERTAIN` when the verifier lacks enough grounded evidence.

## License

This project is licensed under the [MIT License](LICENSE).
