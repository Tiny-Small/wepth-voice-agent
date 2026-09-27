# Jev Ultrafast browser backend

`browser_use` is the default grounded browser backend. The experimental
`jev_ultrafast` backend uses [Jev Ultrafast](https://github.com/browser-use/jev-ultrafast)
through a modified [Tiny-Small fork](https://github.com/Tiny-Small/jev-ultrafast)
pinned in [`pyproject.toml`](../pyproject.toml). The evaluated site-search revision
is `47b8bc16524d11fd91d557726f604838e29b766f`.

The local CLI starts Browser Use headless by default, so its page interactions do
not appear in a desktop window. To select Jev instead, set:

```env
BROWSER_BACKEND=jev_ultrafast
```

## Visible Chrome CDP session

To watch Jev work, start a dedicated visible Chromium CDP session from your own
WSL desktop terminal (`echo "$DISPLAY"` should print a value). Keep that terminal
and the Chrome process running while you use the voice agent. Use a fresh port
and profile if another Chrome is already running headless on port 9337:

```bash
CHROME_BIN="$(python -c 'from playwright.sync_api import sync_playwright; p = sync_playwright().start(); print(p.chromium.executable_path); p.stop()')"
"$CHROME_BIN" --ozone-platform=x11 --remote-debugging-port=9338 \
  --user-data-dir=/tmp/jev-visible-chrome-profile \
  --no-first-run --no-default-browser-check
```

In another terminal, select the same CDP endpoint and check that Chrome is ready.
If you previously used the `jev_visible` Browser Harness daemon with a different
CDP port, reload it before starting the voice agent:

```bash
export BU_CDP_URL=http://127.0.0.1:9338
curl -fsS "$BU_CDP_URL/json/version" >/dev/null && echo 'Chrome ready'
BU_NAME=jev_visible browser-harness --reload
```

Browser Harness connects to this running Chrome session when Jev starts a browser
task. `BU_NAME=jev_visible` gives it a separate daemon from any daemon already
connected to port 9337.

## Run a spoken task

To watch a spoken GitHub request in the dedicated Chrome session and print the
STT transcript:

```bash
BU_NAME=jev_visible BU_CDP_URL=http://127.0.0.1:9338 \
AGENTIC_JEV_BACKEND=live AGENTIC_EXTRACTOR=llama \
AGENTIC_EXECUTION_BACKEND=desktop BROWSER_BACKEND=jev_ultrafast \
PYTHONPATH=src python scripts/run_voice_agent.py \
  --text-output --show-transcripts --verbose-control
```

Jev deliberately opens a background tab in that Chrome profile, so Chrome does
not switch to it; select the new tab to watch it. The tab closes when the agent
process exits. `--show-transcripts` prints `Partial:` and `Final:` STT results;
`--text-output` prints the agent's responses. One-shot `--text-input` is useful
for a typed diagnostic, but it closes Jev's tab when the process exits.

## Troubleshooting

If the named daemon fails to start, inspect
`~/.config/browser-harness/tmp/bu-jev_visible.log`. For the default daemon, use
`bu-default.log` in that directory. Confirm that its connected CDP port matches
`BU_CDP_URL`; reload the named daemon after changing the endpoint.

A successful `FIND` emits `Browser.FIND` and `FindWithGroundedBrowser` in the
control log, followed by `Found it.` A final `Done.` indicates that the run
selected a different goal type; check the `semantic_goal_built` and
`executor_step_start` events to see what ran.
