# Browser evaluation

This is a small evaluation of the project's grounded browser executors. It
measures whether they reach a requested destination or a website's search results
page, with grounded completion verification. It does not measure speech
recognition, intent routing, slot extraction, or general web-search quality.

## Method

The task list is [`eval/find_tasks.json`](../eval/find_tasks.json): 24 destination
tasks and four site-search tasks. The runner builds a `Browser.FIND` or
`Browser.SEARCH_WEBSITE` goal directly for each case, gives both executors the
same site and target, and runs each task arm with its own browser setup. Browser
Use navigates to the site before its first policy step, matching Jev's site
setup. Arms run sequentially.

A **verified success** requires both the expected final URL and a `SATISFIED`
grounded completion verdict. For destination tasks, the URL must have the
expected host and path. For site-search tasks, it must have the expected search
path and query parameters (or a listed alternate URL). A matching URL with an
`UNCERTAIN` verdict is recorded separately. `DONE` from Jev alone is not counted
as success. The runner records status, final URL, duration, controller calls,
errors, and timing details in JSON.

The first four site-search cases are YouTube / PersonaPlex, GitHub / browser-use,
Hugging Face / NuExtract, and arXiv / full-duplex voice agents. Reddit was
removed after browser challenges made the result unsuitable for this comparison.

## Recorded results (27 September 2026)

| Run | Backend | Verified successes | Median arm time |
| --- | --- | ---: | ---: |
| Four site-search tasks, one round | Browser Use | 3/4 | 11.91 s |
| Four site-search tasks, one round | Jev Ultrafast | 3/4 | 7.11 s |
| Four site-search tasks, ten repeats each | Jev Ultrafast | 29/40 | 6.10 s |

In the paired round, Browser Use stopped on the YouTube home page. Jev reached
YouTube search results but opened a NuExtract3 model page on Hugging Face instead
of leaving the tab on NuExtract search results. Both completed the GitHub and
arXiv searches. Neither paired arm raised an execution error.

The ten-repeat Jev run broke down as follows:

| Site | Verified successes | Search URLs reached | Main observed limitation |
| --- | ---: | ---: | --- |
| YouTube | 7/10 | 10/10 | Three matching search pages received `UNCERTAIN` completion verdicts. |
| GitHub | 10/10 | 10/10 | None observed in this run. |
| Hugging Face | 2/10 | 2/10 | Eight runs opened the NuExtract3 model page instead of search results. |
| arXiv | 10/10 | 10/10 | None observed in this run. |

Across these four fixed site-search tasks, Jev recorded 29/40 verified successes
(72.5%). These were sequential repeats in one environment, so this result should
not be interpreted as an estimate of reliability across arbitrary sites or future
site changes.

The earlier 24-task destination round recorded Browser Use **21/24** and Jev
**24/24** verified successes, with median arm times of **12.81 s** and
**5.69 s**, respectively. That round used Jev fork revision
`f7acb2dd256d4e63a636c340b230969b39508fd5`; the site-search rounds used
`47b8bc16524d11fd91d557726f604838e29b766f`. The destination result is
historical evidence, not a same-revision comparison with the site-search run.

## Run the suite

Install the project with its browser extras, provide `OPENROUTER_API_KEY`, and
start Chrome CDP for Jev as described in the [Jev backend guide](jev-backend.md).
Then run from the repository root:

```bash
PYTHONPATH=src python -m scripts.run_find_eval --list --task-type site_search
PYTHONPATH=src python -m scripts.run_find_eval --task-type site_search --tasks 4 --timeout 120
PYTHONPATH=src python -m scripts.run_find_eval --task-type site_search --tasks 4 --backend jev_ultrafast --repeats 10 --timeout 120
```

For the destination set, use `--task-type destination --tasks 24`. The runner
checks that the installed Jev package matches the fork revision pinned in
`pyproject.toml`. Each run writes a timestamped report under `eval/results/`;
that folder is ignored by Git. Keep notable results and limitations in this
document rather than committing generated reports.
