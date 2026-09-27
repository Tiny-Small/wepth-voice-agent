"""The FIND evaluation must keep answer keys out of both browser executors."""

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from ping_ponder.agentic.browser_use_slice import (
    BrowserCompletion,
    BrowserCompletionStatus,
    BrowserObservation,
    BrowserTaskResult,
)
from ping_ponder.agentic.find_eval import (
    FindCase,
    PreparedBrowserUseExecutor,
    SearchCase,
    evaluate,
    load_cases,
    score_destination,
    score_search_url,
)
from scripts import run_find_eval
from scripts.run_find_eval import jev_source_matches_pin, validate_resume_report


def test_case_goal_excludes_the_expected_url():
    case = FindCase("repo", "repository", "GitHub", "Browser Use repository",
                    "https://github.com/browser-use/browser-use")

    goal = case.goal()

    assert goal.capability == "Browser"
    assert goal.goal_type == "FIND"
    assert dict(goal.arguments) == {"site": "GitHub", "target": "Browser Use repository"}


@pytest.mark.parametrize(
    ("actual", "expected", "matches"),
    [
        ("https://github.com/browser-use/browser-use/?ref=search#readme",
         "https://github.com/browser-use/browser-use", True),
        ("https://github.com/other/browser-use", "https://github.com/browser-use/browser-use", False),
        ("https://github.com/browser-use/browser-use/issues",
         "https://github.com/browser-use/browser-use", False),
        ("https://evil.example/browser-use/browser-use",
         "https://github.com/browser-use/browser-use", False),
    ],
)
def test_destination_scoring_requires_the_right_host_and_path(actual, expected, matches):
    assert score_destination(actual, expected) is matches


def test_site_search_goal_keeps_answer_key_out_of_both_executors():
    case = SearchCase("youtube-personaplex", "site_search", "YouTube", "PersonaPlex",
                      "https://www.youtube.com/results?search_query=PersonaPlex")

    goal = case.goal()

    assert goal.goal_type == "SEARCH_WEBSITE"
    assert dict(goal.arguments) == {"site": "YouTube", "query": "PersonaPlex"}


@pytest.mark.asyncio
async def test_site_search_accepts_an_alternate_results_route_without_leaking_it_to_goal():
    case = SearchCase(
        "arxiv-full-duplex-voice", "site_search", "arxiv.org", "full-duplex voice agents",
        "https://arxiv.org/search/?query=full-duplex+voice+agents",
        alternate_urls=("https://arxiv.org/search/advanced?terms-0-term=full-duplex+voice+agents",),
    )
    assert "expected_url" not in case.goal().arguments
    assert "alternate_urls" not in case.goal().arguments

    class Executor:
        async def execute(self, goal):
            observation = BrowserObservation(
                "results", "https://arxiv.org/search/advanced?advanced=1&terms-0-term=full-duplex+voice+agents",
                "Advanced Search | arXiv", "Showing 1–6 of 6 results", frozenset(), "results")
            return BrowserTaskResult(BrowserCompletion(BrowserCompletionStatus.SATISFIED), observation)

    class Resource:
        async def close(self, *, kill_browser=False):
            pass

    report = await evaluate([case], ["jev_ultrafast"],
                            build=lambda _: (Executor(), Resource(), None, None))

    assert report["results"][0]["search_reached"] is True
    assert report["results"][0]["success"] is True


@pytest.mark.parametrize(
    ("actual", "matches"),
    [
        ("https://www.youtube.com/results?search_query=PersonaPlex&sp=CAI%253D", True),
        ("https://www.youtube.com/results?search_query=Other", False),
        ("https://www.youtube.com/watch?v=PersonaPlex", False),
        ("https://www.google.com/search?q=PersonaPlex", False),
        ("https://www.youtube.com/results?search_query=PersonaPlex&js_challenge=1", False),
    ],
)
def test_site_search_scoring_requires_site_results_and_query(actual, matches):
    expected = "https://www.youtube.com/results?search_query=PersonaPlex"

    assert score_search_url(actual, expected) is matches


def test_load_cases_rejects_duplicate_ids(tmp_path):
    path = tmp_path / "tasks.json"
    item = {"id": "same", "category": "repository", "site": "GitHub",
            "target": "Browser Use repository", "expected_url": "https://github.com/browser-use/browser-use"}
    path.write_text(json.dumps([item, item]))

    with pytest.raises(ValueError, match="duplicate"):
        load_cases(path)


def test_load_cases_rejects_search_without_query_in_answer_url(tmp_path):
    path = tmp_path / "tasks.json"
    path.write_text(json.dumps([{
        "id": "search", "task_type": "site_search", "category": "site_search",
        "site": "GitHub", "target": "browser-use", "expected_url": "https://github.com/search",
    }]))

    with pytest.raises(ValueError, match="query"):
        load_cases(path)


def test_bundled_cases_are_distinct_and_hold_out_answer_keys():
    path = Path(__file__).resolve().parents[1] / "eval" / "find_tasks.json"
    cases = load_cases(path)

    assert len(cases) == 28
    assert {case.category for case in cases} == {
        "repository", "python_docs", "mdn_docs", "encyclopedia", "site_search"}
    assert len([case for case in cases if isinstance(case, SearchCase)]) == 4
    assert all(case.id != "reddit-runpod-personaplex" for case in cases)
    assert len({case.expected_url for case in cases}) == len(cases)
    assert all("expected_url" not in case.goal().arguments for case in cases)


def test_cli_can_list_cases_without_credentials():
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, "-m", "scripts.run_find_eval", "--list"],
        capture_output=True, text=True, check=False,
        cwd=root, env={**os.environ, "PYTHONPATH": str(root / "src")},
    )

    assert completed.returncode == 0, completed.stderr
    assert "28 tasks" in completed.stdout
    assert "repository" in completed.stdout


def test_cli_lists_only_site_search_tasks_without_credentials():
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, "-m", "scripts.run_find_eval", "--list", "--task-type", "site_search"],
        capture_output=True, text=True, check=False,
        cwd=root, env={**os.environ, "PYTHONPATH": str(root / "src")},
    )

    assert completed.returncode == 0, completed.stderr
    assert "4 tasks" in completed.stdout
    assert "youtube-personaplex" in completed.stdout
    assert "repo-browser-use" not in completed.stdout


def test_jev_source_check_rejects_upstream_package_with_same_version():
    pin = ("jev-ultrafast @ git+https://github.com/Tiny-Small/jev-ultrafast.git"
           "@98d45b6e91565468f343d9e3bdf5de0cf7632b15")
    upstream = json.dumps({"url": "https://github.com/browser-use/jev-ultrafast.git",
                           "vcs_info": {"commit_id": "1231850a0bf1a0c0341fe408ef1668dbbfdfac46"}})
    fork = json.dumps({"url": "https://github.com/Tiny-Small/jev-ultrafast.git",
                       "vcs_info": {"commit_id": "98d45b6e91565468f343d9e3bdf5de0cf7632b15"}})

    assert not jev_source_matches_pin(upstream, pin)
    assert jev_source_matches_pin(fork, pin)
    assert not jev_source_matches_pin(None, pin)


def test_cli_rejects_wrong_jev_build_before_opening_browser(monkeypatch, capsys):
    class Distribution:
        def read_text(self, name):
            assert name == "direct_url.json"
            return json.dumps({"url": "https://github.com/browser-use/jev-ultrafast.git",
                               "vcs_info": {"commit_id": "1231850a0bf1a0c0341fe408ef1668dbbfdfac46"}})

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(run_find_eval.metadata, "distribution", lambda name: Distribution())

    with pytest.raises(SystemExit) as error:
        run_find_eval.main(["--tasks", "1", "--backend", "jev_ultrafast"])

    assert error.value.code == 2
    assert "--force-reinstall" in capsys.readouterr().err


def test_resume_report_rejects_changed_run_settings():
    config = {"task_file_sha256": "abc", "task_ids": ["one", "two"],
              "task_type": "site_search", "backends": ["browser_use", "jev_ultrafast"], "repeats": 1,
              "timeout_seconds": 90.0, "local_jev_model": "jev",
              "browser_controller_model": "luna", "jev_source": {"commit": "fork"},
              "keep_final_jev_tab_open": False}
    report = {"complete": False, "configuration": config, "results": [{"case_id": "one"}]}

    assert validate_resume_report(report, config) == report["results"]
    with pytest.raises(ValueError, match="task_file_sha256"):
        validate_resume_report(report, {**config, "task_file_sha256": "changed"})
    with pytest.raises(ValueError, match="task_type"):
        validate_resume_report(report, {**config, "task_type": "destination"})
    with pytest.raises(ValueError, match="already complete"):
        validate_resume_report({**report, "complete": True}, config)


@pytest.mark.asyncio
async def test_evaluate_runs_both_backends_with_fresh_resources_and_separate_scores():
    case = FindCase("repo", "repository", "GitHub", "Browser Use repository",
                    "https://github.com/browser-use/browser-use")
    seen_goals = []
    closed = []

    class Executor:
        def __init__(self, backend):
            self.backend = backend

        async def execute(self, goal):
            seen_goals.append(dict(goal.arguments))
            url = case.expected_url if self.backend == "browser_use" else "https://github.com/other/browser-use"
            observation = BrowserObservation("page", url, "repository", "repository", frozenset(), "page")
            return BrowserTaskResult(
                BrowserCompletion(BrowserCompletionStatus.SATISFIED, {"url": url}),
                observation, controller_calls=2,
            )

    class Resource:
        def __init__(self, backend):
            self.backend = backend

        async def close(self, *, kill_browser=False):
            closed.append((self.backend, kill_browser))

    def build(backend):
        resource = Resource(backend)
        return Executor(backend), resource, None, None

    result = await evaluate([case], ["browser_use", "jev_ultrafast"], build=build)

    assert seen_goals == [{"site": "GitHub", "target": "Browser Use repository"}] * 2
    assert closed == [("browser_use", True), ("jev_ultrafast", True)]
    assert [row["success"] for row in result["results"]] == [True, False]
    assert [row["destination_reached"] for row in result["results"]] == [True, False]
    assert result["summary"]["browser_use"]["successes"] == 1
    assert result["summary"]["jev_ultrafast"]["successes"] == 0


@pytest.mark.asyncio
async def test_evaluate_site_search_counts_search_without_claiming_a_destination():
    case = SearchCase("search", "site_search", "GitHub", "browser-use",
                      "https://github.com/search?q=browser-use")

    class Executor:
        async def execute(self, goal):
            assert goal.goal_type == "SEARCH_WEBSITE"
            observation = BrowserObservation(
                "results", "https://github.com/search?q=browser-use&type=repositories",
                "Search results", "browser-use repositories", frozenset(), "results")
            return BrowserTaskResult(BrowserCompletion(BrowserCompletionStatus.SATISFIED), observation)

    class Resource:
        async def close(self, *, kill_browser=False):
            pass

    result = await evaluate([case], ["browser_use"],
                            build=lambda _: (Executor(), Resource(), None, None))

    assert result["results"][0]["search_reached"] is True
    assert result["results"][0]["destination_reached"] is False
    assert result["results"][0]["success"] is True
    assert result["summary"]["browser_use"]["searches_reached"] == 1
    assert result["summary"]["browser_use"]["destinations_reached"] == 0


@pytest.mark.asyncio
async def test_site_search_report_omits_challenge_tokens_from_final_url():
    case = SearchCase("search", "site_search", "reddit.com", "RunPod PersonaPlex",
                      "https://www.reddit.com/search/?q=RunPod+PersonaPlex")

    class Executor:
        async def execute(self, goal):
            observation = BrowserObservation(
                "challenge",
                "https://www.reddit.com/search/?jsc_token=secret&q=RunPod+PersonaPlex&solution=opaque",
                "Challenge", "verify browser", frozenset(), "challenge")
            return BrowserTaskResult(BrowserCompletion(BrowserCompletionStatus.BLOCKED), observation)

    class Resource:
        async def close(self, *, kill_browser=False):
            pass

    result = await evaluate([case], ["browser_use"],
                            build=lambda _: (Executor(), Resource(), None, None))

    assert result["results"][0]["search_reached"] is False
    assert result["results"][0]["success"] is False
    assert result["results"][0]["final_url"] == (
        "https://www.reddit.com/search/?q=RunPod+PersonaPlex")


@pytest.mark.asyncio
async def test_evaluate_announces_completion_before_cleanup_and_can_leave_final_jev_tab_open():
    cases = [
        FindCase("one", "repository", "GitHub", "One", "https://github.com/a/one"),
        FindCase("two", "repository", "GitHub", "Two", "https://github.com/a/two"),
    ]
    events = []

    class Executor:
        def __init__(self, backend):
            self.backend = backend

        async def execute(self, goal):
            case = next(case for case in cases if case.target == goal.argument("target"))
            observation = BrowserObservation("page", case.expected_url, case.target, case.target,
                                             frozenset(), "page")
            return BrowserTaskResult(BrowserCompletion(BrowserCompletionStatus.SATISFIED), observation)

    class Resource:
        def __init__(self, backend):
            self.backend = backend

        async def close(self, *, kill_browser=False):
            events.append(("closed", self.backend))

    await evaluate(
        cases, ["browser_use", "jev_ultrafast"],
        build=lambda backend: (Executor(backend), Resource(backend), None, None),
        keep_final_jev_tab_open=True,
        on_arm_finished=lambda row, tab_kept: events.append(
            ("finished", row["case_id"], row["backend"], tab_kept)),
    )

    assert events == [
        ("finished", "one", "browser_use", False), ("closed", "browser_use"),
        ("finished", "one", "jev_ultrafast", False), ("closed", "jev_ultrafast"),
        ("finished", "two", "browser_use", False), ("closed", "browser_use"),
        ("finished", "two", "jev_ultrafast", True),
    ]


@pytest.mark.asyncio
async def test_evaluate_records_failure_and_still_closes_browser():
    case = FindCase("repo", "repository", "GitHub", "Browser Use repository",
                    "https://github.com/browser-use/browser-use")
    closed = []

    class Executor:
        async def execute(self, goal):
            raise RuntimeError("browser failed")

    class Resource:
        async def close(self, *, kill_browser=False):
            closed.append(kill_browser)

    result = await evaluate([case], ["browser_use"],
                            build=lambda backend: (Executor(), Resource(), None, None))

    assert closed == [True]
    assert result["results"][0]["error"] == "RuntimeError: browser failed"
    assert result["summary"]["browser_use"]["errors"] == 1


@pytest.mark.asyncio
async def test_evaluate_reports_each_completed_arm_for_checkpointing():
    cases = [
        FindCase("one", "repository", "GitHub", "One", "https://github.com/a/one"),
        FindCase("two", "repository", "GitHub", "Two", "https://github.com/a/two"),
    ]
    snapshots = []

    class Executor:
        async def execute(self, goal):
            observation = BrowserObservation("page", "https://github.com/a/one", "One", "One",
                                             frozenset(), "page")
            return BrowserTaskResult(BrowserCompletion(BrowserCompletionStatus.SATISFIED), observation)

    class Resource:
        async def close(self, *, kill_browser=False):
            pass

    await evaluate(cases, ["browser_use"], build=lambda backend: (Executor(), Resource(), None, None),
                   on_result=lambda rows: snapshots.append([row["case_id"] for row in rows]))

    assert snapshots == [["one"], ["one", "two"]]


@pytest.mark.asyncio
async def test_evaluate_resumes_after_checkpoint_without_repeating_completed_arm():
    cases = [
        FindCase("one", "repository", "GitHub", "One", "https://github.com/a/one"),
        FindCase("two", "repository", "GitHub", "Two", "https://github.com/a/two"),
    ]
    built = []

    class Executor:
        async def execute(self, goal):
            case = next(item for item in cases if item.target == goal.argument("target"))
            observation = BrowserObservation("page", case.expected_url, case.target, case.target,
                                             frozenset(), "page")
            return BrowserTaskResult(BrowserCompletion(BrowserCompletionStatus.SATISFIED), observation)

    class Resource:
        async def close(self, *, kill_browser=False):
            pass

    def build(backend):
        built.append(backend)
        return Executor(), Resource(), None, None

    first = await evaluate(cases[:1], ["browser_use", "jev_ultrafast"], build=build)
    built.clear()
    resumed = await evaluate(cases, ["browser_use", "jev_ultrafast"], build=build,
                             previous_results=first["results"])

    assert built == ["browser_use", "jev_ultrafast"]
    assert len(resumed["results"]) == 4
    assert resumed["summary"]["jev_ultrafast"]["attempts"] == 2

    with pytest.raises(ValueError, match="checkpoint order"):
        await evaluate(cases, ["browser_use", "jev_ultrafast"], build=build,
                       previous_results=[{**first["results"][0], "backend": "jev_ultrafast"}])


@pytest.mark.asyncio
async def test_cleanup_error_does_not_change_a_successful_navigation_score():
    case = FindCase("one", "repository", "GitHub", "One", "https://github.com/a/one")

    class Executor:
        async def execute(self, goal):
            observation = BrowserObservation("page", case.expected_url, "One", "One",
                                             frozenset(), "page")
            return BrowserTaskResult(BrowserCompletion(BrowserCompletionStatus.SATISFIED), observation)

    class Resource:
        async def close(self, *, kill_browser=False):
            raise RuntimeError("close failed")

    report = await evaluate([case], ["browser_use"],
                            build=lambda backend: (Executor(), Resource(), None, None))

    assert report["results"][0]["success"] is True
    assert report["results"][0]["cleanup_error"] == "RuntimeError: close failed"


@pytest.mark.asyncio
async def test_stalled_jev_cleanup_keeps_the_event_loop_live_and_checkpoints():
    case = FindCase("one", "repository", "GitHub", "One", "https://github.com/a/one")
    checkpointed = []

    class Executor:
        async def execute(self, goal):
            await __import__("asyncio").sleep(10)

    class Resource:
        async def close(self, *, kill_browser=False):
            threading.Event().wait(10)

    started = time.monotonic()
    with pytest.raises(RuntimeError, match="cleanup timed out"):
        await evaluate([case], ["jev_ultrafast"],
                       build=lambda backend: (Executor(), Resource(), None, None),
                       timeout_seconds=0.01, cleanup_timeout_seconds=0.05,
                       on_result=lambda rows: checkpointed.extend(rows))

    assert time.monotonic() - started < 1
    assert checkpointed[0]["error"].startswith("TimeoutError")
    assert checkpointed[0]["cleanup_error"] == "TimeoutError: cleanup timed out"


@pytest.mark.asyncio
async def test_browser_use_starts_at_same_site_as_jev_before_deciding():
    case = FindCase("one", "repository", "GitHub", "One", "https://github.com/a/one")
    actions = []

    class Browser:
        async def act(self, action):
            actions.append(action)
            return {}

    class Executor:
        async def execute(self, goal):
            assert len(actions) == 1
            assert actions[0].url == "https://github.com/"
            assert dict(goal.arguments) == {"site": "GitHub", "target": "One"}

    await PreparedBrowserUseExecutor(Executor(), Browser()).execute(case.goal())


@pytest.mark.asyncio
async def test_browser_use_setup_navigation_error_does_not_start_decisions():
    class Browser:
        async def act(self, action):
            return {"error": "navigation failed"}

    class Executor:
        async def execute(self, goal):
            pytest.fail("controller ran after failed setup navigation")

    goal = FindCase("one", "repository", "GitHub", "One", "https://github.com/a/one").goal()
    with pytest.raises(RuntimeError, match="navigation failed"):
        await PreparedBrowserUseExecutor(Executor(), Browser()).execute(goal)
