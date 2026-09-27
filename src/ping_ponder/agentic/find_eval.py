"""Small paired evaluation for this project's Browser.FIND executors."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any
from urllib.parse import parse_qs, parse_qsl, urlencode, urlsplit, urlunsplit

from .browser_use_slice import BrowserAction, BrowserCompletionStatus
from .capabilities.browser import resolve_navigation_target
from .goals import SemanticGoal


@dataclass(frozen=True)
class FindCase:
    id: str
    category: str
    site: str
    target: str
    expected_url: str

    def goal(self) -> SemanticGoal:
        # The answer key must never be passed as expected_url to BrowserTask.
        return SemanticGoal("Browser", "FIND", {"site": self.site, "target": self.target})


@dataclass(frozen=True)
class SearchCase:
    id: str
    category: str
    site: str
    target: str
    expected_url: str
    alternate_urls: tuple[str, ...] = ()

    def goal(self) -> SemanticGoal:
        return SemanticGoal("Browser", "SEARCH_WEBSITE", {"site": self.site, "query": self.target})


EvalCase = FindCase | SearchCase


class PreparedBrowserUseExecutor:
    """Open the same site that Jev opens before its first policy decision."""

    def __init__(self, executor: Any, browser: Any) -> None:
        self.executor = executor
        self.browser = browser

    async def execute(self, goal: SemanticGoal) -> Any:
        result = await self.browser.act(BrowserAction(kind="navigate", url=resolve_navigation_target(
            goal.argument("site"))))
        if not isinstance(result, Mapping):
            raise TypeError("site setup navigation returned no result")
        if result.get("error"):
            raise RuntimeError(f"site setup navigation failed: {result['error']}")
        return await self.executor.execute(goal)


def load_cases(path: str | Path) -> list[EvalCase]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise ValueError("task file must contain a nonempty JSON array")
    cases: list[EvalCase] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, dict):
            raise TypeError("each task must be an object")
        try:
            case_type = item.get("task_type", "destination")
            if case_type not in {"destination", "site_search"}:
                raise ValueError(f"unknown task type: {case_type!r}")
            fields = tuple(item[key] for key in
                           ("id", "category", "site", "target", "expected_url"))
            if case_type == "site_search":
                alternate_urls = item.get("alternate_urls", [])
                if not isinstance(alternate_urls, list):
                    raise ValueError("alternate_urls must be a list")
                case = SearchCase(*fields, alternate_urls=tuple(alternate_urls))
            else:
                case = FindCase(*fields)
        except KeyError as error:
            raise ValueError(f"task is missing {error.args[0]}") from error
        if any(not isinstance(value, str) or not value.strip() for value in
               (case.id, case.category, case.site, case.target, case.expected_url)):
            raise ValueError(f"task {case.id!r} has an empty or non-string field")
        if case.id in seen:
            raise ValueError(f"duplicate task id: {case.id}")
        parsed = urlsplit(case.expected_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError(f"task {case.id!r} needs an HTTP(S) expected_url")
        if isinstance(case, SearchCase):
            for answer_url in (case.expected_url, *case.alternate_urls):
                answer = urlsplit(answer_url)
                if answer.scheme not in {"http", "https"} or answer.hostname != parsed.hostname:
                    raise ValueError(f"site search task {case.id!r} has an invalid alternate URL")
                query_values = [value for values in parse_qs(answer.query).values() for value in values]
                if not any(" ".join(value.casefold().split()) == " ".join(case.target.casefold().split())
                           for value in query_values):
                    raise ValueError(f"site search task {case.id!r} needs its query in expected_url")
        seen.add(case.id)
        cases.append(case)
    return cases


def score_destination(actual_url: str, expected_url: str) -> bool:
    actual, expected = urlsplit(actual_url), urlsplit(expected_url)
    return bool(actual.scheme in {"http", "https"}
                and actual.hostname and actual.hostname.lower() == expected.hostname.lower()
                and actual.port == expected.port
                and (actual.path.rstrip("/") or "/") == (expected.path.rstrip("/") or "/"))


def score_search_url(actual_url: str, expected_url: str) -> bool:
    actual, expected = urlsplit(actual_url), urlsplit(expected_url)
    if not score_destination(actual_url, expected_url):
        return False
    actual_params, expected_params = parse_qs(actual.query), parse_qs(expected.query)
    if {key.casefold() for key in actual_params}.intersection({"js_challenge", "jsc_token", "captcha"}):
        return False
    return bool(expected_params and all(
        [value.casefold().strip() for value in actual_params.get(key, [])]
        == [value.casefold().strip() for value in values]
        for key, values in expected_params.items()
    ))


def _report_url(actual_url: str, case: EvalCase) -> str:
    parsed = urlsplit(actual_url)
    allowed_keys = (set().union(*(parse_qs(urlsplit(url).query)
                                 for url in (case.expected_url, *case.alternate_urls)))
                    if isinstance(case, SearchCase) else set())
    query = urlencode([(key, value) for key, value in parse_qsl(parsed.query)
                       if key in allowed_keys])
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, query, ""))


async def _close_browser(browser: Any, backend: str, timeout: float) -> None:
    if backend != "jev_ultrafast":
        await asyncio.wait_for(browser.close(kill_browser=True), timeout)
        return
    # Jev's async close currently takes a threading.Lock synchronously. Run it
    # on a daemon thread so a stalled policy/browser call cannot block this loop.
    done = threading.Event()
    failures: list[BaseException] = []

    def close() -> None:
        try:
            asyncio.run(browser.close(kill_browser=True))
        except BaseException as error:  # noqa: BLE001 - signal cleanup even on cancellation
            failures.append(error)
        finally:
            done.set()

    threading.Thread(target=close, name="jev-eval-close", daemon=True).start()
    deadline = time.monotonic() + timeout
    while not done.is_set():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("cleanup timed out")
        await asyncio.sleep(min(0.02, remaining))
    if failures:
        raise failures[0]


async def evaluate(
    cases: Iterable[EvalCase], backends: Iterable[str], *,
    build: Callable[[str], tuple[Any, Any, Any, Any]], repeats: int = 1,
    timeout_seconds: float = 120,
    on_result: Callable[[list[dict[str, Any]]], None] | None = None,
    on_arm_finished: Callable[[dict[str, Any], bool], None] | None = None,
    keep_final_jev_tab_open: bool = False,
    previous_results: Iterable[Mapping[str, Any]] = (),
    cleanup_timeout_seconds: float = 5,
) -> dict[str, Any]:
    """Run case/repeat/backend order, opening a fresh browser for each arm."""
    if repeats < 1 or timeout_seconds <= 0 or cleanup_timeout_seconds <= 0:
        raise ValueError("repeats, timeout_seconds and cleanup_timeout_seconds must be positive")
    backend_names = tuple(backends)
    if not backend_names or any(name not in {"browser_use", "jev_ultrafast"} for name in backend_names):
        raise ValueError("select browser_use and/or jev_ultrafast")
    case_list = tuple(cases)
    rows = [dict(row) for row in previous_results]
    expected_arms = [(case.id, case.category, repeat, backend)
                     for case in case_list for repeat in range(1, repeats + 1)
                     for backend in backend_names]
    if len(rows) > len(expected_arms):
        raise ValueError("checkpoint order exceeds the selected evaluation")
    for row, expected in zip(rows, expected_arms):
        if (row.get("case_id"), row.get("category"), row.get("repeat"), row.get("backend")) != expected:
            raise ValueError("checkpoint order does not match the selected evaluation")
    resume_count = len(rows)
    arm_index = 0
    for case_index, case in enumerate(case_list):
        for repeat in range(1, repeats + 1):
            for backend_index, backend in enumerate(backend_names):
                if arm_index < resume_count:
                    arm_index += 1
                    continue
                arm_index += 1
                started = time.monotonic()
                row: dict[str, Any] = {
                    "case_id": case.id, "category": case.category, "backend": backend,
                    "task_type": "site_search" if isinstance(case, SearchCase) else "destination",
                    "repeat": repeat, "completion_status": None, "final_url": None,
                    "destination_reached": False, "search_reached": False, "success": False,
                    "controller_calls": None, "duration_seconds": None,
                    "error": None, "cleanup_error": None,
                }
                browser = browser_provider = jev_provider = None
                cleanup_timed_out = False
                try:
                    executor, browser, browser_provider, jev_provider = build(backend)
                    result = await asyncio.wait_for(executor.execute(case.goal()), timeout_seconds)
                    row["completion_status"] = result.completion.status.value
                    row["controller_calls"] = result.controller_calls
                    if isinstance(case, SearchCase):
                        row["search_reached"] = any(
                            score_search_url(result.observation.url, answer_url)
                            for answer_url in (case.expected_url, *case.alternate_urls)
                        )
                        criterion_met = row["search_reached"]
                    else:
                        row["destination_reached"] = score_destination(result.observation.url, case.expected_url)
                        criterion_met = row["destination_reached"]
                    row["success"] = (criterion_met and
                                      result.completion.status is BrowserCompletionStatus.SATISFIED)
                    row["final_url"] = _report_url(result.observation.url, case)
                    row["timings"] = list(result.timings)
                except Exception as error:  # noqa: BLE001 - a failed arm must not stop the comparison
                    detail = str(error) or (f"exceeded {timeout_seconds}s" if isinstance(error, TimeoutError) else "")
                    row["error"] = f"{type(error).__name__}: {detail}"
                finally:
                    row["duration_seconds"] = round(time.monotonic() - started, 3)
                    keep_tab = bool(keep_final_jev_tab_open and backend == "jev_ultrafast"
                                    and row["error"] is None and browser is not None
                                    and case_index == len(case_list) - 1 and repeat == repeats
                                    and backend_index == len(backend_names) - 1)
                    if on_arm_finished is not None:
                        on_arm_finished(row, keep_tab)
                    for resource, kwargs in ((browser, {"kill_browser": True}),
                                             (browser_provider, {}), (jev_provider, {})):
                        if resource is None:
                            continue
                        if resource is browser and keep_tab:
                            continue
                        try:
                            if kwargs:
                                await _close_browser(resource, backend, cleanup_timeout_seconds)
                            else:
                                await asyncio.wait_for(resource.aclose(), cleanup_timeout_seconds)
                        except Exception as error:  # noqa: BLE001 - preserve task result and report cleanup
                            detail = f"{type(error).__name__}: {error}"
                            row["cleanup_error"] = (f"{row['cleanup_error']}; {detail}"
                                                    if row["cleanup_error"] else detail)
                            cleanup_timed_out = cleanup_timed_out or isinstance(error, TimeoutError)
                rows.append(row)
                if on_result is not None:
                    on_result(rows)
                if cleanup_timed_out:
                    raise RuntimeError("browser cleanup timed out; partial results were checkpointed")
    summary = {}
    for backend in backend_names:
        own = [row for row in rows if row["backend"] == backend]
        summary[backend] = {
            "attempts": len(own),
            "destinations_reached": sum(bool(row["destination_reached"]) for row in own),
            "searches_reached": sum(bool(row.get("search_reached")) for row in own),
            "successes": sum(bool(row["success"]) for row in own),
            "errors": sum(row["error"] is not None or row["cleanup_error"] is not None for row in own),
            "median_seconds": median(row["duration_seconds"] for row in own) if own else None,
        }
    return {"results": rows, "summary": summary}
