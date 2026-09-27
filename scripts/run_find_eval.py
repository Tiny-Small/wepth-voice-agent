#!/usr/bin/env python3
"""Run a paired, local Browser.FIND evaluation without voice input."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import subprocess
import tomllib
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

from ping_ponder.agentic.backend_config import BackendSettings
from ping_ponder.agentic.find_eval import (
    PreparedBrowserUseExecutor,
    SearchCase,
    evaluate,
    load_cases,
)

ROOT = Path(__file__).resolve().parents[1]
TASKS = ROOT / "eval" / "find_tasks.json"
RESUME_SETTINGS = (
    "task_file_sha256", "task_ids", "task_type", "backends", "repeats", "timeout_seconds",
    "keep_final_jev_tab_open", "jev_source", "local_jev_model", "browser_controller_model",
)


def jev_source_matches_pin(direct_url: str | None, requirement: str) -> bool:
    """Check the installed distribution's VCS origin, not its shared 0.1.0 version."""
    if not direct_url or not requirement.startswith("jev-ultrafast @ git+"):
        return False
    expected_url, expected_commit = requirement.split(" @ git+", 1)[1].rsplit("@", 1)
    try:
        installed = json.loads(direct_url)
        return (installed["url"].rstrip("/").casefold() == expected_url.rstrip("/").casefold()
                and installed["vcs_info"]["commit_id"].casefold() == expected_commit.casefold())
    except (KeyError, TypeError, ValueError):
        return False


def validate_resume_report(report: dict, configuration: dict) -> list[dict]:
    if report.get("complete") is not False:
        raise ValueError("report is already complete or has no incomplete checkpoint")
    previous = report.get("configuration")
    if not isinstance(previous, dict) or not isinstance(report.get("results"), list):
        raise TypeError("resume report needs configuration and results")
    for key in RESUME_SETTINGS:
        if previous.get(key) != configuration.get(key):
            raise ValueError(f"resume report differs in {key}")
    return report["results"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare this repo's two Browser.FIND executors")
    parser.add_argument("--list", action="store_true", help="Show tasks without opening a browser")
    parser.add_argument("--tasks", type=int, default=4, help="Run the first N selected tasks; default 4")
    parser.add_argument("--task-type", choices=("all", "destination", "site_search"), default="all")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--backend", choices=("both", "browser_use", "jev_ultrafast"), default="both")
    parser.add_argument("--timeout", type=float, default=120, help="Wall-clock seconds per task arm")
    parser.add_argument("--output", type=Path, help="Write results JSON here")
    parser.add_argument("--resume", type=Path, help="Continue an incomplete result file")
    parser.add_argument("--headful", action="store_true", help="Show Browser Use's Chrome window")
    parser.add_argument("--keep-final-jev-tab", action="store_true",
                        help="Leave the final Jev tab open after a completed arm for inspection")
    args = parser.parse_args(argv)

    cases = load_cases(TASKS)
    if args.task_type == "site_search":
        cases = [case for case in cases if isinstance(case, SearchCase)]
    elif args.task_type == "destination":
        cases = [case for case in cases if not isinstance(case, SearchCase)]
    if args.list:
        print(f"{len(cases)} tasks")
        for case in cases:
            print(f"{case.id:22} {case.category:14} {case.site}: {case.target}")
        return 0
    if args.tasks < 1 or args.tasks > len(cases):
        parser.error(f"--tasks must be between 1 and {len(cases)}")
    if args.repeats < 1 or args.timeout <= 0:
        parser.error("--repeats and --timeout must be positive")
    if args.resume and args.output:
        parser.error("--resume and --output cannot be combined")
    if not os.environ.get("OPENROUTER_API_KEY"):
        parser.error("OPENROUTER_API_KEY is required for both paths' completion verifier")

    from scripts.run_voice_agent import build_browser_find

    settings = BackendSettings.from_env()
    backends = ("browser_use", "jev_ultrafast") if args.backend == "both" else (args.backend,)
    if args.keep_final_jev_tab and "jev_ultrafast" not in backends:
        parser.error("--keep-final-jev-tab requires the jev_ultrafast backend")
    jev_source = None
    if "jev_ultrafast" in backends:
        project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        requirement = next(item for item in project["project"]["optional-dependencies"]["jev"]
                           if item.startswith("jev-ultrafast @ git+"))
        try:
            jev_source = metadata.distribution("jev-ultrafast").read_text("direct_url.json")
        except metadata.PackageNotFoundError:
            pass
        if not jev_source_matches_pin(jev_source, requirement):
            parser.error("installed Jev build does not match pyproject.toml; run "
                         f"python -m pip install --no-deps --force-reinstall '{requirement}'")
    output = args.resume or args.output or ROOT / "eval" / "results" / f"find_eval_{datetime.now(UTC):%Y%m%d_%H%M%S}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        working_tree_dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        revision = None
        working_tree_dirty = None
    configuration = {
        "created_utc": datetime.now(UTC).isoformat(),
        "git_revision": revision,
        "working_tree_dirty": working_tree_dirty,
        "task_file_sha256": hashlib.sha256(TASKS.read_bytes()).hexdigest(),
        "task_ids": [case.id for case in cases[:args.tasks]],
        "task_type": args.task_type,
        "backends": list(backends), "repeats": args.repeats,
        "timeout_seconds": args.timeout,
        "keep_final_jev_tab_open": args.keep_final_jev_tab,
        "jev_source": json.loads(jev_source) if jev_source else None,
        "local_jev_model": settings.local_jev_model,
        "browser_controller_model": settings.browser_controller_model,
    }
    previous_results: list[dict] = []
    if args.resume:
        try:
            prior_report = json.loads(output.read_text(encoding="utf-8"))
            previous_results = validate_resume_report(prior_report, configuration)
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
            parser.error(f"cannot resume {output}: {error}")
        configuration = prior_report["configuration"]

    def write_report(report: dict) -> None:
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        temporary.replace(output)

    def checkpoint(rows: list[dict]) -> None:
        write_report({"configuration": configuration, "complete": False, "results": rows})

    completed_arms = len(previous_results)

    def announce_arm(row: dict, tab_kept: bool) -> None:
        nonlocal completed_arms
        completed_arms += 1
        print(f"{completed_arms}/{args.tasks * args.repeats * len(backends)} "
              f"{row['case_id']} {row['backend']}: "
              f"{'success' if row['success'] else row['error'] or row['completion_status']} "
              f"(arm finished; {'leaving final Jev tab open' if tab_kept else 'closing browser'})",
              flush=True)

    def build(backend: str):
        capability, browser, browser_provider, jev_provider = build_browser_find(
            settings, headless=not args.headful, browser_backend=backend,
            max_seconds=args.timeout)
        executor = (PreparedBrowserUseExecutor(capability.executor, browser)
                    if backend == "browser_use" else capability.executor)
        return executor, browser, browser_provider, jev_provider

    report = asyncio.run(evaluate(cases[:args.tasks], backends, build=build,
                                  repeats=args.repeats, timeout_seconds=args.timeout,
                                  on_result=checkpoint, on_arm_finished=announce_arm,
                                  keep_final_jev_tab_open=args.keep_final_jev_tab,
                                  previous_results=previous_results))
    report["configuration"] = configuration
    report["complete"] = True
    write_report(report)
    for backend, summary in report["summary"].items():
        print(f"{backend}: {summary['successes']}/{summary['attempts']} successful; "
              f"{summary['destinations_reached']} destinations reached; "
              f"{summary['searches_reached']} search URLs matched; "
              f"{summary['errors']} errors; median {summary['median_seconds']}s")
    print(f"Results: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
