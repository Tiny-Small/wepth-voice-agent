"""Optional Jev Ultrafast Level-2 executor; completion remains grounded here."""

from __future__ import annotations

import asyncio
import hashlib
import os
import threading
import time
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from typing import Any
from urllib.parse import parse_qsl, urlsplit

from .browser_use_slice import (
    BrowserCompletion,
    BrowserCompletionStatus,
    BrowserCompletionVerifier,
    BrowserObservation,
    BrowserTask,
    BrowserTaskResult,
    _completion_eligible,
    _conflicting_named_entity_identity,
)
from .capabilities.browser import resolve_navigation_target
from .goals import SemanticGoal
from .operators import OperatorError

_TYPESAFE_URL = "https://api.typesafe.ai/v1/systemone"
_OPENROUTER_DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"
_policy_lock = threading.Lock()


@contextmanager
def _openrouter_policy_request(model_module: Any, policy_model: str):
    """Redirect Jev's single TypeSafe choice call without changing its action loop."""
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is required for Jev policy decisions")
    with _policy_lock:
        original = model_module.post_json
        previous_key = os.environ.get("TYPESAFE_API_KEY")

        def route(url: str, _key: str, body: Mapping[str, Any]) -> Any:
            if url != _TYPESAFE_URL:
                raise RuntimeError(f"unsupported Jev policy endpoint: {url}")
            return original(_OPENROUTER_DECISIONS_URL, key, {**body, "model": policy_model})

        # Jev reads this variable before calling post_json. The marker is never
        # sent to TypeSafe; route rejects any endpoint except the known policy URL.
        os.environ["TYPESAFE_API_KEY"] = "OPENROUTER_ROUTED"
        model_module.post_json = route
        try:
            yield
        finally:
            model_module.post_json = original
            if previous_key is None:
                os.environ.pop("TYPESAFE_API_KEY", None)
            else:
                os.environ["TYPESAFE_API_KEY"] = previous_key


@contextmanager
def _text_model_settings():
    """Supply upstream's TYPE_TEXT helper with the existing OpenRouter key."""
    previous: dict[str, str | None] = {}
    key = os.environ.get("OPENROUTER_API_KEY")
    if not os.environ.get("TEXT_MODEL_API_KEY") and key:
        values = {"TEXT_MODEL_API_KEY": key,
                  "TEXT_MODEL_BASE_URL": "https://openrouter.ai/api/v1"}
        if not os.environ.get("TEXT_MODEL"):
            values["TEXT_MODEL"] = "inception/mercury-2.5"
        for name, value in values.items():
            previous[name] = os.environ.get(name)
            os.environ[name] = value
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _agent_factory(url: str, task: str) -> Any:
    if not (os.environ.get("TYPESAFE_API_KEY") or os.environ.get("OPENROUTER_API_KEY")):
        raise RuntimeError("TYPESAFE_API_KEY or OPENROUTER_API_KEY is required for jev_ultrafast")
    try:
        from jev_ultrafast import Agent
    except ImportError as error:
        raise RuntimeError("install the 'jev' optional dependency to use jev_ultrafast") from error
    agent = Agent(url, task)
    try:
        # Keep animations from leaving controls transparent in Jev's background tab.
        agent.browser.call("Emulation.setEmulatedMedia", media="", features=[
            {"name": "prefers-reduced-motion", "value": "reduce"},
        ])
    except Exception:
        agent.close()
        raise
    return agent


def _fill_nodes(page: Mapping[str, Any]) -> set[int]:
    return {action["node"] for action in page.get("actions", [])
            if isinstance(action.get("node"), int) and action.get("kind") == "fill"}


def _start_url(task: BrowserTask) -> str:
    site = task.semantic_goal.argument("site")
    if site:
        try:
            return resolve_navigation_target(site)
        except OperatorError:
            pass
    return "https://www.google.com/"


def _observation(page: Mapping[str, Any]) -> BrowserObservation:
    url = str(page["url"])
    title = str(page.get("title") or "")
    visible_text = str(page.get("text") or "")
    fingerprint = str(page.get("fingerprint") or hashlib.sha256(
        repr((url, title, visible_text)).encode()).hexdigest())
    return BrowserObservation(
        observation_id=fingerprint,
        url=url,
        title=title,
        dom=visible_text,
        interactive_indices=frozenset(),
        grounding_fingerprint=fingerprint,
    )


class JevUltrafastExecutor:
    """Run Jev in one worker thread, then check its fresh page with our verifier.

    Cancellation stops before the next act, including when it arrives during a
    synchronous policy request. A browser operation already in flight cannot be
    interrupted by the upstream synchronous API.
    """

    def __init__(self, completion_verifier: BrowserCompletionVerifier, *,
                 agent_factory: Callable[[str, str], Any] = _agent_factory,
                 policy_model: str = "typesafe/jev-1.13",
                 max_seconds: float = 90.0) -> None:
        self.completion_verifier = completion_verifier
        self.agent_factory = agent_factory
        self.policy_model = policy_model
        self.max_seconds = max_seconds
        self._agent: Any | None = None
        self._lock = threading.Lock()

    def _command(self, agent: Any, name: str, body: Mapping[str, Any] | None = None) -> Any:
        if self.agent_factory is _agent_factory:
            if name == "predict" and not os.environ.get("TYPESAFE_API_KEY"):
                from jev_ultrafast import model
                with _openrouter_policy_request(model, self.policy_model):
                    return agent.command(name, body)
            if name == "act":
                with _text_model_settings():
                    return agent.command(name, body)
        return agent.command(name, body)

    def _settle_expanded_control(self, agent: Any, before: Mapping[str, Any],
                                 after: Mapping[str, Any], choice: str | None,
                                 stop: threading.Event, started: float) -> None:
        selected = next((action for action in before.get("actions", [])
                         if action.get("id") == choice), None)
        if (selected is None or selected.get("kind") != "click"
                or selected.get("role") != "button"
                or selected.get("expanded") != "false"
                or "search" not in selected.get("label", "").casefold()
                or after.get("url") != before.get("url")):
            return
        previous = _fill_nodes(before)
        if _fill_nodes(after) - previous:
            return
        deadline = min(started + self.max_seconds, time.monotonic() + 2.0)
        while time.monotonic() < deadline and not stop.wait(0.1):
            current = agent.browser.observe(screenshot=False)
            if current.get("url") != before.get("url") or _fill_nodes(current) - previous:
                return

    def _settle_search_before_blocked(self, agent: Any, page: Mapping[str, Any],
                                      stop: threading.Event, started: float,
                                      settled: set[tuple[str, int]]) -> bool:
        """Give a submitted search one brief chance to expose result links."""
        url = str(page.get("url") or "")
        parsed = urlsplit(url)
        path_segments = {part.casefold() for part in parsed.path.split("/") if part}
        query_keys = {key.casefold() for key, _ in parse_qsl(parsed.query)}
        if "search" not in path_segments and not query_keys.intersection({"q", "query", "search"}):
            return False
        history = getattr(agent, "state", {}).get("history", [])
        submission = next((index for index in range(len(history) - 1, max(-1, len(history) - 4), -1)
                           if history[index].get("kind") == "press_enter" or
                           (history[index].get("kind") == "click" and
                            str(history[index].get("action", "")).casefold() in {"go", "search", "submit"})), None)
        if submission is None or (url, submission) in settled:
            return False
        settled.add((url, submission))
        previous_links = {str(action.get("label")) for action in page.get("actions", [])
                          if action.get("kind") == "click" and action.get("role") == "link"}
        deadline = min(started + self.max_seconds, time.monotonic() + 1.0)
        while time.monotonic() < deadline:
            if stop.wait(min(0.1, deadline - time.monotonic())):
                return False
            try:
                current = agent.browser.observe(screenshot=False)
            except ValueError as error:
                if type(error).__name__ != "StalePage":
                    raise
                continue
            current_links = {str(action.get("label")) for action in current.get("actions", [])
                             if action.get("kind") == "click" and action.get("role") == "link"}
            if current.get("url") != url or current_links - previous_links:
                return True
        return False

    def _run(self, task: BrowserTask, stop: threading.Event) -> tuple[dict[str, Any], int, str, float]:
        with self._lock:
            if self._agent is not None:
                self._agent.close()
                self._agent = None
            started = time.monotonic()
            agent = self.agent_factory(
                _start_url(task),
                f"Find {task.target}" + (
                    f" on {task.semantic_goal.argument('site')}" if task.semantic_goal.argument("site") else ""
                ) + ". Stop when the requested destination is open.",
            )
            self._agent = agent
            cycles = 0
            terminal = "timeout"
            stale_prediction_started: float | None = None
            settled_searches: set[tuple[str, int]] = set()
            try:
                while not stop.is_set() and time.monotonic() - started < self.max_seconds:
                    try:
                        prediction = self._command(agent, "predict")
                    except ValueError as error:
                        if type(error).__name__ != "StalePage":
                            raise
                        now = time.monotonic()
                        if stale_prediction_started is None:
                            stale_prediction_started = now
                        if (now - stale_prediction_started >= 1.0
                                or now - started >= self.max_seconds):
                            raise
                        stop.wait(min(0.1, started + self.max_seconds - now))
                        continue
                    stale_prediction_started = None
                    cycles += 1
                    if stop.is_set():
                        break
                    page = prediction["page"]
                    if (prediction.get("decision", {}).get("choice") == "BLOCKED" and
                            self._settle_search_before_blocked(
                                agent, page, stop, started, settled_searches)):
                        continue
                    if stop.is_set():
                        break
                    try:
                        state = self._command(agent, "act", {"fingerprint": page["fingerprint"]})
                    except ValueError as error:
                        if type(error).__name__ != "StalePage":
                            raise
                        # The next predict refreshes the page through Agent's
                        # own freshness check before asking the policy again.
                        continue
                    if state["status"] in {"done", "blocked"}:
                        terminal = str(state["status"])
                        break
                    self._settle_expanded_control(
                        agent, page, state["page"], prediction.get("decision", {}).get("choice"),
                        stop, started,
                    )
                if stop.is_set():
                    return {}, cycles, "cancelled", time.monotonic() - started
                # A fresh observation, including after DONE/BLOCKED, is the only
                # state allowed into the project's completion verifier.
                page = agent.browser.observe(screenshot=False)
                return dict(page), cycles, terminal, time.monotonic() - started
            except Exception:
                agent.close()
                self._agent = None
                raise
            finally:
                if stop.is_set():
                    agent.close()
                    self._agent = None

    async def execute(self, goal: SemanticGoal) -> BrowserTaskResult:
        task = BrowserTask.from_semantic_goal(goal)
        stop = threading.Event()
        loop = asyncio.get_running_loop()
        work: asyncio.Future[tuple[dict[str, Any], int, str, float]] = loop.create_future()

        def run() -> None:
            try:
                result = self._run(task, stop)
            except Exception as error:  # noqa: BLE001 - surface worker failures to the awaiting task
                loop.call_soon_threadsafe(
                    lambda error=error: None if work.done() else work.set_exception(error))
            else:
                loop.call_soon_threadsafe(lambda: None if work.done() else work.set_result(result))

        threading.Thread(target=run, name="jev-ultrafast", daemon=True).start()
        try:
            page, cycles, terminal, elapsed = await work
        except asyncio.CancelledError:
            stop.set()
            raise
        observation = _observation(page)
        decision = BrowserCompletionStatus.UNCERTAIN
        verifier_started = time.monotonic()
        if task.satisfied_by(observation):
            decision = BrowserCompletionStatus.SATISFIED
        elif _completion_eligible(task, observation):
            try:
                decision = await self.completion_verifier.verify(task, observation)
            except Exception:  # noqa: BLE001 - same uncertain behavior as BrowserTaskExecutor
                decision = BrowserCompletionStatus.UNCERTAIN
            if (decision is BrowserCompletionStatus.SATISFIED
                    and _conflicting_named_entity_identity(task, observation)):
                decision = BrowserCompletionStatus.UNSATISFIED
        status = decision
        if decision is BrowserCompletionStatus.UNCERTAIN and terminal == "blocked":
            status = BrowserCompletionStatus.BLOCKED
        reason = ("grounded Jev completion verification" if status is BrowserCompletionStatus.SATISFIED
                  else f"Jev terminal state: {terminal}; verifier: {decision.value}")
        return BrowserTaskResult(
            BrowserCompletion(status, {"url": observation.url, "title": observation.title}, reason),
            observation,
            controller_calls=cycles,
            timings=(
                {"stage": "jev_ultrafast", "latency_s": elapsed, "cycles": cycles,
                 "terminal": terminal, "fallback": False},
                {"stage": "jev_completion", "latency_s": time.monotonic() - verifier_started,
                 "decision": decision.name, "fallback_to_controller": False},
                {"stage": "completion", "latency_s": elapsed + time.monotonic() - verifier_started,
                 "status": status.value},
            ),
        )

    async def close(self, *, kill_browser: bool = False) -> None:
        with self._lock:
            if self._agent is not None:
                self._agent.close()
                self._agent = None
