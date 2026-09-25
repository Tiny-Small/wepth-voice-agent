"""Windows Chrome adapter using the shared PowerShell transport."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping
from urllib.parse import parse_qs, urlparse

from .powershell import PowerShellRunner

_SCRIPT = Path(__file__).with_name("scripts") / "browser.ps1"


class WindowsBrowserAdapter:
    def __init__(self, runner: PowerShellRunner | None = None, *, script: str | Path = _SCRIPT) -> None:
        self.runner = runner or PowerShellRunner()
        self.script = str(script)

    async def _command(self, operation: str, *args: str) -> Mapping[str, Any]:
        return await self.runner.run(self.script, operation, *args)

    async def open(self) -> None:
        await self._command("open")

    async def search(self, query: str) -> None:
        await self._command("search", query)

    async def navigate(self, target: str) -> None:
        await self._command("navigate", target)

    async def back(self) -> None:
        await self._command("back")

    async def forward(self) -> None:
        await self._command("forward")

    async def observe(self) -> Mapping[str, Any]:
        raw = await self._command("observe")
        current_url = raw.get("current_url")
        query = raw.get("search_query")
        if query is None and current_url:
            parsed = urlparse(str(current_url))
            query = (parse_qs(parsed.query).get("q") or [None])[0]
        target = raw.get("navigation_target")
        return {
            "browser.running": bool(raw.get("running", raw.get("process", False))),
            "browser.current_url": current_url or target,
            "browser.search_results": raw.get("search_results") or query,
            "browser.search_query": query,
            "browser.navigation_target": target,
            "browser.history": tuple(raw.get("history") or ()),
            "browser.history_index": int(raw.get("history_index", -1)),
            "browser.can_back": bool(raw.get("can_back", False)),
            "browser.can_forward": bool(raw.get("can_forward", False)),
        }
