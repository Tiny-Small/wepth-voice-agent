"""Windows Spotify adapter: UI commands plus media-session observations."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any, Mapping

from .base import AdapterError
from .powershell import PowerShellRunner, PowerShellTimeout


_SCRIPT = Path(__file__).with_name("scripts") / "spotify.ps1"


class WindowsSpotifyAdapter:
    def __init__(self, runner: PowerShellRunner | None = None, *, script: str | Path = _SCRIPT,
                 settle_timeout: float = 2.0) -> None:
        self.runner = runner or PowerShellRunner()
        self.script = str(script)
        self._search_query: str | None = None
        self._skip_pending = False
        self._pending_query: str | None = None
        self._play_verified_pending = False
        self._baseline_track: tuple[Any, Any] | None = None
        self.settle_timeout = settle_timeout

    async def _command(self, operation: str, *args: str) -> Mapping[str, Any]:
        return await self.runner.run(self.script, operation, *args)

    async def open(self) -> None:
        await self._command("open")

    async def search(self, query: str) -> None:
        try:
            result = await self._command("search", query)
        except PowerShellTimeout:
            # UI Automation can outlive the helper's transport timeout while
            # Spotify is opening or replacing its search view. Searching is
            # idempotent, so give the desktop one short settle interval and
            # retry once before reporting the adapter failure to the planner.
            await asyncio.sleep(0.25)
            result = await self._command("search", query)
        if result.get("search_query") != query or not result.get("search_ready"):
            raise AdapterError("Spotify search did not verify the requested query")
        self._search_query = query

    async def play_result(self, query: str | None) -> None:
        before = await self._command("observe")
        self._baseline_track = (before.get("artist"), before.get("title") or before.get("track"))
        before_status = str(before.get("status") or "").casefold()
        self._pending_query = query
        self._play_verified_pending = False
        await self._command("play", query or "")
        deadline = time.monotonic() + self.settle_timeout
        while time.monotonic() < deadline:
            observed = await self._command("observe")
            status = str(observed.get("status") or "").casefold()
            title = observed.get("title") or observed.get("track")
            artist = observed.get("artist")
            changed = (artist, title) != self._baseline_track
            search_query = observed.get("search_query") or self._search_query
            resumed_selected_track = before_status == "paused" and not changed
            if status == "playing" and title and artist and (changed or resumed_selected_track) \
                    and (query is None or search_query == query):
                self._play_verified_pending = True
                return
            await asyncio.sleep(min(0.1, max(0.0, deadline - time.monotonic())))
        raise AdapterError("Spotify did not verify playback of the requested result")

    async def pause(self) -> None:
        await self._command("pause")

    async def resume(self) -> None:
        await self._command("resume")

    async def skip(self) -> None:
        await self._command("skip")
        # The Windows media-session API does not expose a cumulative skip
        # counter. Keep a one-shot completion pulse so this command satisfies
        # exactly one SKIP goal and a later utterance can skip again.
        self._skip_pending = True

    async def observe(self) -> Mapping[str, Any]:
        raw = await self._command("observe")
        status = str(raw.get("status") or "").casefold()
        source = str(raw.get("source") or "").casefold()
        spotify_source = not source or "spotify" in source
        playing = status == "playing" and spotify_source
        title = raw.get("title") or raw.get("track")
        artist = raw.get("artist")
        current_track = f"{artist} - {title}" if artist and title else title or None
        observed_query = raw.get("search_query")
        observed_results = raw.get("search_results")
        # The native Spotify window exposes its Chromium omnibox as an
        # unrelated, read-only UIA Edit value. The script therefore reports
        # only whether a Search results grid is present; retain the exact
        # query only after the search command itself verified that grid.
        if not isinstance(observed_query, str) or observed_query.endswith("xpui.app.spotify.com/index.html"):
            observed_query = self._search_query
        if not isinstance(observed_results, str) or observed_results.endswith("xpui.app.spotify.com/index.html"):
            observed_results = self._search_query
        media_query = (self._pending_query
                       if playing and title and artist and self._play_verified_pending
                       and self._pending_query is not None
                       and observed_query == self._pending_query
                       else None)
        result = {
            "spotify.running": bool(raw.get("running", raw.get("process", False))),
            "spotify.focused": bool(raw.get("focused", False)),
            "spotify.search_results": observed_results,
            "spotify.search_query": observed_query,
            "spotify.search_ready": bool(raw.get("search_ready", False) or self._search_query),
            "spotify.current_track": current_track,
            "spotify.current_artist": artist,
            "spotify.paused": bool(raw.get("paused", status == "paused")),
            "spotify.skipped": self._skip_pending,
            "media.playing": playing,
            "media.query": media_query,
        }
        self._skip_pending = False
        if self._play_verified_pending:
            self._play_verified_pending = False
            self._pending_query = None
        return result
