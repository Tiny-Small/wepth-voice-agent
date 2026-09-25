"""Deterministic application adapters for tests and offline execution."""

from __future__ import annotations

from typing import Any, Mapping

from .base import AdapterError


class MemorySpotifyAdapter:
    def __init__(self, initial_state: Mapping[str, Any]) -> None:
        self._state = dict(initial_state)
        self._skip_pending = False

    async def open(self) -> None:
        self._state.update({"spotify.running": True, "spotify.focused": True})

    async def search(self, query: str) -> None:
        if not self._state.get("spotify.running"):
            raise AdapterError("Spotify is not running")
        self._state.update({"spotify.search_results": query,
                            "spotify.search_query": query,
                            "spotify.search_ready": True})

    async def play_result(self, query: str | None) -> None:
        if not self._state.get("spotify.running"):
            raise AdapterError("Spotify is not running")
        selected = query or self._state.get("spotify.search_query") or self._state.get("spotify.search_results")
        if not selected:
            raise AdapterError("Spotify has no search result")
        self._state.update({"spotify.current_track": f"{selected} - top result",
                            "media.playing": True, "media.query": selected,
                            "spotify.paused": False})

    async def pause(self) -> None:
        self._state.update({"spotify.paused": True, "media.playing": False})

    async def resume(self) -> None:
        if self._state.get("spotify.current_track") is None:
            raise AdapterError("Spotify has no current track")
        self._state.update({"spotify.paused": False, "media.playing": True})

    async def skip(self) -> None:
        if self._state.get("spotify.current_track") is None:
            raise AdapterError("Spotify has no current track")
        self._skip_pending = True
        self._state.update({"media.playing": True, "spotify.paused": False})

    async def observe(self) -> Mapping[str, Any]:
        state = dict(self._state)
        state["spotify.skipped"] = self._skip_pending
        self._skip_pending = False
        return state


class MemoryBrowserAdapter:
    def __init__(self, initial_state: Mapping[str, Any]) -> None:
        self._state = dict(initial_state)

    async def open(self) -> None:
        self._state["browser.running"] = True

    async def search(self, query: str) -> None:
        if not self._state.get("browser.running"):
            raise AdapterError("Browser is not running")
        history = tuple(self._state.get("browser.history") or ())
        url = f"search://{query}"
        self._state.update({"browser.search_results": query, "browser.search_query": query,
                            "browser.current_url": url, "browser.history": (*history, url),
                            "browser.history_index": len(history)})

    async def navigate(self, target: str) -> None:
        if not self._state.get("browser.running"):
            raise AdapterError("Browser is not running")
        history = tuple(self._state.get("browser.history") or ())
        self._state.update({"browser.current_url": target, "browser.navigation_target": target,
                            "browser.history": (*history, target), "browser.history_index": len(history)})

    async def back(self) -> None:
        history = tuple(self._state.get("browser.history") or ())
        index = int(self._state.get("browser.history_index") or 0)
        if index <= 0:
            raise AdapterError("Browser has no back history")
        self._state.update({"browser.history_index": index - 1, "browser.current_url": history[index - 1]})

    async def forward(self) -> None:
        history = tuple(self._state.get("browser.history") or ())
        index = int(self._state.get("browser.history_index") or 0)
        if index + 1 >= len(history):
            raise AdapterError("Browser has no forward history")
        self._state.update({"browser.history_index": index + 1, "browser.current_url": history[index + 1]})

    async def observe(self) -> Mapping[str, Any]:
        index = int(self._state.get("browser.history_index") or 0)
        history = tuple(self._state.get("browser.history") or ())
        self._state["browser.can_back"] = index > 0
        self._state["browser.can_forward"] = index + 1 < len(history)
        return dict(self._state)
