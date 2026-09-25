"""Execution backend selection and capability adapter construction."""

from __future__ import annotations

import os
from dataclasses import dataclass

from .adapters import MemoryBrowserAdapter, MemorySpotifyAdapter
from .capabilities.browser import WORLD_SCHEMA as BROWSER_WORLD_SCHEMA
from .capabilities.spotify import WORLD_SCHEMA as SPOTIFY_WORLD_SCHEMA

SIMULATED = "simulated"
DESKTOP = "desktop"
VALID_EXECUTION_BACKENDS = (SIMULATED, DESKTOP)


@dataclass(frozen=True)
class ExecutionSettings:
    backend: str = SIMULATED

    @classmethod
    def from_env(cls, environ: dict[str, str] | None = None) -> "ExecutionSettings":
        env = os.environ if environ is None else environ
        backend = (env.get("AGENTIC_EXECUTION_BACKEND") or SIMULATED).strip().casefold()
        if backend not in VALID_EXECUTION_BACKENDS:
            raise ValueError("AGENTIC_EXECUTION_BACKEND must be simulated or desktop")
        return cls(backend=backend)

    @property
    def desktop(self) -> bool:
        return self.backend == DESKTOP

    def describe(self) -> str:
        return self.backend


@dataclass(frozen=True)
class AdapterBundle:
    spotify: object
    browser: object


def build_adapter_bundle(settings: ExecutionSettings | None = None) -> AdapterBundle:
    settings = settings or ExecutionSettings.from_env()
    if settings.backend == SIMULATED:
        return AdapterBundle(MemorySpotifyAdapter(SPOTIFY_WORLD_SCHEMA),
                             MemoryBrowserAdapter(BROWSER_WORLD_SCHEMA))
    from .adapters.windows_spotify import WindowsSpotifyAdapter
    from .adapters.windows_browser import WindowsBrowserAdapter
    return AdapterBundle(WindowsSpotifyAdapter(), WindowsBrowserAdapter())
