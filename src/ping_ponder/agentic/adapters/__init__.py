"""Application adapter boundaries used by capability operators."""

from .base import AdapterError, BrowserAdapter, SpotifyAdapter
from .memory import MemoryBrowserAdapter, MemorySpotifyAdapter
from .powershell import PowerShellOutputError, PowerShellProcessError, PowerShellRunner, PowerShellTimeout
from .windows_spotify import WindowsSpotifyAdapter
from .windows_browser import WindowsBrowserAdapter

__all__ = [
    "AdapterError", "BrowserAdapter", "SpotifyAdapter",
    "MemoryBrowserAdapter", "MemorySpotifyAdapter",
    "PowerShellOutputError", "PowerShellProcessError", "PowerShellRunner", "PowerShellTimeout",
    "WindowsSpotifyAdapter",
    "WindowsBrowserAdapter",
]
