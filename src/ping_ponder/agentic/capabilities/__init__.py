"""Built-in capabilities for the voice-to-action spine."""

from .browser import BrowserCapability, browser_goal_types, build_browser_descriptor
from .spotify import SpotifyCapability, build_spotify_descriptor, spotify_goal_types
from .transfer import TransferCapability, build_transfer_descriptor, transfer_goal_types,     transfer_world_schema

__all__ = ["BrowserCapability", "SpotifyCapability", "TransferCapability",
           "browser_goal_types", "spotify_goal_types", "transfer_goal_types",
           "build_browser_descriptor", "build_spotify_descriptor", "build_transfer_descriptor",
           "transfer_world_schema"]
