"""Task-local, factual execution events for the voice presentation layer."""

from __future__ import annotations

import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Callable, Mapping


@dataclass(frozen=True)
class ProgressEvent:
    type: str
    source: str
    data: Mapping[str, str] = field(default_factory=dict)
    at: float = field(default_factory=time.monotonic)


ProgressSink = Callable[[ProgressEvent], None]
progress_sink: ContextVar[ProgressSink | None] = ContextVar("voice_progress_sink", default=None)


def publish_progress(type: str, source: str, **data: str) -> None:
    sink = progress_sink.get()
    if sink is not None:
        sink(ProgressEvent(type, source, data))
