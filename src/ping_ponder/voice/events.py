"""Provider-neutral audio and speech-recognition events."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias


@dataclass(frozen=True)
class AudioFrame:
    pcm: bytes
    sample_rate: int = 16_000
    channels: int = 1
    sample_width: int = 2

    def __post_init__(self) -> None:
        if self.sample_rate <= 0 or self.channels <= 0 or self.sample_width <= 0:
            raise ValueError("audio format values must be positive")


@dataclass(frozen=True)
class TranscriptState:
    turn_id: str
    revision: int
    text: str
    final: bool

    def __post_init__(self) -> None:
        if not self.turn_id:
            raise ValueError("turn_id is required")
        if self.revision < 1:
            raise ValueError("revision must be positive")


@dataclass(frozen=True)
class SpeechStarted:
    turn_id: str


@dataclass(frozen=True)
class PartialTranscript:
    state: TranscriptState

    def __post_init__(self) -> None:
        if self.state.final:
            raise ValueError("partial transcript state cannot be final")


@dataclass(frozen=True)
class FinalTranscript:
    state: TranscriptState

    def __post_init__(self) -> None:
        if not self.state.final:
            raise ValueError("final transcript state must be final")


@dataclass(frozen=True)
class SpeechStopped:
    turn_id: str


@dataclass(frozen=True)
class SpeechRecognitionError:
    message: str
    code: int | None = None


SpeechEvent: TypeAlias = (
    SpeechStarted | PartialTranscript | FinalTranscript | SpeechStopped
    | SpeechRecognitionError
)
