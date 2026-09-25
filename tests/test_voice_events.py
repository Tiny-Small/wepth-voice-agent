import pytest

from ping_ponder.voice.events import (
    AudioFrame,
    FinalTranscript,
    PartialTranscript,
    SpeechStarted,
    SpeechStopped,
    TranscriptState,
)


def test_transcript_state_requires_positive_revision_and_matching_authority():
    with pytest.raises(ValueError):
        TranscriptState(turn_id="t1", revision=0, text="hello", final=False)

    partial = PartialTranscript(TranscriptState("t1", 1, "hello", False))
    final = FinalTranscript(TranscriptState("t1", 2, "hello", True))

    assert partial.state.final is False
    assert final.state.final is True


def test_event_and_audio_payloads_are_provider_neutral():
    frame = AudioFrame(b"\x00\x01", sample_rate=16_000, channels=1, sample_width=2)

    assert frame.pcm == b"\x00\x01"
    assert SpeechStarted("t1").turn_id == SpeechStopped("t1").turn_id
