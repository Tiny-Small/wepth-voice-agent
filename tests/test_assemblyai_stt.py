import asyncio
import sys
from types import ModuleType
from types import SimpleNamespace

import pytest

from ping_ponder.voice.assemblyai_stt import (
    AssemblyAIRecognizerSettings,
    SpeechRecognitionRuntimeError,
    AssemblyAIStreamingRecognizer,
)
from ping_ponder.voice.events import (
    AudioFrame,
    FinalTranscript,
    PartialTranscript,
    SpeechStarted,
    SpeechStopped,
)


def test_turn_events_get_monotonic_revisions_and_final_authority():
    queue = asyncio.Queue()
    recognizer = AssemblyAIStreamingRecognizer(api_key="test", event_queue=queue)

    recognizer.handle_turn(SimpleNamespace(
        turn_order=4, transcript="Play", end_of_turn=False,
    ))
    recognizer.handle_turn(SimpleNamespace(
        turn_order=4, transcript="Play some jazz", end_of_turn=False,
    ))
    recognizer.handle_turn(SimpleNamespace(
        turn_order=4, transcript="Play some jazz", end_of_turn=True,
    ))

    first = queue.get_nowait()
    second = queue.get_nowait()
    final = queue.get_nowait()
    stopped = queue.get_nowait()
    assert isinstance(first, PartialTranscript) and first.state.revision == 1
    assert isinstance(second, PartialTranscript) and second.state.revision == 2
    assert isinstance(final, FinalTranscript)
    assert final.state.revision == 3 and final.state.final
    assert isinstance(stopped, SpeechStopped) and stopped.turn_id == "4"


def test_speech_started_and_empty_turn_callbacks_only_enqueue_events():
    queue = asyncio.Queue()
    recognizer = AssemblyAIStreamingRecognizer(api_key="test", event_queue=queue)

    recognizer.handle_speech_started(SimpleNamespace(turn_order=8))
    recognizer.handle_turn(SimpleNamespace(
        turn_order=8, transcript="   ", end_of_turn=False,
    ))

    assert isinstance(queue.get_nowait(), SpeechStarted)
    assert queue.empty()


@pytest.mark.asyncio
async def test_run_configures_u35_and_streams_pcm_without_downstream_awaits():
    observed = {}

    class FakeClient:
        async def connect(self, parameters):
            observed["parameters"] = parameters

        async def stream(self, audio):
            observed["audio"] = [chunk async for chunk in audio]

        async def disconnect(self, *, terminate=True):
            observed["terminate"] = terminate

    def factory(settings, handlers):
        observed["settings"] = settings
        observed["handlers"] = handlers
        return FakeClient()

    async def frames():
        yield AudioFrame(b"pcm", sample_rate=16_000, channels=1, sample_width=2)

    queue = asyncio.Queue()
    recognizer = AssemblyAIStreamingRecognizer(
        api_key="test",
        settings=AssemblyAIRecognizerSettings(sample_rate=16_000),
        event_queue=queue,
        client_factory=factory,
    )

    await recognizer.run(frames(), queue)

    assert observed["settings"].speech_model == "universal-3-5-pro"
    assert observed["parameters"]["sample_rate"] == 16_000
    assert observed["audio"] == [b"pcm"]
    assert observed["terminate"] is True


@pytest.mark.asyncio
async def test_default_sdk_boundary_builds_realtime_parameters(monkeypatch):
    observed = {}

    class FakeParameters:
        def __init__(self, **values):
            self.values = values

    class FakeOptions:
        def __init__(self, **values):
            self.values = values

    class FakeEvents:
        SpeechStarted = "speech_started"
        Turn = "turn"
        Error = "error"

    class FakeClient:
        def __init__(self, options):
            observed["options"] = options

        def on(self, event, handler):
            observed.setdefault("handlers", {})[event] = handler

        async def connect(self, parameters):
            observed["parameters"] = parameters

        async def stream(self, audio):
            observed["audio"] = [chunk async for chunk in audio]

        async def disconnect(self, *, terminate=True):
            pass

    package = ModuleType("assemblyai")
    streaming = ModuleType("assemblyai.streaming")
    v3 = ModuleType("assemblyai.streaming.v3")
    v3.AsyncRealTimeTranscriber = FakeClient
    v3.RealTimeEvents = FakeEvents
    v3.RealTimeParameters = FakeParameters
    v3.RealTimeTranscriberOptions = FakeOptions
    monkeypatch.setitem(sys.modules, "assemblyai", package)
    monkeypatch.setitem(sys.modules, "assemblyai.streaming", streaming)
    monkeypatch.setitem(sys.modules, "assemblyai.streaming.v3", v3)

    async def frames():
        yield AudioFrame(b"pcm")

    queue = asyncio.Queue()
    recognizer = AssemblyAIStreamingRecognizer(api_key="secret", event_queue=queue)
    await recognizer.run(frames(), queue)

    assert isinstance(observed["parameters"], FakeParameters)
    assert observed["parameters"].values["speech_model"] == "universal-3-5-pro"
    assert set(observed["handlers"]) == {"speech_started", "turn", "error"}


@pytest.mark.asyncio
async def test_provider_error_terminates_recognizer_without_blocking_callback():
    class ErrorClient:
        async def connect(self, parameters):
            pass

        async def stream(self, audio):
            await handlers["error"](self, SimpleNamespace(code=4001, message="bad key"))
            await asyncio.Event().wait()

        async def disconnect(self, *, terminate=True):
            pass

    def factory(settings, supplied_handlers):
        handlers.update(supplied_handlers)
        return ErrorClient()

    async def frames():
        if False:
            yield AudioFrame(b"")

    handlers = {}
    queue = asyncio.Queue()
    recognizer = AssemblyAIStreamingRecognizer(
        api_key="bad", event_queue=queue, client_factory=factory,
    )

    with pytest.raises(SpeechRecognitionRuntimeError, match="bad key"):
        await asyncio.wait_for(recognizer.run(frames(), queue), timeout=0.2)
