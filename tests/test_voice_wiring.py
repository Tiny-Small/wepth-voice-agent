import pytest

from ping_ponder.voice.wiring import VoiceRuntime


@pytest.mark.asyncio
async def test_runtime_cleans_up_when_recognizer_fails():
    calls = []

    class AudioInput:
        async def start(self): calls.append("input.start")
        async def stop(self): calls.append("input.stop")
        async def frames(self):
            if False:
                yield None

    class Recognizer:
        async def run(self, audio, events):
            raise RuntimeError("stt failed")
        async def stop(self): calls.append("recognizer.stop")

    class Coordinator:
        events = object()
        async def start(self): calls.append("coordinator.start")
        async def stop(self): calls.append("coordinator.stop")

    runtime = VoiceRuntime(
        coordinator=Coordinator(), audio_input=AudioInput(), recognizer=Recognizer(),
    )

    with pytest.raises(RuntimeError, match="stt failed"):
        await runtime.run()

    assert "recognizer.stop" in calls
    assert "coordinator.stop" in calls
    assert "input.stop" in calls
