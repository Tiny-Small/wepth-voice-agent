import pytest

from ping_ponder.voice.events import AudioFrame
from ping_ponder.voice.audio import SoundDeviceAudioInput, SoundDeviceAudioOutput


@pytest.mark.parametrize("adapter", [SoundDeviceAudioInput, SoundDeviceAudioOutput])
def test_local_audio_adapters_reject_invalid_formats(adapter):
    with pytest.raises(ValueError, match="sample_rate"):
        adapter(sample_rate=0)


def test_audio_output_uses_streamed_sample_rate_when_rate_is_dynamic():
    class FakeStream:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.written = []

        def start(self):
            pass

        def write(self, pcm):
            self.written.append(pcm)

        def abort(self):
            pass

        def stop(self):
            pass

        def close(self):
            pass

    class FakeSoundDevice:
        def __init__(self):
            self.streams = []

        def RawOutputStream(self, **kwargs):
            stream = FakeStream(**kwargs)
            self.streams.append(stream)
            return stream

    device = FakeSoundDevice()
    output = SoundDeviceAudioOutput(sample_rate=None, sounddevice_module=device)
    output._sounddevice_module = device
    output._ensure_stream(AudioFrame(b"\x01\x00", sample_rate=44_100).sample_rate)
    assert device.streams[0].kwargs["samplerate"] == 44_100
    device.streams[0].stop()
    device.streams[0].close()
