import httpx
import pytest

from ping_ponder.voice.events import AudioFrame
from ping_ponder.voice.fish_audio import FishAudioSynthesizer


@pytest.mark.asyncio
async def test_fish_audio_streams_pcm_frames_and_uses_configured_model():
    request_seen = {}

    async def handler(request):
        request_seen["url"] = str(request.url)
        request_seen["authorization"] = request.headers.get("authorization")
        request_seen["json"] = __import__("json").loads(request.content)
        return httpx.Response(200, headers={
            "content-type": "audio/pcm;rate=44100;channels=1",
        }, content=b"\x01\x00\x02\x00")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    synth = FishAudioSynthesizer(client=client, api_key="key", voice="voice-id")

    async def text():
        yield "Hello there."

    frames = [frame async for frame in synth.synthesize_stream(text())]
    assert request_seen["url"] == "https://openrouter.ai/api/v1/audio/speech"
    assert request_seen["authorization"] == "Bearer key"
    assert request_seen["json"] == {
        "model": "fish-audio/s2.1-pro-free:free",
        "input": "Hello there.",
        "voice": "voice-id",
        "response_format": "pcm",
    }
    assert frames == [AudioFrame(b"\x01\x00\x02\x00", 44100, 1, 2)]
    await client.aclose()


@pytest.mark.asyncio
async def test_fish_audio_accepts_provider_default_voice_and_rejects_non_pcm():
    seen = {}

    async def handler(request):
        seen["json"] = __import__("json").loads(request.content)
        return httpx.Response(200, headers={"content-type": "audio/mpeg"}, content=b"mp3")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    synth = FishAudioSynthesizer(client=client, api_key="key")

    async def text():
        yield "Hi."

    with pytest.raises(ValueError, match="audio/pcm"):
        _ = [frame async for frame in synth.synthesize_stream(text())]
    assert "voice" not in seen["json"]
    await client.aclose()


@pytest.mark.asyncio
async def test_fish_audio_includes_openrouter_error_detail_on_http_failure():
    async def handler(request):
        return httpx.Response(
            400,
            json={"error": {"message": "voice is not supported for this model"}},
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    synth = FishAudioSynthesizer(client=client, api_key="key", model="google/test-tts")

    async def text():
        yield "Hello."

    with pytest.raises(httpx.HTTPStatusError, match="voice is not supported"):
        _ = [frame async for frame in synth.synthesize_stream(text())]
    await client.aclose()
