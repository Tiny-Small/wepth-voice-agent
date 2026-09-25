import json

import httpx
import pytest

from ping_ponder.voice.dialogue import (
    ControlContext,
    ExecutionStatus,
    OpenAICompatibleDialogueModel,
    ResponseMode,
    RuleBasedResponsePolicy,
    corrective_response,
    guarded_dialogue,
)
from ping_ponder.voice.tts import RecordingSpeechSynthesizer


async def collect(tokens):
    return [token async for token in tokens]


def test_complete_command_with_goal_gets_ack_only():
    context = ControlContext(
        turn_id="t1", active_capability="Spotify",
        goal="Spotify.PLAY(query='jazz')",
        execution_status=ExecutionStatus.RUNNING,
    )

    assert RuleBasedResponsePolicy().decide(
        "Play some jazz", context,
    ) is ResponseMode.ACK_ONLY


def test_question_and_mixed_turn_get_llm_response():
    policy = RuleBasedResponsePolicy()

    assert policy.decide(
        "Why is jazz improvisation difficult?", ControlContext(turn_id="t1"),
    ) is ResponseMode.LLM_RESPONSE
    assert policy.decide(
        "Play jazz and tell me why it is difficult?",
        ControlContext(turn_id="t1", goal="Spotify.PLAY(query='jazz')"),
    ) is ResponseMode.LLM_RESPONSE


@pytest.mark.asyncio
async def test_unconfirmed_completion_claim_is_not_released():
    class FakeModel:
        async def stream_response(self, user_text, context):
            yield "Jazz is now playing."

    result = await collect(guarded_dialogue(
        FakeModel(), "Play jazz",
        ControlContext(
            turn_id="t1", goal="Spotify.PLAY(query='jazz')",
            execution_status=ExecutionStatus.RUNNING,
        ),
    ))

    assert result == []


def test_spotify_open_failure_has_truthful_deterministic_correction():
    context = ControlContext(
        turn_id="t1", active_capability="Spotify",
        goal="Spotify.PLAY(query='jazz')",
        execution_status=ExecutionStatus.FAILED,
        reason="OpenSpotify failed: application unavailable",
    )

    assert corrective_response(context) == "I couldn't open Spotify."


@pytest.mark.asyncio
async def test_openai_compatible_dialogue_sends_no_tools_and_streams_tokens():
    captured = {}

    def handler(request):
        captured.update(json.loads(request.content))
        body = (
            'data: {"choices":[{"delta":{"content":"Jazz "}}]}\n\n'
            'data: {"choices":[{"delta":{"content":"is hard."}}]}\n\n'
            "data: [DONE]\n\n"
        )
        return httpx.Response(200, text=body)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        model = OpenAICompatibleDialogueModel(
            endpoint="https://example.test/v1/chat/completions",
            model="dialogue-test", client=client,
        )
        tokens = await collect(model.stream_response(
            "Why is jazz hard?", ControlContext(turn_id="t1"),
        ))

    assert tokens == ["Jazz ", "is hard."]
    assert "tools" not in captured
    assert captured["stream"] is True


@pytest.mark.asyncio
async def test_recording_synthesizer_records_requested_utterance():
    async def tokens():
        yield "Sure"

    synthesizer = RecordingSpeechSynthesizer()
    frames = [frame async for frame in synthesizer.synthesize_stream(tokens())]

    assert synthesizer.utterances == ["Sure"]
    assert frames and frames[0].pcm
