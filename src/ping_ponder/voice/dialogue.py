"""Read-only dialogue policy and guarded conversational model adapters."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Protocol

import httpx

from ping_ponder.agentic.reply import TurnSituation, violates_safety


DEFAULT_DIALOGUE_SYSTEM_PROMPT = """You are the conversational component of a voice agent.
A separate control system performs computer actions. Do not generate or simulate
computer tool calls. Do not claim an action succeeded unless the supplied execution
state explicitly confirms success. Respond naturally using only the conversation and
read-only control-system context supplied to you."""


class ResponseMode(StrEnum):
    SILENT = "SILENT"
    ACK_ONLY = "ACK_ONLY"
    CLARIFICATION = "CLARIFICATION"
    LLM_RESPONSE = "LLM_RESPONSE"


class ExecutionStatus(StrEnum):
    NOT_REQUESTED = "NOT_REQUESTED"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"
    INFEASIBLE = "INFEASIBLE"


@dataclass(frozen=True)
class ControlContext:
    turn_id: str
    active_capability: str | None = None
    goal: str | None = None
    execution_status: ExecutionStatus = ExecutionStatus.NOT_REQUESTED
    reason: str | None = None
    goal_type: str | None = None
    missing_slots: tuple[str, ...] = ()


class ResponsePolicy(Protocol):
    def decide(self, user_text: str, context: ControlContext) -> ResponseMode: ...


class RuleBasedResponsePolicy:
    """Small V1 policy: questions converse; recognized commands acknowledge."""

    _QUESTION_STARTS = (
        "who ", "what ", "when ", "where ", "why ", "how ", "can you tell",
        "could you explain", "tell me ",
    )

    def __init__(self, *, silent_commands: bool = False) -> None:
        self.silent_commands = silent_commands

    def decide(self, user_text: str, context: ControlContext) -> ResponseMode:
        if context.missing_slots:
            return ResponseMode.CLARIFICATION
        lowered = user_text.strip().casefold()
        conversational = "?" in lowered or lowered.startswith(self._QUESTION_STARTS)
        if conversational:
            return ResponseMode.LLM_RESPONSE
        if context.goal:
            return ResponseMode.SILENT if self.silent_commands else ResponseMode.ACK_ONLY
        return ResponseMode.SILENT


def clarification_response(user_text: str, context: ControlContext) -> str:
    """Ask for the missing control detail without guessing or changing the goal."""
    if context.active_capability == "Browser" and context.goal_type == "SEARCH":
        normalized = " ".join(user_text.casefold().split()).rstrip(".,!?;:")
        if normalized == "search youtube":
            return "Would you like me to open YouTube, or search for something on YouTube?"
        return "What would you like me to search for?"
    if context.active_capability == "Browser" and context.goal_type == "NAVIGATE":
        return "Which website would you like me to open?"
    if context.active_capability == "Spotify" and context.goal_type == "PLAY":
        return "What would you like me to play?"
    return "Could you clarify what you would like me to do?"


class DialogueModel(Protocol):
    def stream_response(
        self, user_text: str, context: ControlContext,
    ) -> AsyncIterator[str]: ...


async def guarded_dialogue(model: DialogueModel, user_text: str,
                           context: ControlContext) -> AsyncIterator[str]:
    """Buffer one V1 response so an unsafe completion claim is never released."""
    chunks = [chunk async for chunk in model.stream_response(user_text, context)]
    response = "".join(chunks)
    situation = (
        TurnSituation.COMPLETED
        if context.execution_status is ExecutionStatus.SUCCEEDED
        else TurnSituation.NO_GOAL
    )
    if not response.strip() or violates_safety(response, situation):
        return
    for chunk in chunks:
        yield chunk


def acknowledgement(turn_id: str) -> str:
    """Choose a stable low-commitment acknowledgement without claiming success."""
    choices = ("Sure", "Okay", "Got it")
    return choices[sum(turn_id.encode("utf-8")) % len(choices)]


def corrective_response(context: ControlContext) -> str:
    """Phrase a known failed action without asking a model to infer what happened."""
    reason = (context.reason or "").casefold()
    if context.active_capability == "Spotify" and "open" in reason:
        return "I couldn't open Spotify."
    if context.active_capability:
        return f"I couldn't complete that in {context.active_capability}."
    return "I couldn't complete that."


class OpenAICompatibleDialogueModel:
    """Streaming chat-completions client with deliberately no tool schema."""

    def __init__(self, *, endpoint: str, model: str, client: httpx.AsyncClient,
                 api_key: str | None = None,
                 system_prompt: str = DEFAULT_DIALOGUE_SYSTEM_PROMPT) -> None:
        self.endpoint = endpoint
        self.model = model
        self.client = client
        self.api_key = api_key
        self.system_prompt = system_prompt

    async def stream_response(self, user_text: str,
                              context: ControlContext) -> AsyncIterator[str]:
        headers = ({"Authorization": f"Bearer {self.api_key}"}
                   if self.api_key else {})
        payload = {
            "model": self.model,
            "stream": True,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "system", "content": (
                    "Read-only control context: "
                    + json.dumps(asdict(context), default=str, sort_keys=True))},
                {"role": "user", "content": user_text},
            ],
        }
        async with self.client.stream(
                "POST", self.endpoint, headers=headers, json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line.removeprefix("data:").strip()
                if data == "[DONE]":
                    return
                chunk = json.loads(data)
                content = chunk.get("choices", [{}])[0].get("delta", {}).get("content")
                if content:
                    yield str(content)
