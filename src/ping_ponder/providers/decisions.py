"""Provider-neutral wire contract for typed Decisions API questions and answers."""

from typing import Annotated, Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from ping_ponder.providers.base import InferenceResponse


class ChoiceAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["choice"]
    choice: str
    confidence: float | None = Field(default=None, ge=0, le=1)
    probabilities: dict[str, float] | None = None


class NoulAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["noul"]
    noul: float = Field(ge=0, le=1)


DecisionAnswer = Annotated[ChoiceAnswer | NoulAnswer, Field(discriminator="type")]


class DecisionsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    answers: dict[str, DecisionAnswer]


class DecisionsProvider(Protocol):
    async def decide(self, *, model: str, state: dict[str, Any], questions: dict[str, dict[str, Any]]) -> InferenceResponse[DecisionsResponse]: ...
