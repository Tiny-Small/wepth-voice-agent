import logging
from dataclasses import dataclass
from typing import Any, Generic, Protocol, TypeVar

from pydantic import BaseModel, ValidationError, model_validator

from ping_ponder.observability import emit

T = TypeVar("T", bound=BaseModel)


class ProviderError(Exception):
    def __init__(self, message: str, *, retries: int = 0,
                 status_code: int | None = None) -> None:
        super().__init__(message)
        self.retries = retries
        self.status_code = status_code


class ProviderTimeout(ProviderError):
    pass


class StructuredOutputError(ProviderError):
    pass


class TokenUsage(BaseModel):
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    reasoning_tokens: int | None = None
    cached_tokens: int | None = None

    @model_validator(mode="before")
    @classmethod
    def flatten_details(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        data = dict(value)
        completion = data.get("completion_tokens_details") or {}
        prompt = data.get("prompt_tokens_details") or {}
        data.setdefault("prompt_tokens", data.get("input_tokens"))
        data.setdefault("completion_tokens", data.get("output_tokens"))
        data.setdefault("reasoning_tokens", completion.get("reasoning_tokens"))
        data.setdefault("cached_tokens", prompt.get("cached_tokens"))
        return data


@dataclass(frozen=True)
class InferenceResponse(Generic[T]):
    value: T
    provider: str
    model: str
    latency_seconds: float
    retries: int
    usage: TokenUsage | None
    generation_id: str | None = None
    resolved_model: str | None = None
    provider_generation_ms: float | None = None
    routing_attempt: int | None = None
    upstream_provider: str | None = None


class StructuredLLMProvider(Protocol):
    async def infer(self, *, model: str, messages: list[dict[str, str]], response_model: type[T]) -> InferenceResponse[T]: ...


def validate_output(payload: Any, response_model: type[T], *, logger: logging.Logger, provider: str, model: str) -> T:
    try:
        return response_model.model_validate(payload)
    except (ValidationError, TypeError, ValueError) as error:
        emit(logger, "structured_output_validation_failed", provider=provider, model=model, schema=response_model.__name__, error_type=type(error).__name__)
        raise StructuredOutputError(f"invalid structured output for {response_model.__name__}") from error
