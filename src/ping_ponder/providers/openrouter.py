"""OpenAI-compatible structured chat completions over OpenRouter."""

import asyncio
import json
import logging
import os
import time
from typing import TypeVar

import httpx
from pydantic import BaseModel

from ping_ponder.observability import emit
from ping_ponder.providers.base import InferenceResponse, ProviderError, ProviderTimeout, StructuredOutputError, TokenUsage, validate_output

T = TypeVar("T", bound=BaseModel)
logger = logging.getLogger(__name__)


class OpenRouterProvider:
    ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self, *, api_key: str | None = None, client: httpx.AsyncClient | None = None, timeout: float = 20, max_retries: int = 1, retry_delay: float = 0.1) -> None:
        self._api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self._api_key:
            raise ValueError("OPENROUTER_API_KEY is required for OpenRouterProvider")
        if max_retries < 0:
            raise ValueError("max_retries must be nonnegative")
        self._client = client or httpx.AsyncClient(timeout=timeout)
        self._owns_client = client is None
        self._timeout = timeout
        self._max_retries = max_retries
        self._retry_delay = retry_delay

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def infer(self, *, model: str, messages: list[dict[str, str]], response_model: type[T]) -> InferenceResponse[T]:
        started = time.monotonic()
        emit(logger, "provider_request_start", provider="openrouter", model=model)
        body = {
            "model": model,
            "messages": messages,
            "stream": False,
            "response_format": {"type": "json_schema", "json_schema": {"name": response_model.__name__, "strict": True, "schema": response_model.model_json_schema()}},
            "provider": {"require_parameters": True},
        }
        headers = {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json", "X-OpenRouter-Metadata": "enabled"}
        for retry in range(self._max_retries + 1):
            try:
                response = await self._client.post(self.ENDPOINT, headers=headers, json=body, timeout=self._timeout)
            except httpx.TimeoutException as error:
                if retry < self._max_retries:
                    emit(logger, "provider_retry", provider="openrouter", model=model, retry=retry + 1, reason="timeout")
                    await asyncio.sleep(self._retry_delay * (retry + 1))
                    continue
                emit(logger, "provider_request_error", provider="openrouter", model=model, error_type="timeout", retries=retry)
                raise ProviderTimeout("OpenRouter request timed out", retries=retry) from error
            except httpx.HTTPError as error:
                emit(logger, "provider_request_error", provider="openrouter", model=model, error_type=type(error).__name__, retries=retry)
                raise ProviderError("OpenRouter transport failed", retries=retry) from error
            if response.status_code in {429, 500, 502, 503, 504} and retry < self._max_retries:
                emit(logger, "provider_retry", provider="openrouter", model=model, retry=retry + 1, reason=f"http_{response.status_code}")
                await asyncio.sleep(self._retry_delay * (retry + 1))
                continue
            if response.is_error:
                emit(logger, "provider_request_error", provider="openrouter", model=model, error_type=f"http_{response.status_code}", retries=retry)
                provider_message = ""
                try:
                    error_payload = response.json()
                    error_object = error_payload.get("error") if isinstance(error_payload, dict) else None
                    if isinstance(error_object, dict) and isinstance(error_object.get("message"), str):
                        provider_message = error_object["message"].replace(self._api_key, "[redacted]")[:500]
                except (ValueError, TypeError):
                    pass
                detail = f": {provider_message}" if provider_message else ""
                raise ProviderError(f"OpenRouter returned HTTP {response.status_code}{detail}",
                                    retries=retry, status_code=response.status_code)
            try:
                envelope = response.json()
                content = envelope["choices"][0]["message"]["content"]
                if not isinstance(content, str):
                    raise TypeError("message content is not text")
                payload = json.loads(content)
            except (ValueError, TypeError, KeyError, IndexError) as error:
                emit(logger, "structured_output_validation_failed", provider="openrouter", model=model, schema=response_model.__name__, error_type=type(error).__name__)
                raise StructuredOutputError("OpenRouter returned malformed structured output") from error
            value = validate_output(payload, response_model, logger=logger, provider="openrouter", model=model)
            usage = TokenUsage.model_validate(envelope["usage"]) if isinstance(envelope.get("usage"), dict) else None
            metadata = envelope.get("openrouter_metadata") or {}
            endpoints = metadata.get("endpoints") or {}
            selected = next((item for item in endpoints.get("available", []) if item.get("selected")), None)
            generation_id = envelope.get("id") if isinstance(envelope.get("id"), str) else None
            resolved_model = envelope.get("model") if isinstance(envelope.get("model"), str) else None
            generation_ms = metadata.get("generation_time") if isinstance(metadata.get("generation_time"), (int, float)) else None
            routing_attempt = metadata.get("attempt") if isinstance(metadata.get("attempt"), int) else None
            upstream_provider = selected.get("provider") if selected else None
            latency = time.monotonic() - started
            emit(logger, "provider_request_end", provider="openrouter", model=model, resolved_model=resolved_model, generation_id=generation_id, latency_seconds=latency, provider_generation_ms=generation_ms, routing_attempt=routing_attempt, upstream_provider=upstream_provider, retries=retry, usage=usage.model_dump(exclude_none=True) if usage else None)
            return InferenceResponse(value=value, provider="openrouter", model=model, latency_seconds=latency, retries=retry, usage=usage, generation_id=generation_id, resolved_model=resolved_model, provider_generation_ms=generation_ms, routing_attempt=routing_attempt, upstream_provider=upstream_provider)
        raise AssertionError("unreachable retry state")
