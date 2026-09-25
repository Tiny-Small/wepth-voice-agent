"""OpenRouter Alpha Decisions transport for Jev-style typed questions."""

import asyncio
import logging
import os
import time
from typing import Any

import httpx

from ping_ponder.observability import emit
from ping_ponder.providers.base import InferenceResponse, ProviderError, ProviderTimeout, StructuredOutputError, TokenUsage, validate_output
from ping_ponder.providers.decisions import DecisionsResponse

logger = logging.getLogger(__name__)


class OpenRouterDecisionsProvider:
    ENDPOINT = "https://openrouter.ai/api/alpha/decisions"

    def __init__(self, *, api_key: str | None = None, client: httpx.AsyncClient | None = None,
                 timeout: float = 20, max_retries: int = 1, retry_delay: float = 0.1) -> None:
        self._api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self._api_key:
            raise ValueError("OPENROUTER_API_KEY is required for OpenRouterDecisionsProvider")
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

    async def decide(self, *, model: str, state: dict[str, Any], questions: dict[str, dict[str, Any]]) -> InferenceResponse[DecisionsResponse]:
        started = time.monotonic()
        emit(logger, "provider_request_start", provider="openrouter_decisions", model=model)
        body = {"model": model, "state": state, "questions": questions}
        headers = {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json", "X-OpenRouter-Metadata": "enabled"}
        for retry in range(self._max_retries + 1):
            try:
                response = await self._client.post(self.ENDPOINT, headers=headers, json=body, timeout=self._timeout)
            except httpx.TimeoutException as error:
                if retry < self._max_retries:
                    emit(logger, "provider_retry", provider="openrouter_decisions", model=model, retry=retry + 1, reason="timeout")
                    await asyncio.sleep(self._retry_delay * (retry + 1))
                    continue
                emit(logger, "provider_request_error", provider="openrouter_decisions", model=model, error_type="timeout", retries=retry)
                raise ProviderTimeout("OpenRouter Decisions request timed out") from error
            except httpx.HTTPError as error:
                emit(logger, "provider_request_error", provider="openrouter_decisions", model=model, error_type=type(error).__name__, retries=retry)
                raise ProviderError("OpenRouter Decisions transport failed") from error
            if response.status_code in {429, 500, 502, 503, 504, 524, 529} and retry < self._max_retries:
                emit(logger, "provider_retry", provider="openrouter_decisions", model=model, retry=retry + 1, reason=f"http_{response.status_code}")
                await asyncio.sleep(self._retry_delay * (retry + 1))
                continue
            if response.is_error:
                emit(logger, "provider_request_error", provider="openrouter_decisions", model=model, error_type=f"http_{response.status_code}", retries=retry)
                raise ProviderError(f"OpenRouter Decisions returned HTTP {response.status_code}")
            try:
                envelope = response.json()
                if not isinstance(envelope, dict):
                    raise TypeError("response is not an object")
                value = validate_output({"answers": envelope["answers"]}, DecisionsResponse, logger=logger, provider="openrouter_decisions", model=model)
                usage = TokenUsage.model_validate(envelope["usage"]) if isinstance(envelope.get("usage"), dict) else None
                metadata = envelope.get("openrouter_metadata") or {}
                if not isinstance(metadata, dict):
                    metadata = {}
            except (ValueError, TypeError, KeyError, StructuredOutputError) as error:
                emit(logger, "structured_output_validation_failed", provider="openrouter_decisions", model=model, schema="DecisionsResponse", error_type=type(error).__name__)
                raise StructuredOutputError("OpenRouter Decisions returned malformed output") from error
            generation_ms = metadata.get("generation_time") if isinstance(metadata.get("generation_time"), (int, float)) else None
            upstream_provider = envelope.get("provider") if isinstance(envelope.get("provider"), str) else None
            resolved_model = envelope.get("model") if isinstance(envelope.get("model"), str) else None
            generation_id = envelope.get("id") if isinstance(envelope.get("id"), str) else None
            routing_attempt = metadata.get("attempt") if isinstance(metadata.get("attempt"), int) else None
            latency = time.monotonic() - started
            emit(logger, "provider_request_end", provider="openrouter_decisions", model=model, resolved_model=resolved_model,
                 upstream_provider=upstream_provider, generation_id=generation_id, latency_seconds=latency,
                 provider_generation_ms=generation_ms, routing_attempt=routing_attempt, retries=retry,
                 usage=usage.model_dump(exclude_none=True) if usage else None)
            return InferenceResponse(value=value, provider="openrouter_decisions", model=model, latency_seconds=latency, retries=retry,
                                     usage=usage, generation_id=generation_id, resolved_model=resolved_model,
                                     provider_generation_ms=generation_ms, routing_attempt=routing_attempt,
                                     upstream_provider=upstream_provider)
        raise AssertionError("unreachable retry state")
