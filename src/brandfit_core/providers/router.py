from __future__ import annotations

import asyncio
import time
from typing import TypeVar

import httpx
from pydantic import BaseModel, SecretStr, ValidationError

from brandfit_core.config import Settings, get_settings
from brandfit_core.domain.analysis import ProviderAttempt
from brandfit_core.providers.base import (
    ProviderFailure,
    ProviderPayload,
    ProviderResponse,
    StructuredProvider,
)
from brandfit_core.providers.fake import FakeProvider
from brandfit_core.providers.fixture_factory import fixture_response
from brandfit_core.providers.gemini import GeminiProvider
from brandfit_core.providers.openai_compatible import OpenAICompatibleProvider

T = TypeVar("T", bound=BaseModel)


class ProviderRouter:
    def __init__(
        self,
        settings: Settings | None = None,
        providers: dict[str, StructuredProvider] | None = None,
        attempts_per_provider: int = 3,
    ) -> None:
        self.settings = settings or get_settings()
        self.attempts_per_provider = attempts_per_provider
        self.providers = providers or self._default_providers()

    def _default_providers(self) -> dict[str, StructuredProvider]:
        def secret(value: SecretStr | None) -> str | None:
            return value.get_secret_value() if value else None

        return {
            "fake": FakeProvider(response_factory=fixture_response),
            "gemini": GeminiProvider(
                base_url=self.settings.gemini_base_url,
                api_key=secret(self.settings.gemini_api_key),
            ),
            "openai": OpenAICompatibleProvider(
                name="openai",
                base_url=self.settings.openai_base_url,
                api_key=secret(self.settings.openai_api_key),
            ),
            "groq": OpenAICompatibleProvider(
                name="groq",
                base_url=self.settings.groq_base_url,
                api_key=secret(self.settings.groq_api_key),
            ),
            "ollama": OpenAICompatibleProvider(
                name="ollama", base_url=self.settings.ollama_base_url, api_key=None
            ),
        }

    async def generate(
        self,
        response_model: type[T],
        payload: ProviderPayload,
        *,
        timeout_seconds: float = 180,
        start_target_index: int = 0,
    ) -> ProviderResponse[T]:
        attempts: list[ProviderAttempt] = []
        last_error: Exception | None = None
        schema = response_model.model_json_schema()

        targets = self.settings.provider_targets[start_target_index:]
        if not targets:
            raise ProviderFailure(
                "No configured provider is available for escalation",
                retryable=False,
                code="no_escalation_provider",
            )
        for target in targets:
            provider = self.providers[target.provider]
            for attempt_number in range(1, self.attempts_per_provider + 1):
                started = time.perf_counter()
                try:
                    raw, metadata = await provider.generate_json(
                        model=target.model,
                        payload=payload,
                        response_schema=schema,
                        timeout_seconds=timeout_seconds,
                    )
                    value = response_model.model_validate(raw)
                    estimated_cost = self.settings.estimate_cost_usd(
                        target.provider,
                        target.model,
                        metadata.get("input_tokens"),
                        metadata.get("output_tokens"),
                    )
                    attempts.append(
                        ProviderAttempt(
                            provider=target.provider,
                            model=target.model,
                            attempt=attempt_number,
                            latency_ms=int((time.perf_counter() - started) * 1000),
                            input_tokens=metadata.get("input_tokens"),
                            output_tokens=metadata.get("output_tokens"),
                            estimated_cost_usd=estimated_cost,
                            status="success",
                        )
                    )
                    return ProviderResponse(
                        value=value,
                        provider=target.provider,
                        model=target.model,
                        attempts=attempts,
                        raw=metadata,
                    )
                except ValidationError as exc:
                    last_error = exc
                    attempts.append(
                        ProviderAttempt(
                            provider=target.provider,
                            model=target.model,
                            attempt=attempt_number,
                            latency_ms=int((time.perf_counter() - started) * 1000),
                            status="invalid_output",
                            error_code="schema_validation",
                        )
                    )
                except ProviderFailure as exc:
                    last_error = exc
                    attempts.append(
                        ProviderAttempt(
                            provider=target.provider,
                            model=target.model,
                            attempt=attempt_number,
                            latency_ms=int((time.perf_counter() - started) * 1000),
                            status="blocked" if exc.blocked else "retryable_error",
                            error_code=exc.code,
                        )
                    )
                    if exc.blocked:
                        raise
                    if not exc.retryable:
                        # Authentication, configuration and ordinary client
                        # rejections must not be hidden by another provider.
                        # Fallback is reserved for operational failures and
                        # exhausted structured-output repair.
                        raise
                except (httpx.TimeoutException, httpx.NetworkError) as exc:
                    last_error = exc
                    attempts.append(
                        ProviderAttempt(
                            provider=target.provider,
                            model=target.model,
                            attempt=attempt_number,
                            latency_ms=int((time.perf_counter() - started) * 1000),
                            status="retryable_error",
                            error_code="network_error",
                        )
                    )

                if attempt_number < self.attempts_per_provider:
                    await asyncio.sleep(min(2 ** (attempt_number - 1), 4))

        raise ProviderFailure(
            f"All configured providers failed: {last_error}",
            retryable=False,
            code="fallback_exhausted",
        )
