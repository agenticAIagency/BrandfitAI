from typing import Any

import pytest
from pydantic import BaseModel

from brandfit_core.config import Settings
from brandfit_core.providers.base import ProviderFailure, ProviderPayload, StructuredProvider
from brandfit_core.providers.router import ProviderRouter


class Output(BaseModel):
    value: str


class StubProvider(StructuredProvider):
    def __init__(self, values: list[dict[str, Any] | Exception]) -> None:
        self.values = values
        self.calls = 0

    async def generate_json(self, **_: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        self.calls += 1
        value = self.values[min(self.calls - 1, len(self.values) - 1)]
        if isinstance(value, Exception):
            raise value
        return value, {"input_tokens": 1, "output_tokens": 1}


@pytest.mark.asyncio
async def test_retry_then_fallback_records_attempts() -> None:
    first = StubProvider([ProviderFailure("rate", retryable=True, code="rate_limit")])
    second = StubProvider([{"value": "ok"}])
    router = ProviderRouter(
        Settings(provider_chain="openai:first,groq:second"),
        providers={"openai": first, "groq": second},
        attempts_per_provider=2,
    )
    response = await router.generate(Output, ProviderPayload(prompt="test"))
    assert response.value.value == "ok"
    assert first.calls == 2
    assert second.calls == 1
    assert [item.status for item in response.attempts] == [
        "retryable_error",
        "retryable_error",
        "success",
    ]


@pytest.mark.asyncio
async def test_safety_block_never_falls_back() -> None:
    first = StubProvider(
        [ProviderFailure("blocked", retryable=False, blocked=True, code="safety_block")]
    )
    second = StubProvider([{"value": "must-not-run"}])
    router = ProviderRouter(
        Settings(provider_chain="openai:first,groq:second"),
        providers={"openai": first, "groq": second},
        attempts_per_provider=2,
    )
    with pytest.raises(ProviderFailure, match="blocked"):
        await router.generate(Output, ProviderPayload(prompt="test"))
    assert second.calls == 0


@pytest.mark.asyncio
async def test_non_retryable_configuration_error_never_falls_back() -> None:
    first = StubProvider(
        [ProviderFailure("missing key", retryable=False, code="missing_credentials")]
    )
    second = StubProvider([{"value": "must-not-run"}])
    router = ProviderRouter(
        Settings(provider_chain="openai:first,groq:second"),
        providers={"openai": first, "groq": second},
        attempts_per_provider=3,
    )
    with pytest.raises(ProviderFailure, match="missing key"):
        await router.generate(Output, ProviderPayload(prompt="test"))
    assert first.calls == 1
    assert second.calls == 0


@pytest.mark.asyncio
async def test_uncertainty_escalation_can_start_at_second_provider() -> None:
    first = StubProvider([{"value": "must-not-run"}])
    second = StubProvider([{"value": "reviewed"}])
    router = ProviderRouter(
        Settings(provider_chain="openai:first,groq:second"),
        providers={"openai": first, "groq": second},
    )
    response = await router.generate(
        Output,
        ProviderPayload(prompt="test"),
        start_target_index=1,
    )
    assert response.value.value == "reviewed"
    assert first.calls == 0
    assert second.calls == 1
