from __future__ import annotations

import json
from collections.abc import Callable

import httpx
import pytest

from brandfit_core.providers.base import MediaPart, ProviderPayload
from brandfit_core.providers.gemini import GeminiProvider
from brandfit_core.providers.openai_compatible import OpenAICompatibleProvider


def _mock_client(
    monkeypatch: pytest.MonkeyPatch,
    target: str,
    handler: Callable[[httpx.Request], httpx.Response],
) -> None:
    real_client = httpx.AsyncClient
    transport = httpx.MockTransport(handler)

    def factory(*_: object, **__: object) -> httpx.AsyncClient:
        return real_client(transport=transport)

    monkeypatch.setattr(target, factory)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("name", "api_key"),
    [("openai", "key"), ("groq", "key"), ("ollama", None)],
)
async def test_openai_compatible_adapters_send_multimodal_json(
    monkeypatch: pytest.MonkeyPatch, name: str, api_key: str | None
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/chat/completions")
        assert (request.headers.get("Authorization") is not None) is (api_key is not None)
        body = json.loads(request.content)
        assert body["messages"][0]["content"][1]["image_url"]["url"].startswith(
            "data:image/png;base64,"
        )
        return httpx.Response(
            200,
            json={
                "id": "response-1",
                "choices": [{"message": {"content": '{"value":"ok"}'}}],
                "usage": {"prompt_tokens": 2, "completion_tokens": 1},
            },
        )

    _mock_client(
        monkeypatch,
        "brandfit_core.providers.openai_compatible.httpx.AsyncClient",
        handler,
    )
    provider = OpenAICompatibleProvider(
        name=name,
        base_url="https://provider.test/v1",
        api_key=api_key,
    )
    value, metadata = await provider.generate_json(
        model="model",
        payload=ProviderPayload(
            prompt="analyze",
            media=[MediaPart(mime_type="image/png", data=b"png", locator="frame:0")],
        ),
        response_schema={"type": "object"},
        timeout_seconds=1,
    )
    assert value == {"value": "ok"}
    assert metadata["input_tokens"] == 2


@pytest.mark.asyncio
async def test_gemini_adapter_sends_multimodal_json(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/models/gemini-test:generateContent")
        assert request.url.params["key"] == "key"
        body = json.loads(request.content)
        assert body["contents"][0]["parts"][1]["inline_data"]["mime_type"] == "image/png"
        return httpx.Response(
            200,
            json={
                "responseId": "response-1",
                "candidates": [{"content": {"parts": [{"text": '{"value":"ok"}'}]}}],
                "usageMetadata": {"promptTokenCount": 2, "candidatesTokenCount": 1},
            },
        )

    _mock_client(
        monkeypatch,
        "brandfit_core.providers.gemini.httpx.AsyncClient",
        handler,
    )
    provider = GeminiProvider(base_url="https://gemini.test/v1beta", api_key="key")
    value, metadata = await provider.generate_json(
        model="gemini-test",
        payload=ProviderPayload(
            prompt="analyze",
            media=[MediaPart(mime_type="image/png", data=b"png", locator="frame:0")],
        ),
        response_schema={"type": "object"},
        timeout_seconds=1,
    )
    assert value == {"value": "ok"}
    assert metadata["output_tokens"] == 1
