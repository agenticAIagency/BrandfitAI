from __future__ import annotations

import base64
import json
from typing import Any

import httpx

from brandfit_core.providers.base import ProviderFailure, ProviderPayload, StructuredProvider


class OpenAICompatibleProvider(StructuredProvider):
    def __init__(self, *, name: str, base_url: str, api_key: str | None) -> None:
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    async def generate_json(
        self,
        *,
        model: str,
        payload: ProviderPayload,
        response_schema: dict[str, Any],
        timeout_seconds: float,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if self.name != "ollama" and not self.api_key:
            raise ProviderFailure(
                f"{self.name} API key is not configured",
                retryable=False,
                code="missing_credentials",
            )

        content: list[dict[str, Any]] = [{"type": "text", "text": payload.prompt}]
        for part in payload.media:
            encoded = base64.b64encode(part.data).decode("ascii")
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{part.mime_type};base64,{encoded}"},
                }
            )

        body = {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0.1,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "brandfit_response",
                    "strict": True,
                    "schema": response_schema,
                },
            },
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions", headers=headers, json=body
            )
            if response.status_code == 400 and self.name in {"groq", "ollama"}:
                body["response_format"] = {"type": "json_object"}
                response = await client.post(
                    f"{self.base_url}/chat/completions", headers=headers, json=body
                )

        if response.status_code in {401, 403}:
            raise ProviderFailure(
                f"{self.name} authentication failed",
                retryable=False,
                code="authentication_failed",
            )
        if response.status_code == 400 and any(
            marker in response.text.lower()
            for marker in ["safety", "content policy", "content_filter", "moderation"]
        ):
            raise ProviderFailure(
                f"{self.name} safety policy blocked the request",
                retryable=False,
                blocked=True,
                code="safety_block",
            )
        if response.status_code == 429 or response.status_code >= 500:
            raise ProviderFailure(
                f"{self.name} temporary error: HTTP {response.status_code}",
                retryable=True,
                code=f"http_{response.status_code}",
            )
        if response.status_code >= 400:
            raise ProviderFailure(
                f"{self.name} rejected request: HTTP {response.status_code}",
                retryable=False,
                code=f"http_{response.status_code}",
            )

        data = response.json()
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        if message.get("refusal") or choice.get("finish_reason") == "content_filter":
            raise ProviderFailure(
                f"{self.name} safety policy blocked the request",
                retryable=False,
                blocked=True,
                code="safety_block",
            )
        try:
            text = message["content"]
            parsed = json.loads(text) if isinstance(text, str) else text
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise ProviderFailure(
                f"{self.name} returned invalid structured output",
                retryable=True,
                code="invalid_output",
            ) from exc

        usage = data.get("usage", {})
        metadata = {
            "input_tokens": usage.get("prompt_tokens"),
            "output_tokens": usage.get("completion_tokens"),
            "response_id": data.get("id"),
        }
        return parsed, metadata
