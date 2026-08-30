from __future__ import annotations

import base64
import json
from typing import Any

import httpx

from brandfit_core.providers.base import ProviderFailure, ProviderPayload, StructuredProvider


class GeminiProvider(StructuredProvider):
    name = "gemini"

    def __init__(self, *, base_url: str, api_key: str | None) -> None:
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
        if not self.api_key:
            raise ProviderFailure(
                "Gemini API key is not configured", retryable=False, code="missing_credentials"
            )

        parts: list[dict[str, Any]] = [{"text": payload.prompt}]
        parts.extend(
            {
                "inline_data": {
                    "mime_type": part.mime_type,
                    "data": base64.b64encode(part.data).decode("ascii"),
                }
            }
            for part in payload.media
        )
        body = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "temperature": 0.1,
                "responseMimeType": "application/json",
                "responseSchema": response_schema,
            },
        }
        url = f"{self.base_url}/models/{model}:generateContent"
        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            response = await client.post(url, params={"key": self.api_key}, json=body)

        if response.status_code in {401, 403}:
            raise ProviderFailure(
                "Gemini authentication failed", retryable=False, code="authentication_failed"
            )
        if response.status_code == 429 or response.status_code >= 500:
            raise ProviderFailure(
                f"Gemini temporary error: HTTP {response.status_code}",
                retryable=True,
                code=f"http_{response.status_code}",
            )
        if response.status_code >= 400:
            raise ProviderFailure(
                f"Gemini rejected request: HTTP {response.status_code}",
                retryable=False,
                code=f"http_{response.status_code}",
            )

        data = response.json()
        candidates = data.get("candidates", [])
        if candidates and candidates[0].get("finishReason") in {"SAFETY", "PROHIBITED_CONTENT"}:
            raise ProviderFailure(
                "Gemini safety policy blocked the request",
                retryable=False,
                blocked=True,
                code="safety_block",
            )
        try:
            text = candidates[0]["content"]["parts"][0]["text"]
            parsed = json.loads(text)
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise ProviderFailure(
                "Gemini returned invalid structured output",
                retryable=True,
                code="invalid_output",
            ) from exc

        usage = data.get("usageMetadata", {})
        metadata = {
            "input_tokens": usage.get("promptTokenCount"),
            "output_tokens": usage.get("candidatesTokenCount"),
            "response_id": data.get("responseId"),
        }
        return parsed, metadata
