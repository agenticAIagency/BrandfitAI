from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any

from brandfit_core.providers.base import ProviderPayload, StructuredProvider


class FakeProvider(StructuredProvider):
    name = "fake"

    def __init__(self, response_factory: Callable[[ProviderPayload], dict[str, Any]] | None = None):
        self.response_factory = response_factory
        self.queued_responses: list[dict[str, Any] | Exception] = []

    def queue(self, response: dict[str, Any] | Exception) -> None:
        self.queued_responses.append(response)

    async def generate_json(
        self,
        *,
        model: str,
        payload: ProviderPayload,
        response_schema: dict[str, Any],
        timeout_seconds: float,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if self.queued_responses:
            response = self.queued_responses.pop(0)
            if isinstance(response, Exception):
                raise response
            return copy.deepcopy(response), {"input_tokens": 0, "output_tokens": 0}
        if self.response_factory:
            return self.response_factory(payload), {"input_tokens": 0, "output_tokens": 0}
        raise RuntimeError("Fake provider has no queued response or response factory")
