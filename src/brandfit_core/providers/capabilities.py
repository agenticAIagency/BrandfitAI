from __future__ import annotations

from typing import Any, Protocol, TypeVar

from pydantic import BaseModel

from brandfit_core.providers.base import ProviderPayload, ProviderResponse

T = TypeVar("T", bound=BaseModel)


class StructuredGenerator(Protocol):
    async def generate(
        self, response_model: type[T], payload: ProviderPayload, *, timeout_seconds: float = 180
    ) -> ProviderResponse[T]: ...


class MultimodalAnalyzer(StructuredGenerator, Protocol):
    """A structured generator that accepts image/audio MediaPart inputs."""


class Transcriber(Protocol):
    def transcribe(self, object_store: Any, object_key: str) -> dict[str, object]: ...


class Embedder(Protocol):
    model_version: str

    def embed(self, texts: list[str]) -> list[list[float]]: ...
