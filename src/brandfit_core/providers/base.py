from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from pydantic import BaseModel

from brandfit_core.domain.analysis import ProviderAttempt

T = TypeVar("T", bound=BaseModel)


@dataclass(slots=True)
class MediaPart:
    mime_type: str
    data: bytes
    locator: str


@dataclass(slots=True)
class ProviderPayload:
    prompt: str
    media: list[MediaPart] = field(default_factory=list)


@dataclass(slots=True)
class ProviderResponse(Generic[T]):
    value: T
    provider: str
    model: str
    attempts: list[ProviderAttempt]
    raw: dict[str, Any] = field(default_factory=dict)


class ProviderFailure(RuntimeError):
    def __init__(
        self, message: str, *, retryable: bool, blocked: bool = False, code: str = "provider_error"
    ):
        super().__init__(message)
        self.retryable = retryable
        self.blocked = blocked
        self.code = code


class StructuredProvider(ABC):
    name: str

    @abstractmethod
    async def generate_json(
        self,
        *,
        model: str,
        payload: ProviderPayload,
        response_schema: dict[str, Any],
        timeout_seconds: float,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Return parsed JSON plus provider metadata."""
