from __future__ import annotations

from functools import lru_cache
from uuid import UUID

from pydantic import BaseModel, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ProviderTarget(BaseModel):
    provider: str
    model: str

    @classmethod
    def parse(cls, value: str) -> ProviderTarget:
        provider, separator, model = value.strip().partition(":")
        if not separator or not provider or not model:
            raise ValueError("Provider targets must use provider:model syntax")
        return cls(provider=provider.lower(), model=model)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="BRANDFIT_", case_sensitive=False, extra="ignore"
    )

    env: str = "development"
    database_url: str = "postgresql+asyncpg://brandfit:brandfit@localhost:5432/brandfit"
    database_url_sync: str = "postgresql+psycopg://brandfit:brandfit@localhost:5432/brandfit"
    redis_url: str = "redis://localhost:6379/0"
    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "brandfit"
    minio_secret_key: SecretStr = SecretStr("change-me")
    minio_secure: bool = False
    minio_bucket: str = "brandfit-evidence"
    default_workspace_id: UUID = UUID("00000000-0000-0000-0000-000000000001")

    provider_chain: str = "fake:fixture-v1"
    provider_pricing_per_million: dict[str, tuple[float, float]] = Field(
        default_factory=lambda: {"fake": (0.0, 0.0), "ollama": (0.0, 0.0)}
    )
    openai_api_key: SecretStr | None = None
    openai_base_url: str = "https://api.openai.com/v1"
    gemini_api_key: SecretStr | None = None
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    groq_api_key: SecretStr | None = None
    groq_base_url: str = "https://api.groq.com/openai/v1"
    ollama_base_url: str = "http://localhost:11434/v1"

    embedding_model: str = "BAAI/bge-m3"
    transcription_model: str = "small"
    enable_ocr: bool = True
    enable_transcription: bool = True
    max_react_tool_calls: int = Field(6, ge=1, le=12)
    analysis_confidence_threshold: float = Field(0.70, ge=0, le=1)
    persona_refresh_day: str = "sunday"
    log_level: str = "INFO"

    @field_validator("provider_chain")
    @classmethod
    def validate_provider_chain(cls, value: str) -> str:
        targets = [ProviderTarget.parse(item) for item in value.split(",") if item.strip()]
        if not targets:
            raise ValueError("At least one provider target is required")
        supported = {"fake", "gemini", "openai", "groq", "ollama"}
        unknown = {target.provider for target in targets} - supported
        if unknown:
            raise ValueError(f"Unsupported providers: {sorted(unknown)}")
        return value

    @property
    def provider_targets(self) -> list[ProviderTarget]:
        return [
            ProviderTarget.parse(item) for item in self.provider_chain.split(",") if item.strip()
        ]

    def estimate_cost_usd(
        self,
        provider: str,
        model: str,
        input_tokens: int | None,
        output_tokens: int | None,
    ) -> float | None:
        rates = self.provider_pricing_per_million.get(
            f"{provider}:{model}", self.provider_pricing_per_million.get(provider)
        )
        if rates is None or (input_tokens is None and output_tokens is None):
            return None
        value = ((input_tokens or 0) * rates[0] + (output_tokens or 0) * rates[1]) / 1_000_000
        return round(value, 12)


@lru_cache
def get_settings() -> Settings:
    return Settings()
