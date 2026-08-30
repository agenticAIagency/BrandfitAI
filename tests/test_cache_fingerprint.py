from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import uuid4

from brandfit_core.config import Settings
from brandfit_core.services.post_analysis import _fingerprint


def test_observation_cache_never_crosses_post_evidence_scope() -> None:
    common = {
        "caption": "same content",
        "published_at": datetime.now(UTC),
        "media_type": "image",
    }
    first = SimpleNamespace(id=uuid4(), **common)
    second = SimpleNamespace(id=uuid4(), **common)
    assets = [SimpleNamespace(sha256="a" * 64)]
    settings = Settings()

    first_key = _fingerprint(first, assets, settings, metrics={}, profile={})  # type: ignore[arg-type]
    second_key = _fingerprint(second, assets, settings, metrics={}, profile={})  # type: ignore[arg-type]

    assert first_key != second_key


def test_configured_model_pricing_estimates_provider_cost() -> None:
    settings = Settings(
        provider_pricing_per_million={"openai:model": (2.0, 8.0)},
    )
    assert settings.estimate_cost_usd("openai", "model", 1_000, 500) == 0.006
