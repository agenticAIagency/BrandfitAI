from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from statistics import mean
from uuid import UUID

from brandfit_core.domain.analysis import PostObservation
from brandfit_core.domain.enums import PersonaStatus
from brandfit_core.domain.persona import PersonaConfidence


@dataclass(slots=True)
class AggregateResult:
    status: PersonaStatus
    content: dict[str, float]
    communication: dict[str, float]
    visual: dict[str, float]
    commercial: dict[str, float]
    performance: dict[str, float | int | None]
    safety: list[str]
    recent_90d: dict[str, object]
    trends: dict[str, object]
    confidence: PersonaConfidence
    evidence_post_ids: list[UUID]
    evidence_examples: list[dict[str, object]]


def _distribution(keys: list[str]) -> dict[str, float]:
    if not keys:
        return {}
    counts = Counter(keys)
    total = sum(counts.values())
    return {key: round(value / total, 6) for key, value in sorted(counts.items())}


def _labels(observation: PostObservation, facet: str) -> list[str]:
    if facet == "content":
        return [
            item.key
            for item in observation.content.topics
            + observation.content.formats
            + observation.content.intents
        ]
    if facet == "communication":
        return observation.communication.languages + [
            item.key
            for item in observation.communication.tones + observation.communication.speaking_styles
        ]
    if facet == "visual":
        return [
            item.key
            for item in observation.visual.composition_styles + observation.visual.production_styles
        ]
    return observation.commercial.brand_mentions + [
        item.key for item in observation.commercial.integration_styles
    ]


def _performance(observations: list[PostObservation]) -> dict[str, float | int | None]:
    result: dict[str, float | int | None] = {}
    for key in ["likes", "comments", "views", "engagement_rate"]:
        values = [
            getattr(item.performance, key)
            for item in observations
            if getattr(item.performance, key) is not None
        ]
        result[f"mean_{key}"] = round(mean(values), 6) if values else None
    return result


def aggregate_observations(
    observations: list[PostObservation],
    *,
    available_post_count: int,
    published_at: dict[UUID, datetime] | None = None,
    now: datetime | None = None,
) -> AggregateResult:
    now = now or datetime.now(UTC)
    published_at = published_at or {}
    observations = sorted(observations, key=lambda item: item.created_at)
    count = len(observations)
    status = (
        PersonaStatus.INSUFFICIENT
        if count < 5
        else PersonaStatus.PRELIMINARY
        if count < 10
        else PersonaStatus.ESTABLISHED
    )
    all_facets = {
        facet: _distribution([key for item in observations for key in _labels(item, facet)])
        for facet in ["content", "communication", "visual", "commercial"]
    }
    recent = [
        item
        for item in observations
        if published_at.get(item.post_id, item.created_at) >= now - timedelta(days=90)
    ]
    recent_facets = {
        facet: _distribution([key for item in recent for key in _labels(item, facet)])
        for facet in ["content", "communication", "visual", "commercial"]
    }
    trends: dict[str, object] = {}
    for facet in all_facets:
        changes = {
            key: round(recent_facets[facet].get(key, 0) - all_facets[facet].get(key, 0), 6)
            for key in set(all_facets[facet]) | set(recent_facets[facet])
            if abs(recent_facets[facet].get(key, 0) - all_facets[facet].get(key, 0)) >= 0.2
        }
        if changes:
            trends[facet] = changes
    average_extraction = (
        mean([item.confidence.extraction_coverage for item in observations])
        if observations
        else 0.0
    )
    newest = max(
        (published_at.get(item.post_id, item.created_at) for item in observations),
        default=now - timedelta(days=365),
    )
    age_days = max((now - newest).total_seconds() / 86_400, 0)
    freshness = math.exp(-age_days / 180)
    confidence = PersonaConfidence(
        evidence_coverage=min(count / max(available_post_count, 1), 1),
        sample_size=min(count / 10, 1),
        extraction_completeness=average_extraction,
        freshness=freshness,
        overall=0,
    )
    confidence.overall = round(
        0.35 * confidence.evidence_coverage
        + 0.25 * confidence.sample_size
        + 0.25 * confidence.extraction_completeness
        + 0.15 * confidence.freshness,
        6,
    )
    examples: list[dict[str, object]] = []
    for facet in ["content", "communication", "visual", "commercial"]:
        for key, share in sorted(all_facets[facet].items(), key=lambda item: item[1], reverse=True)[
            :2
        ]:
            matching = next((item for item in observations if key in _labels(item, facet)), None)
            if matching:
                reference = matching.evidence[0]
                examples.append(
                    {
                        "category": facet,
                        "statement": f"{key} appears in {share:.0%} of classified {facet} signals.",
                        "confidence": min(matching.confidence.overall, 0.95),
                        "evidence": [reference.model_dump(mode="json")],
                    }
                )
    safety = sorted({signal.category for item in observations for signal in item.safety})
    return AggregateResult(
        status=status,
        content=all_facets["content"],
        communication=all_facets["communication"],
        visual=all_facets["visual"],
        commercial=all_facets["commercial"],
        performance=_performance(observations),
        safety=safety,
        recent_90d={"post_count": len(recent), "facets": recent_facets},
        trends=trends,
        confidence=confidence,
        evidence_post_ids=[item.post_id for item in observations],
        evidence_examples=examples,
    )
