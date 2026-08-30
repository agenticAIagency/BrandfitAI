from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

from brandfit_core.domain.analysis import (
    CommercialFacet,
    CommunicationFacet,
    ContentFacet,
    EvidenceReference,
    PerformanceContext,
    PostConfidence,
    PostObservation,
    TaxonomyLabel,
    VisualFacet,
)


def observation(
    *,
    post_id: UUID | None = None,
    topic: str = "topic.tech",
    language: str = "en",
    confidence: float = 0.9,
) -> PostObservation:
    post_id = post_id or uuid4()
    evidence = EvidenceReference(post_id=post_id, kind="caption", locator="caption:full")
    return PostObservation(
        post_id=post_id,
        analysis_run_id=uuid4(),
        original_language=language,
        normalized_summary="Grounded fixture observation",
        content=ContentFacet(topics=[TaxonomyLabel(key=topic, label=topic, confidence=confidence)]),
        communication=CommunicationFacet(languages=[language]),
        visual=VisualFacet(),
        commercial=CommercialFacet(),
        performance=PerformanceContext(likes=100, comments=10, followers_at_capture=1000),
        confidence=PostConfidence(
            extraction_coverage=confidence,
            schema_completeness=confidence,
            evidence_grounding=confidence,
            overall=confidence,
        ),
        evidence=[evidence],
        prompt_version="test",
        taxonomy_version="1.0",
        provider="fake",
        model="fixture",
        created_at=datetime.now(UTC),
    )
