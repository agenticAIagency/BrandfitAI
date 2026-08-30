from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field, model_validator


class EvidenceReference(BaseModel):
    post_id: UUID
    artifact_id: UUID | None = None
    kind: Literal["caption", "profile", "keyframe", "transcript", "ocr", "metrics"]
    locator: str = Field(min_length=1, max_length=1000)
    excerpt: str | None = Field(None, max_length=2000)


class TaxonomyLabel(BaseModel):
    key: str = Field(pattern=r"^[a-z0-9][a-z0-9_.-]{1,99}$")
    label: str = Field(min_length=1, max_length=200)
    confidence: float = Field(ge=0, le=1)
    evidence: list[EvidenceReference] = Field(default_factory=list)


class ContentFacet(BaseModel):
    topics: list[TaxonomyLabel] = Field(default_factory=list)
    formats: list[TaxonomyLabel] = Field(default_factory=list)
    intents: list[TaxonomyLabel] = Field(default_factory=list)
    open_tags: list[str] = Field(default_factory=list, max_length=30)


class CommunicationFacet(BaseModel):
    languages: list[str] = Field(default_factory=list)
    tones: list[TaxonomyLabel] = Field(default_factory=list)
    speaking_styles: list[TaxonomyLabel] = Field(default_factory=list)
    calls_to_action: list[str] = Field(default_factory=list, max_length=20)


class VisualFacet(BaseModel):
    composition_styles: list[TaxonomyLabel] = Field(default_factory=list)
    production_styles: list[TaxonomyLabel] = Field(default_factory=list)
    settings: list[str] = Field(default_factory=list, max_length=20)
    dominant_colors: list[str] = Field(default_factory=list, max_length=12)
    product_presence: bool = False


class CommercialFacet(BaseModel):
    brand_mentions: list[str] = Field(default_factory=list, max_length=30)
    sponsorship_disclosed: bool | None = None
    integration_styles: list[TaxonomyLabel] = Field(default_factory=list)
    affiliate_or_coupon_signal: bool = False


class SafetySignal(BaseModel):
    category: str = Field(min_length=1, max_length=100)
    severity: Literal["low", "medium", "high"]
    description: str = Field(min_length=1, max_length=1000)
    evidence: list[EvidenceReference] = Field(min_length=1)


class PerformanceContext(BaseModel):
    likes: int | None = Field(None, ge=0)
    comments: int | None = Field(None, ge=0)
    views: int | None = Field(None, ge=0)
    followers_at_capture: int | None = Field(None, ge=0)
    engagement_rate: float | None = Field(None, ge=0)
    relative_engagement_percentile: float | None = Field(None, ge=0, le=1)


class PostConfidence(BaseModel):
    extraction_coverage: float = Field(ge=0, le=1)
    schema_completeness: float = Field(ge=0, le=1)
    evidence_grounding: float = Field(ge=0, le=1)
    overall: float = Field(ge=0, le=1)


class PostObservation(BaseModel):
    schema_version: str = "1.0"
    post_id: UUID
    analysis_run_id: UUID
    original_language: str = Field(min_length=2, max_length=30)
    normalized_summary: str = Field(min_length=1, max_length=5000)
    content: ContentFacet
    communication: CommunicationFacet
    visual: VisualFacet
    commercial: CommercialFacet
    safety: list[SafetySignal] = Field(default_factory=list)
    performance: PerformanceContext
    confidence: PostConfidence
    requires_human_review: bool = False
    evidence: list[EvidenceReference] = Field(min_length=1)
    prompt_version: str
    taxonomy_version: str
    provider: str
    model: str
    created_at: datetime = Field(default_factory=lambda: datetime.now().astimezone())

    @model_validator(mode="after")
    def ensure_evidence_matches_post(self) -> PostObservation:
        refs = list(self.evidence)
        for signal in self.safety:
            refs.extend(signal.evidence)
        for ref in refs:
            if ref.post_id != self.post_id:
                raise ValueError("All evidence must refer to the analyzed post")
        return self


class ProviderAttempt(BaseModel):
    provider: str
    model: str
    attempt: int = Field(ge=1)
    latency_ms: int = Field(ge=0)
    input_tokens: int | None = Field(None, ge=0)
    output_tokens: int | None = Field(None, ge=0)
    estimated_cost_usd: float | None = Field(None, ge=0)
    status: Literal["success", "retryable_error", "blocked", "invalid_output"]
    error_code: str | None = None


class ReActAction(BaseModel):
    thought: str = Field(min_length=1, max_length=2000)
    action: Literal[
        "inspect_frame",
        "read_transcript",
        "read_ocr",
        "resolve_taxonomy",
        "compare_evidence",
        "finish",
    ]
    arguments: dict[str, Any] = Field(default_factory=dict)


class PostAnalysisDraft(BaseModel):
    """Provider-authored fields; identity and provenance are added by the publisher."""

    original_language: str = Field(min_length=2, max_length=30)
    normalized_summary: str = Field(min_length=1, max_length=5000)
    content: ContentFacet
    communication: CommunicationFacet
    visual: VisualFacet
    commercial: CommercialFacet
    safety: list[SafetySignal] = Field(default_factory=list)
    performance: PerformanceContext
    confidence: PostConfidence
    evidence: list[EvidenceReference] = Field(min_length=1)
