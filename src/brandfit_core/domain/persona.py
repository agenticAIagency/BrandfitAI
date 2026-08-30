from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field, model_validator

from brandfit_core.domain.analysis import EvidenceReference
from brandfit_core.domain.enums import PersonaStatus, ReviewStatus


class PlannedPost(BaseModel):
    post_id: UUID
    action: Literal["baseline", "deep_inspection", "reuse", "quarantine"]
    reason: str = Field(min_length=1, max_length=1000)


class CreatorAnalysisPlan(BaseModel):
    schema_version: str = "1.0"
    creator_id: UUID
    selected_posts: list[PlannedPost]
    missing_evidence: list[str] = Field(default_factory=list)
    requires_review: bool = False
    max_react_tool_calls: int = Field(6, ge=1, le=12)


class PersonaClaim(BaseModel):
    category: Literal[
        "content", "communication", "visual", "commercial", "performance", "safety", "trend"
    ]
    statement: str = Field(min_length=1, max_length=2000)
    confidence: float = Field(ge=0, le=1)
    evidence: list[EvidenceReference] = Field(min_length=1)


class PersonaConfidence(BaseModel):
    evidence_coverage: float = Field(ge=0, le=1)
    sample_size: float = Field(ge=0, le=1)
    extraction_completeness: float = Field(ge=0, le=1)
    freshness: float = Field(ge=0, le=1)
    overall: float = Field(ge=0, le=1)


class PersonaProposal(BaseModel):
    schema_version: str = "1.0"
    creator_id: UUID
    status: PersonaStatus
    summary: str = Field(min_length=1, max_length=10_000)
    claims: list[PersonaClaim]
    content_fingerprint: dict[str, float]
    communication_fingerprint: dict[str, float]
    visual_fingerprint: dict[str, float]
    commercial_fingerprint: dict[str, float]
    performance_context: dict[str, float | int | None]
    safety_observations: list[str]
    recent_90d: dict[str, object]
    trends: dict[str, object]
    confidence: PersonaConfidence
    evidence_post_ids: list[UUID]

    @model_validator(mode="after")
    def ensure_claim_evidence_is_in_scope(self) -> PersonaProposal:
        allowed = set(self.evidence_post_ids)
        for claim in self.claims:
            if any(ref.post_id not in allowed for ref in claim.evidence):
                raise ValueError("Persona claim references evidence outside this persona build")
        return self


class PersonaNarrativeDraft(BaseModel):
    """The only generative part of synthesis; aggregates stay deterministic."""

    summary: str = Field(min_length=1, max_length=10_000)
    claims: list[PersonaClaim] = Field(max_length=30)


class ClaimVerification(BaseModel):
    claim_index: int = Field(ge=0)
    accepted: bool
    reasons: list[str] = Field(default_factory=list)


class PersonaVerificationReport(BaseModel):
    accepted: bool
    claims: list[ClaimVerification]
    unsupported_claims: list[int] = Field(default_factory=list)
    policy_violations: list[str] = Field(default_factory=list)
    requires_human_review: bool = False


class PersonaVersionRead(BaseModel):
    id: UUID
    creator_id: UUID
    version: int
    status: PersonaStatus
    is_latest: bool
    summary: str
    persona: PersonaProposal
    verification: PersonaVerificationReport
    created_at: datetime
    approved_at: datetime | None = None


class PersonaReviewCreate(BaseModel):
    status: Literal["approved", "rejected"]
    reviewer: str = Field(min_length=1, max_length=300)
    notes: str | None = Field(None, max_length=10_000)


class PersonaReviewRead(BaseModel):
    id: UUID
    persona_version_id: UUID
    status: ReviewStatus
    reviewer: str
    notes: str | None
    created_at: datetime


class CreatorRelationshipRead(BaseModel):
    creator_id: UUID
    related_creator_id: UUID
    overall_similarity: float = Field(ge=-1, le=1)
    content_similarity: float = Field(ge=-1, le=1)
    visual_similarity: float = Field(ge=-1, le=1)
    communication_similarity: float = Field(ge=-1, le=1)
    commercial_similarity: float = Field(ge=-1, le=1)
    shared_terms: list[str]
    differences: list[str]
