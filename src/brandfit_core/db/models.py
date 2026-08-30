from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    Uuid,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from brandfit_core.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin
from brandfit_core.domain.enums import (
    ArtifactType,
    BatchStatus,
    JobStatus,
    MediaType,
    PersonaStatus,
    ReviewStatus,
    WorkflowKind,
)


class Workspace(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "workspaces"

    name: Mapped[str] = mapped_column(String(300), nullable=False)


class IngestionBatch(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "ingestion_batches"
    __table_args__ = (UniqueConstraint("workspace_id", "source", "source_run_id"),)

    workspace_id: Mapped[UUID] = mapped_column(ForeignKey("workspaces.id"), index=True)
    source: Mapped[str] = mapped_column(String(100), nullable=False)
    source_run_id: Mapped[str] = mapped_column(String(300), nullable=False)
    schema_version: Mapped[str] = mapped_column(String(30), nullable=False)
    status: Mapped[BatchStatus] = mapped_column(Enum(BatchStatus), nullable=False)
    accepted_posts: Mapped[int] = mapped_column(Integer, default=0)
    duplicate_posts: Mapped[int] = mapped_column(Integer, default=0)
    quarantined_posts: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, default=list)
    raw_manifest: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    job_id: Mapped[UUID | None] = mapped_column(Uuid, nullable=True)


class Creator(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "creators"
    __table_args__ = (UniqueConstraint("workspace_id", "platform", "platform_user_id"),)

    workspace_id: Mapped[UUID] = mapped_column(ForeignKey("workspaces.id"), index=True)
    platform: Mapped[str] = mapped_column(String(100), nullable=False)
    platform_user_id: Mapped[str] = mapped_column(String(300), nullable=False)
    current_username: Mapped[str] = mapped_column(String(300), nullable=False, index=True)
    latest_persona_version: Mapped[int | None] = mapped_column(Integer, nullable=True)


class CreatorProfileSnapshot(UUIDPrimaryKeyMixin, Base):
    __tablename__ = "creator_profile_snapshots"

    workspace_id: Mapped[UUID] = mapped_column(ForeignKey("workspaces.id"), index=True)
    creator_id: Mapped[UUID] = mapped_column(ForeignKey("creators.id"), index=True)
    username: Mapped[str] = mapped_column(String(300), nullable=False)
    profile: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    captured_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class Post(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "posts"
    __table_args__ = (UniqueConstraint("workspace_id", "platform", "platform_post_id"),)

    workspace_id: Mapped[UUID] = mapped_column(ForeignKey("workspaces.id"), index=True)
    creator_id: Mapped[UUID] = mapped_column(ForeignKey("creators.id"), index=True)
    ingestion_batch_id: Mapped[UUID] = mapped_column(ForeignKey("ingestion_batches.id"), index=True)
    platform: Mapped[str] = mapped_column(String(100), nullable=False)
    platform_post_id: Mapped[str] = mapped_column(String(300), nullable=False)
    post_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    media_type: Mapped[MediaType] = mapped_column(Enum(MediaType), nullable=False)
    caption: Mapped[str | None] = mapped_column(Text, nullable=True)
    published_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    is_pinned: Mapped[bool] = mapped_column(Boolean, default=False)
    is_quarantined: Mapped[bool] = mapped_column(Boolean, default=False)
    quarantine_reasons: Mapped[list[str]] = mapped_column(JSONB, default=list)


class MediaAsset(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "media_assets"
    __table_args__ = (
        UniqueConstraint("post_id", "position"),
        UniqueConstraint("workspace_id", "sha256", "post_id"),
    )

    workspace_id: Mapped[UUID] = mapped_column(ForeignKey("workspaces.id"), index=True)
    post_id: Mapped[UUID] = mapped_column(ForeignKey("posts.id"), index=True)
    object_key: Mapped[str] = mapped_column(String(1024), nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    mime_type: Mapped[str] = mapped_column(String(150), nullable=False)
    size_bytes: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    position: Mapped[int] = mapped_column(Integer, nullable=False)


class PostMetricSnapshot(UUIDPrimaryKeyMixin, Base):
    __tablename__ = "post_metric_snapshots"

    workspace_id: Mapped[UUID] = mapped_column(ForeignKey("workspaces.id"), index=True)
    post_id: Mapped[UUID] = mapped_column(ForeignKey("posts.id"), index=True)
    metrics: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    captured_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class StoredComment(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "stored_comments"

    workspace_id: Mapped[UUID] = mapped_column(ForeignKey("workspaces.id"), index=True)
    post_id: Mapped[UUID] = mapped_column(ForeignKey("posts.id"), index=True)
    platform_comment_id: Mapped[str | None] = mapped_column(String(300), nullable=True)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)


class ExtractionArtifact(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "extraction_artifacts"

    workspace_id: Mapped[UUID] = mapped_column(ForeignKey("workspaces.id"), index=True)
    post_id: Mapped[UUID] = mapped_column(ForeignKey("posts.id"), index=True)
    artifact_type: Mapped[ArtifactType] = mapped_column(Enum(ArtifactType), nullable=False)
    object_key: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    content: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    locator: Mapped[str] = mapped_column(String(1000), nullable=False)
    checksum: Mapped[str | None] = mapped_column(String(64), nullable=True)


class AnalysisRun(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "analysis_runs"
    __table_args__ = (UniqueConstraint("workspace_id", "fingerprint"),)

    workspace_id: Mapped[UUID] = mapped_column(ForeignKey("workspaces.id"), index=True)
    post_id: Mapped[UUID] = mapped_column(ForeignKey("posts.id"), index=True)
    fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[JobStatus] = mapped_column(Enum(JobStatus), nullable=False)
    provider: Mapped[str | None] = mapped_column(String(100), nullable=True)
    model: Mapped[str | None] = mapped_column(String(300), nullable=True)
    prompt_version: Mapped[str] = mapped_column(String(100), nullable=False)
    schema_version: Mapped[str] = mapped_column(String(30), nullable=False)
    taxonomy_version: Mapped[str] = mapped_column(String(30), nullable=False)
    attempts: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, default=list)
    error: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)


class PostObservationRecord(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "post_observations"
    __table_args__ = (UniqueConstraint("analysis_run_id"),)

    workspace_id: Mapped[UUID] = mapped_column(ForeignKey("workspaces.id"), index=True)
    post_id: Mapped[UUID] = mapped_column(ForeignKey("posts.id"), index=True)
    analysis_run_id: Mapped[UUID] = mapped_column(ForeignKey("analysis_runs.id"), index=True)
    observation: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)


class PostFacetEmbedding(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "post_facet_embeddings"
    __table_args__ = (UniqueConstraint("observation_id", "facet", "model_version"),)

    workspace_id: Mapped[UUID] = mapped_column(ForeignKey("workspaces.id"), index=True)
    post_id: Mapped[UUID] = mapped_column(ForeignKey("posts.id"), index=True)
    observation_id: Mapped[UUID] = mapped_column(ForeignKey("post_observations.id"), index=True)
    facet: Mapped[str] = mapped_column(String(100), nullable=False)
    model_version: Mapped[str] = mapped_column(String(300), nullable=False)
    embedding: Mapped[list[float]] = mapped_column(Vector(1024), nullable=False)


class TaxonomyTerm(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "taxonomy_terms"
    __table_args__ = (UniqueConstraint("version", "facet", "key"),)

    version: Mapped[str] = mapped_column(String(30), nullable=False)
    facet: Mapped[str] = mapped_column(String(100), nullable=False)
    key: Mapped[str] = mapped_column(String(100), nullable=False)
    label: Mapped[str] = mapped_column(String(200), nullable=False)
    parent_key: Mapped[str | None] = mapped_column(String(100), nullable=True)
    aliases: Mapped[list[str]] = mapped_column(JSONB, default=list)


class PostTaxonomyAssignment(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "post_taxonomy_assignments"
    __table_args__ = (UniqueConstraint("observation_id", "term_id"),)

    workspace_id: Mapped[UUID] = mapped_column(ForeignKey("workspaces.id"), index=True)
    observation_id: Mapped[UUID] = mapped_column(ForeignKey("post_observations.id"), index=True)
    term_id: Mapped[UUID] = mapped_column(ForeignKey("taxonomy_terms.id"), index=True)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    evidence: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, default=list)


class CreatorPersonaVersion(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "creator_persona_versions"
    __table_args__ = (UniqueConstraint("creator_id", "version"),)

    workspace_id: Mapped[UUID] = mapped_column(ForeignKey("workspaces.id"), index=True)
    creator_id: Mapped[UUID] = mapped_column(ForeignKey("creators.id"), index=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[PersonaStatus] = mapped_column(Enum(PersonaStatus), nullable=False)
    is_latest: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    requires_review: Mapped[bool] = mapped_column(Boolean, default=True)
    review_status: Mapped[ReviewStatus] = mapped_column(
        Enum(ReviewStatus), default=ReviewStatus.PENDING
    )
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    persona: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    verification: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    evidence_cutoff: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    approved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class CreatorPersonaTrait(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "creator_persona_traits"

    workspace_id: Mapped[UUID] = mapped_column(ForeignKey("workspaces.id"), index=True)
    persona_version_id: Mapped[UUID] = mapped_column(
        ForeignKey("creator_persona_versions.id"), index=True
    )
    facet: Mapped[str] = mapped_column(String(100), nullable=False)
    key: Mapped[str] = mapped_column(String(200), nullable=False)
    value: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    evidence: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, default=list)


class CreatorFacetEmbedding(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "creator_facet_embeddings"
    __table_args__ = (UniqueConstraint("persona_version_id", "facet", "model_version"),)

    workspace_id: Mapped[UUID] = mapped_column(ForeignKey("workspaces.id"), index=True)
    creator_id: Mapped[UUID] = mapped_column(ForeignKey("creators.id"), index=True)
    persona_version_id: Mapped[UUID] = mapped_column(
        ForeignKey("creator_persona_versions.id"), index=True
    )
    facet: Mapped[str] = mapped_column(String(100), nullable=False)
    model_version: Mapped[str] = mapped_column(String(300), nullable=False)
    embedding: Mapped[list[float]] = mapped_column(Vector(1024), nullable=False)


class CreatorRelationship(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "creator_relationships"
    __table_args__ = (
        UniqueConstraint("persona_version_id", "related_persona_version_id"),
        Index("ix_relationship_creator_overall", "creator_id", "overall_similarity"),
    )

    workspace_id: Mapped[UUID] = mapped_column(ForeignKey("workspaces.id"), index=True)
    creator_id: Mapped[UUID] = mapped_column(ForeignKey("creators.id"), index=True)
    related_creator_id: Mapped[UUID] = mapped_column(ForeignKey("creators.id"), index=True)
    persona_version_id: Mapped[UUID] = mapped_column(
        ForeignKey("creator_persona_versions.id"), index=True
    )
    related_persona_version_id: Mapped[UUID] = mapped_column(
        ForeignKey("creator_persona_versions.id"), index=True
    )
    overall_similarity: Mapped[float] = mapped_column(Float, nullable=False)
    content_similarity: Mapped[float] = mapped_column(Float, nullable=False)
    visual_similarity: Mapped[float] = mapped_column(Float, nullable=False)
    communication_similarity: Mapped[float] = mapped_column(Float, nullable=False)
    commercial_similarity: Mapped[float] = mapped_column(Float, nullable=False)
    shared_terms: Mapped[list[str]] = mapped_column(JSONB, default=list)
    differences: Mapped[list[str]] = mapped_column(JSONB, default=list)


class PersonaReview(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "persona_reviews"

    workspace_id: Mapped[UUID] = mapped_column(ForeignKey("workspaces.id"), index=True)
    persona_version_id: Mapped[UUID] = mapped_column(
        ForeignKey("creator_persona_versions.id"), index=True
    )
    status: Mapped[ReviewStatus] = mapped_column(Enum(ReviewStatus), nullable=False)
    reviewer: Mapped[str] = mapped_column(String(300), nullable=False)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)


class WorkflowRun(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "workflow_runs"

    workspace_id: Mapped[UUID] = mapped_column(ForeignKey("workspaces.id"), index=True)
    kind: Mapped[WorkflowKind] = mapped_column(Enum(WorkflowKind), nullable=False)
    subject_id: Mapped[UUID | None] = mapped_column(Uuid, nullable=True, index=True)
    status: Mapped[JobStatus] = mapped_column(Enum(JobStatus), nullable=False)
    state: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    parent_run_id: Mapped[UUID | None] = mapped_column(ForeignKey("workflow_runs.id"))


class AgentStep(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "agent_steps"

    workspace_id: Mapped[UUID] = mapped_column(ForeignKey("workspaces.id"), index=True)
    workflow_run_id: Mapped[UUID] = mapped_column(ForeignKey("workflow_runs.id"), index=True)
    agent: Mapped[str] = mapped_column(String(100), nullable=False)
    step_number: Mapped[int] = mapped_column(Integer, nullable=False)
    kind: Mapped[str] = mapped_column(String(100), nullable=False)
    input: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    output: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    provider: Mapped[str | None] = mapped_column(String(100), nullable=True)
    model: Mapped[str | None] = mapped_column(String(300), nullable=True)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)


class ProcessingJob(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "processing_jobs"

    workspace_id: Mapped[UUID] = mapped_column(ForeignKey("workspaces.id"), index=True)
    kind: Mapped[WorkflowKind] = mapped_column(Enum(WorkflowKind), nullable=False)
    status: Mapped[JobStatus] = mapped_column(Enum(JobStatus), nullable=False)
    subject_id: Mapped[UUID | None] = mapped_column(Uuid, nullable=True, index=True)
    celery_task_id: Mapped[str | None] = mapped_column(String(300), nullable=True)
    progress_current: Mapped[int] = mapped_column(Integer, default=0)
    progress_total: Mapped[int] = mapped_column(Integer, default=0)
    result: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    error: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)


class DatasetExport(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "dataset_exports"

    workspace_id: Mapped[UUID] = mapped_column(ForeignKey("workspaces.id"), index=True)
    job_id: Mapped[UUID] = mapped_column(ForeignKey("processing_jobs.id"), index=True)
    status: Mapped[JobStatus] = mapped_column(Enum(JobStatus), nullable=False)
    schema_version: Mapped[str] = mapped_column(String(30), nullable=False)
    selection: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    object_keys: Mapped[dict[str, str]] = mapped_column(JSONB, default=dict)
    row_counts: Mapped[dict[str, int]] = mapped_column(JSONB, default=dict)
