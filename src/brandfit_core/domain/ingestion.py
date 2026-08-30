from __future__ import annotations

from datetime import datetime
from typing import Annotated
from uuid import UUID

from pydantic import BaseModel, Field, StringConstraints, model_validator

from brandfit_core.domain.enums import BatchStatus, MediaType

Sha256 = Annotated[str, StringConstraints(pattern=r"^[a-fA-F0-9]{64}$")]


class ProfileSnapshotManifest(BaseModel):
    display_name: str | None = Field(None, max_length=300)
    bio: str | None = Field(None, max_length=10_000)
    followers: int | None = Field(None, ge=0)
    following: int | None = Field(None, ge=0)
    post_count: int | None = Field(None, ge=0)
    profile_image_url: str | None = None
    captured_at: datetime


class MediaObjectManifest(BaseModel):
    object_key: str = Field(min_length=1, max_length=1024)
    sha256: Sha256
    mime_type: str = Field(min_length=3, max_length=150)
    size_bytes: int | None = Field(None, ge=0)
    position: int = Field(0, ge=0)


class MetricSnapshotManifest(BaseModel):
    likes: int | None = Field(None, ge=0)
    comments: int | None = Field(None, ge=0)
    views: int | None = Field(None, ge=0)
    shares: int | None = Field(None, ge=0)
    saves: int | None = Field(None, ge=0)
    captured_at: datetime


class CommentManifest(BaseModel):
    platform_comment_id: str | None = Field(None, max_length=300)
    text: str = Field(min_length=1, max_length=10_000)
    like_count: int | None = Field(None, ge=0)
    published_at: datetime | None = None


class PostManifest(BaseModel):
    platform_post_id: str = Field(min_length=1, max_length=300)
    post_url: str | None = None
    media_type: MediaType
    caption: str | None = Field(None, max_length=50_000)
    published_at: datetime
    is_pinned: bool = False
    media: list[MediaObjectManifest] = Field(min_length=1)
    metrics: MetricSnapshotManifest | None = None
    comments: list[CommentManifest] = Field(default_factory=list, max_length=500)

    @model_validator(mode="after")
    def validate_media_positions(self) -> PostManifest:
        positions = [item.position for item in self.media]
        if len(positions) != len(set(positions)):
            raise ValueError("Media positions must be unique within a post")
        return self


class CreatorManifest(BaseModel):
    platform_user_id: str = Field(min_length=1, max_length=300)
    username: str = Field(min_length=1, max_length=300)
    profile: ProfileSnapshotManifest


class IngestionManifest(BaseModel):
    schema_version: str = Field("1.0", pattern=r"^1\.\d+$")
    source: str = Field("instagram", min_length=1, max_length=100)
    source_run_id: str = Field(min_length=1, max_length=300)
    workspace_id: UUID | None = None
    creator: CreatorManifest
    posts: list[PostManifest] = Field(min_length=1, max_length=10_000)

    @model_validator(mode="after")
    def validate_unique_posts(self) -> IngestionManifest:
        ids = [post.platform_post_id for post in self.posts]
        if len(ids) != len(set(ids)):
            raise ValueError("platform_post_id must be unique within a manifest")
        return self


class QuarantinedPost(BaseModel):
    platform_post_id: str
    reasons: list[str]


class IngestionBatchRead(BaseModel):
    id: UUID
    workspace_id: UUID
    source: str
    source_run_id: str
    status: BatchStatus
    accepted_posts: int
    duplicate_posts: int
    quarantined_posts: list[QuarantinedPost]
    job_id: UUID | None = None
    created_at: datetime
