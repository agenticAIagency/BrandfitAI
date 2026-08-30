from __future__ import annotations

import asyncio
import hashlib
import subprocess
import tempfile
from pathlib import Path
from typing import Any
from uuid import UUID

from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from brandfit_core.config import Settings, get_settings
from brandfit_core.db.models import (
    Creator,
    CreatorProfileSnapshot,
    IngestionBatch,
    MediaAsset,
    Post,
    PostMetricSnapshot,
    ProcessingJob,
    StoredComment,
)
from brandfit_core.domain.enums import BatchStatus, JobStatus, WorkflowKind
from brandfit_core.domain.ingestion import (
    CreatorManifest,
    IngestionBatchRead,
    PostManifest,
    QuarantinedPost,
)
from brandfit_core.storage import ObjectStore


class ManifestEnvelopeError(ValueError):
    pass


def _required_string(payload: dict[str, Any], key: str, maximum: int) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise ManifestEnvelopeError(f"{key} must be a non-empty string up to {maximum} characters")
    return value


async def _media_errors(store: ObjectStore, post: PostManifest) -> list[str]:
    errors: list[str] = []
    for media in post.media:
        try:
            data = await asyncio.to_thread(store.get_bytes, media.object_key)
        except Exception:
            errors.append(f"media object is missing or unreadable: {media.object_key}")
            continue
        actual = hashlib.sha256(data).hexdigest()
        if actual.lower() != media.sha256.lower():
            errors.append(f"checksum mismatch: {media.object_key}")
        if media.size_bytes is not None and len(data) != media.size_bytes:
            errors.append(f"size mismatch: {media.object_key}")
        try:
            await asyncio.to_thread(_validate_decodable, data, media.mime_type)
        except Exception:
            errors.append(f"media is corrupt or unsupported: {media.object_key}")
    return errors


def _validate_decodable(data: bytes, mime_type: str) -> None:
    if mime_type.startswith("image/"):
        from PIL import Image

        with Image.open(__import__("io").BytesIO(data)) as image:
            image.verify()
        return
    if mime_type.startswith("video/"):
        suffix = ".mp4" if "mp4" in mime_type else ".bin"
        with tempfile.TemporaryDirectory(prefix="brandfit-validate-") as directory:
            path = Path(directory) / f"media{suffix}"
            path.write_bytes(data)
            subprocess.run(  # noqa: S603
                ["ffprobe", "-v", "error", "-show_entries", "format=duration", str(path)],
                capture_output=True,
                check=True,
            )


def _batch_read(record: IngestionBatch) -> IngestionBatchRead:
    return IngestionBatchRead(
        id=record.id,
        workspace_id=record.workspace_id,
        source=record.source,
        source_run_id=record.source_run_id,
        status=record.status,
        accepted_posts=record.accepted_posts,
        duplicate_posts=record.duplicate_posts,
        quarantined_posts=[
            QuarantinedPost.model_validate(item) for item in record.quarantined_posts
        ],
        job_id=record.job_id,
        created_at=record.created_at,
    )


async def ingest_manifest(
    session: AsyncSession,
    payload: dict[str, Any],
    *,
    store: ObjectStore | None = None,
    settings: Settings | None = None,
    verify_objects: bool = True,
) -> tuple[IngestionBatchRead, bool]:
    """Validate a loose envelope, quarantine bad posts, and atomically commit valid evidence.

    The boolean indicates whether a new batch was created. Replaying source_run_id returns the
    original batch and never creates duplicate downstream work.
    """

    settings = settings or get_settings()
    workspace_id = UUID(str(payload.get("workspace_id") or settings.default_workspace_id))
    source = str(payload.get("source") or "instagram")
    source_run_id = _required_string(payload, "source_run_id", 300)
    schema_version = str(payload.get("schema_version") or "1.0")
    if not schema_version.startswith("1."):
        raise ManifestEnvelopeError("Only ingestion schema 1.x is supported")
    try:
        creator_manifest = CreatorManifest.model_validate(payload.get("creator"))
    except ValidationError as exc:
        raise ManifestEnvelopeError(f"Invalid creator: {exc}") from exc
    raw_posts = payload.get("posts")
    if not isinstance(raw_posts, list) or not raw_posts:
        raise ManifestEnvelopeError("posts must be a non-empty list")

    existing = await session.scalar(
        select(IngestionBatch).where(
            IngestionBatch.workspace_id == workspace_id,
            IngestionBatch.source == source,
            IngestionBatch.source_run_id == source_run_id,
        )
    )
    if existing is not None:
        return _batch_read(existing), False

    valid_posts: list[PostManifest] = []
    quarantined: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    object_store = store or ObjectStore(settings)
    for index, raw_post in enumerate(raw_posts):
        post_id = (
            raw_post.get("platform_post_id", f"index:{index}")
            if isinstance(raw_post, dict)
            else f"index:{index}"
        )
        try:
            parsed_post = PostManifest.model_validate(raw_post)
        except ValidationError as exc:
            quarantined.append({"platform_post_id": str(post_id), "reasons": [str(exc)]})
            continue
        if parsed_post.platform_post_id in seen_ids:
            quarantined.append(
                {
                    "platform_post_id": parsed_post.platform_post_id,
                    "reasons": ["duplicate ID in manifest"],
                }
            )
            continue
        seen_ids.add(parsed_post.platform_post_id)
        media_errors = (
            await _media_errors(object_store, parsed_post) if verify_objects else []
        )
        if media_errors:
            quarantined.append(
                {"platform_post_id": parsed_post.platform_post_id, "reasons": media_errors}
            )
        else:
            valid_posts.append(parsed_post)

    batch = IngestionBatch(
        workspace_id=workspace_id,
        source=source,
        source_run_id=source_run_id,
        schema_version=schema_version,
        status=BatchStatus.VALIDATING,
        quarantined_posts=quarantined,
        raw_manifest=payload,
    )
    session.add(batch)
    await session.flush()

    creator = await session.scalar(
        select(Creator).where(
            Creator.workspace_id == workspace_id,
            Creator.platform == source,
            Creator.platform_user_id == creator_manifest.platform_user_id,
        )
    )
    if creator is None:
        creator = Creator(
            workspace_id=workspace_id,
            platform=source,
            platform_user_id=creator_manifest.platform_user_id,
            current_username=creator_manifest.username,
        )
        session.add(creator)
        await session.flush()
    else:
        creator.current_username = creator_manifest.username

    session.add(
        CreatorProfileSnapshot(
            workspace_id=workspace_id,
            creator_id=creator.id,
            username=creator_manifest.username,
            profile=creator_manifest.profile.model_dump(mode="json"),
            captured_at=creator_manifest.profile.captured_at,
        )
    )

    accepted = 0
    duplicates = 0
    for post_manifest in valid_posts:
        duplicate_id = await session.scalar(
            select(Post.id).where(
                Post.workspace_id == workspace_id,
                Post.platform == source,
                Post.platform_post_id == post_manifest.platform_post_id,
            )
        )
        checksum_match = await session.scalar(
            select(Post.id)
            .join(MediaAsset, MediaAsset.post_id == Post.id)
            .where(
                MediaAsset.workspace_id == workspace_id,
                MediaAsset.sha256 == post_manifest.media[0].sha256.lower(),
                Post.creator_id == creator.id,
            )
            .limit(1)
        )
        canonical_post_id = duplicate_id or checksum_match
        if canonical_post_id is not None:
            if post_manifest.metrics:
                metrics = post_manifest.metrics
                metric_exists = await session.scalar(
                    select(PostMetricSnapshot.id).where(
                        PostMetricSnapshot.post_id == canonical_post_id,
                        PostMetricSnapshot.captured_at == metrics.captured_at,
                    )
                )
                if metric_exists is None:
                    session.add(
                        PostMetricSnapshot(
                            workspace_id=workspace_id,
                            post_id=canonical_post_id,
                            metrics=metrics.model_dump(mode="json", exclude={"captured_at"}),
                            captured_at=metrics.captured_at,
                        )
                    )
            # Duplicate posts can still carry newly captured comments. They
            # remain audit evidence and never enter persona prompts.
            for comment in post_manifest.comments:
                comment_exists = None
                if comment.platform_comment_id:
                    comment_exists = await session.scalar(
                        select(StoredComment.id).where(
                            StoredComment.post_id == canonical_post_id,
                            StoredComment.platform_comment_id == comment.platform_comment_id,
                        )
                    )
                if comment_exists is None:
                    session.add(
                        StoredComment(
                            workspace_id=workspace_id,
                            post_id=canonical_post_id,
                            platform_comment_id=comment.platform_comment_id,
                            text=comment.text,
                            metadata_json=comment.model_dump(
                                mode="json", exclude={"text", "platform_comment_id"}
                            ),
                        )
                    )
            duplicates += 1
            continue
        db_post = Post(
            workspace_id=workspace_id,
            creator_id=creator.id,
            ingestion_batch_id=batch.id,
            platform=source,
            platform_post_id=post_manifest.platform_post_id,
            post_url=post_manifest.post_url,
            media_type=post_manifest.media_type,
            caption=post_manifest.caption,
            published_at=post_manifest.published_at,
            is_pinned=post_manifest.is_pinned,
        )
        session.add(db_post)
        await session.flush()
        for item in post_manifest.media:
            session.add(
                MediaAsset(
                    workspace_id=workspace_id,
                    post_id=db_post.id,
                    object_key=item.object_key,
                    sha256=item.sha256.lower(),
                    mime_type=item.mime_type,
                    size_bytes=item.size_bytes,
                    position=item.position,
                )
            )
        if post_manifest.metrics:
            metrics = post_manifest.metrics
            session.add(
                PostMetricSnapshot(
                    workspace_id=workspace_id,
                    post_id=db_post.id,
                    metrics=metrics.model_dump(mode="json", exclude={"captured_at"}),
                    captured_at=metrics.captured_at,
                )
            )
        # Comments are retained for audit/export but no persona service queries this table.
        for comment in post_manifest.comments:
            session.add(
                StoredComment(
                    workspace_id=workspace_id,
                    post_id=db_post.id,
                    platform_comment_id=comment.platform_comment_id,
                    text=comment.text,
                    metadata_json=comment.model_dump(
                        mode="json", exclude={"text", "platform_comment_id"}
                    ),
                )
            )
        accepted += 1

    if accepted and quarantined:
        batch.status = BatchStatus.PARTIAL
    elif accepted:
        batch.status = BatchStatus.ACCEPTED
    elif quarantined:
        batch.status = BatchStatus.QUARANTINED
    else:
        batch.status = BatchStatus.ACCEPTED
    batch.accepted_posts = accepted
    batch.duplicate_posts = duplicates

    job = ProcessingJob(
        workspace_id=workspace_id,
        kind=WorkflowKind.INGESTION,
        status=JobStatus.QUEUED if accepted else JobStatus.COMPLETED,
        subject_id=batch.id,
        progress_total=accepted,
        result={"creator_id": str(creator.id), "accepted_posts": accepted}
        if not accepted
        else None,
    )
    session.add(job)
    await session.flush()
    batch.job_id = job.id
    await session.flush()
    return _batch_read(batch), True


async def get_batch(
    session: AsyncSession, workspace_id: UUID, batch_id: UUID
) -> IngestionBatchRead | None:
    record = await session.scalar(
        select(IngestionBatch).where(
            IngestionBatch.id == batch_id, IngestionBatch.workspace_id == workspace_id
        )
    )
    return _batch_read(record) if record else None
