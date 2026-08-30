from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import exists, select

from brandfit_core.db.models import (
    Creator,
    CreatorPersonaVersion,
    CreatorProfileSnapshot,
    Post,
    PostMetricSnapshot,
    ProcessingJob,
)
from brandfit_core.db.session import SessionFactory
from brandfit_core.domain.enums import JobStatus, ReviewStatus, WorkflowKind
from brandfit_core.services.exports import generate_export
from brandfit_core.services.workflow_runner import run_creator_job, run_ingestion_job
from brandfit_core.workers.celery_app import celery_app

_worker_loop: asyncio.AbstractEventLoop | None = None


def _run_async[T](awaitable: Awaitable[T]) -> T:
    """Run async work on one event loop for the lifetime of a Celery process.

    Async SQLAlchemy pools bind their connections to the event loop that created
    them.  Creating a fresh loop with ``asyncio.run`` for every Celery task can
    therefore return a pooled asyncpg connection owned by a closed loop.
    """
    global _worker_loop
    if _worker_loop is None or _worker_loop.is_closed():
        _worker_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(_worker_loop)
    return _worker_loop.run_until_complete(awaitable)


@celery_app.task(name="brandfit.process_ingestion_batch", bind=True, max_retries=2)
def process_ingestion_batch(self: object, batch_id: str) -> dict[str, object]:
    return _run_async(run_ingestion_job(UUID(batch_id)))


@celery_app.task(name="brandfit.rebuild_creator", bind=True, max_retries=2)
def rebuild_creator(self: object, job_id: str, creator_id: str) -> dict[str, object]:
    return _run_async(run_creator_job(UUID(job_id), UUID(creator_id)))


async def _queue_weekly_refreshes() -> list[str]:
    queued: list[str] = []
    async with SessionFactory() as session:
        creators = list((await session.scalars(select(Creator))).all())
        for creator in creators:
            pending_review = await session.scalar(
                select(
                    exists().where(
                        CreatorPersonaVersion.creator_id == creator.id,
                        CreatorPersonaVersion.review_status == ReviewStatus.PENDING,
                    )
                )
            )
            active_refresh = await session.scalar(
                select(
                    exists().where(
                        ProcessingJob.kind == WorkflowKind.CREATOR_PERSONA,
                        ProcessingJob.subject_id == creator.id,
                        ProcessingJob.status.in_([JobStatus.QUEUED, JobStatus.RUNNING]),
                    )
                )
            )
            if pending_review or active_refresh:
                continue
            latest = await session.scalar(
                select(CreatorPersonaVersion)
                .where(
                    CreatorPersonaVersion.creator_id == creator.id,
                    CreatorPersonaVersion.is_latest.is_(True),
                )
                .limit(1)
            )
            if latest is None:
                has_new = True
            else:
                new_posts = await session.scalar(
                    select(
                        exists().where(
                            Post.creator_id == creator.id,
                            Post.created_at > latest.evidence_cutoff,
                        )
                    )
                )
                new_metrics = await session.scalar(
                    select(
                        exists().where(
                            PostMetricSnapshot.post_id.in_(
                                select(Post.id).where(Post.creator_id == creator.id)
                            ),
                            PostMetricSnapshot.captured_at > latest.evidence_cutoff,
                        )
                    )
                )
                new_profile = await session.scalar(
                    select(
                        exists().where(
                            CreatorProfileSnapshot.creator_id == creator.id,
                            CreatorProfileSnapshot.captured_at > latest.evidence_cutoff,
                        )
                    )
                )
                has_new = bool(new_posts or new_metrics or new_profile)
            if not has_new:
                continue
            job = ProcessingJob(
                workspace_id=creator.workspace_id,
                kind=WorkflowKind.CREATOR_PERSONA,
                status=JobStatus.QUEUED,
                subject_id=creator.id,
                progress_total=1,
                result={"scheduled_at": datetime.now(UTC).isoformat(), "reason": "new_evidence"},
            )
            session.add(job)
            await session.flush()
            queued.append(str(job.id))
            await session.commit()
            task = rebuild_creator.delay(str(job.id), str(creator.id))
            job.celery_task_id = task.id
            await session.commit()
    return queued


@celery_app.task(name="brandfit.weekly_persona_refresh")
def weekly_persona_refresh() -> list[str]:
    return _run_async(_queue_weekly_refreshes())


@celery_app.task(name="brandfit.generate_export", bind=True, max_retries=2)
def generate_dataset_export(self: object, export_id: str) -> dict[str, str]:
    return _run_async(generate_export(UUID(export_id)))
