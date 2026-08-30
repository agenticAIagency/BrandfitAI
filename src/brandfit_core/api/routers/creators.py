from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from brandfit_core.api.dependencies import workspace_id
from brandfit_core.db.models import (
    Creator,
    CreatorPersonaVersion,
    CreatorProfileSnapshot,
    CreatorRelationship,
    ExtractionArtifact,
    Post,
    PostObservationRecord,
    ProcessingJob,
)
from brandfit_core.db.session import get_session
from brandfit_core.domain.enums import JobStatus, ReviewStatus, WorkflowKind
from brandfit_core.domain.jobs import JobRead
from brandfit_core.domain.persona import (
    CreatorRelationshipRead,
    PersonaProposal,
    PersonaReviewCreate,
    PersonaReviewRead,
    PersonaVerificationReport,
    PersonaVersionRead,
)
from brandfit_core.services.publisher import review_persona

router = APIRouter(prefix="/v1/creators", tags=["creators"])


def _persona_read(record: CreatorPersonaVersion) -> PersonaVersionRead:
    return PersonaVersionRead(
        id=record.id,
        creator_id=record.creator_id,
        version=record.version,
        status=record.status,
        is_latest=record.is_latest,
        summary=record.summary,
        persona=PersonaProposal.model_validate(record.persona),
        verification=PersonaVerificationReport.model_validate(record.verification),
        created_at=record.created_at,
        approved_at=record.approved_at,
    )


@router.get("/{creator_id}/persona", response_model=PersonaVersionRead)
async def current_persona(
    creator_id: UUID,
    workspace: UUID = Depends(workspace_id),
    session: AsyncSession = Depends(get_session),
) -> PersonaVersionRead:
    record = await session.scalar(
        select(CreatorPersonaVersion).where(
            CreatorPersonaVersion.creator_id == creator_id,
            CreatorPersonaVersion.workspace_id == workspace,
            CreatorPersonaVersion.is_latest.is_(True),
        )
    )
    if not record:
        raise HTTPException(status_code=404, detail="No approved current persona")
    return _persona_read(record)


@router.get("/{creator_id}/persona/versions", response_model=list[PersonaVersionRead])
async def persona_versions(
    creator_id: UUID,
    workspace: UUID = Depends(workspace_id),
    session: AsyncSession = Depends(get_session),
) -> list[PersonaVersionRead]:
    records = list(
        (
            await session.scalars(
                select(CreatorPersonaVersion)
                .where(
                    CreatorPersonaVersion.creator_id == creator_id,
                    CreatorPersonaVersion.workspace_id == workspace,
                )
                .order_by(CreatorPersonaVersion.version.desc())
            )
        ).all()
    )
    return [_persona_read(item) for item in records]


@router.get("/{creator_id}/evidence")
async def creator_evidence(
    creator_id: UUID,
    workspace: UUID = Depends(workspace_id),
    session: AsyncSession = Depends(get_session),
) -> dict[str, object]:
    creator = await session.scalar(
        select(Creator).where(Creator.id == creator_id, Creator.workspace_id == workspace)
    )
    if not creator:
        raise HTTPException(status_code=404, detail="Creator not found")
    profile = await session.scalar(
        select(CreatorProfileSnapshot)
        .where(CreatorProfileSnapshot.creator_id == creator_id)
        .order_by(CreatorProfileSnapshot.captured_at.desc())
        .limit(1)
    )
    posts = list(
        (
            await session.scalars(
                select(Post).where(Post.creator_id == creator_id).order_by(Post.published_at.desc())
            )
        ).all()
    )
    post_ids = [item.id for item in posts]
    observations = (
        list(
            (
                await session.scalars(
                    select(PostObservationRecord).where(PostObservationRecord.post_id.in_(post_ids))
                )
            ).all()
        )
        if post_ids
        else []
    )
    artifacts = (
        list(
            (
                await session.scalars(
                    select(ExtractionArtifact).where(ExtractionArtifact.post_id.in_(post_ids))
                )
            ).all()
        )
        if post_ids
        else []
    )
    return {
        "creator": {
            "id": str(creator.id),
            "platform_user_id": creator.platform_user_id,
            "username": creator.current_username,
        },
        "profile": profile.profile if profile else None,
        "posts": [
            {
                "id": str(item.id),
                "platform_post_id": item.platform_post_id,
                "media_type": item.media_type,
                "caption": item.caption,
                "published_at": item.published_at.isoformat(),
            }
            for item in posts
        ],
        "observations": [item.observation for item in observations],
        "artifacts": [
            {
                "id": str(item.id),
                "post_id": str(item.post_id),
                "type": item.artifact_type,
                "locator": item.locator,
                "object_key": item.object_key,
            }
            for item in artifacts
        ],
        "comments_policy": "stored_but_excluded_from_persona_generation",
    }


@router.get("/{creator_id}/relationships", response_model=list[CreatorRelationshipRead])
async def creator_relationships(
    creator_id: UUID,
    workspace: UUID = Depends(workspace_id),
    session: AsyncSession = Depends(get_session),
) -> list[CreatorRelationshipRead]:
    values = list(
        (
            await session.scalars(
                select(CreatorRelationship)
                .where(
                    CreatorRelationship.creator_id == creator_id,
                    CreatorRelationship.workspace_id == workspace,
                )
                .order_by(CreatorRelationship.overall_similarity.desc())
                .limit(20)
            )
        ).all()
    )
    return [CreatorRelationshipRead.model_validate(item, from_attributes=True) for item in values]


@router.post("/{creator_id}/rebuild", response_model=JobRead, status_code=status.HTTP_202_ACCEPTED)
async def rebuild(
    creator_id: UUID,
    workspace: UUID = Depends(workspace_id),
    session: AsyncSession = Depends(get_session),
) -> JobRead:
    creator = await session.scalar(
        select(Creator).where(Creator.id == creator_id, Creator.workspace_id == workspace)
    )
    if not creator:
        raise HTTPException(status_code=404, detail="Creator not found")
    total = await session.scalar(select(func.count(Post.id)).where(Post.creator_id == creator_id))
    job = ProcessingJob(
        workspace_id=workspace,
        kind=WorkflowKind.CREATOR_PERSONA,
        status=JobStatus.QUEUED,
        subject_id=creator_id,
        progress_total=int(total or 0),
    )
    session.add(job)
    await session.commit()
    await session.refresh(job)
    from brandfit_core.workers.tasks import rebuild_creator

    task = rebuild_creator.delay(str(job.id), str(creator_id))
    job.celery_task_id = task.id
    await session.commit()
    return JobRead.model_validate(job, from_attributes=True)


@router.post("/{creator_id}/reviews", response_model=PersonaReviewRead)
async def submit_review(
    creator_id: UUID,
    review: PersonaReviewCreate,
    workspace: UUID = Depends(workspace_id),
    session: AsyncSession = Depends(get_session),
) -> PersonaReviewRead:
    pending = await session.scalar(
        select(CreatorPersonaVersion)
        .where(
            CreatorPersonaVersion.creator_id == creator_id,
            CreatorPersonaVersion.workspace_id == workspace,
            CreatorPersonaVersion.requires_review.is_(True),
            CreatorPersonaVersion.review_status == ReviewStatus.PENDING,
        )
        .order_by(CreatorPersonaVersion.version.desc())
        .limit(1)
    )
    if not pending:
        raise HTTPException(status_code=404, detail="No pending persona version")
    try:
        result = await review_persona(session, workspace, pending.id, review)
        await session.commit()
        await session.refresh(result)
    except ValueError as exc:
        await session.rollback()
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return PersonaReviewRead.model_validate(result, from_attributes=True)
