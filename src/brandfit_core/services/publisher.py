from __future__ import annotations

import json
from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from brandfit_core.config import Settings, get_settings
from brandfit_core.db.models import (
    Creator,
    CreatorFacetEmbedding,
    CreatorPersonaTrait,
    CreatorPersonaVersion,
    PersonaReview,
)
from brandfit_core.domain.enums import ReviewStatus
from brandfit_core.domain.persona import (
    PersonaProposal,
    PersonaReviewCreate,
    PersonaVerificationReport,
)
from brandfit_core.providers.embedding import configured_embedder
from brandfit_core.services.relationships import recompute_relationships


def _facet_text(proposal: PersonaProposal, facet: str) -> str:
    return json.dumps(getattr(proposal, f"{facet}_fingerprint"), sort_keys=True, ensure_ascii=False)


async def publish_persona(
    session: AsyncSession,
    proposal: PersonaProposal,
    verification: PersonaVerificationReport,
    *,
    evidence_cutoff: datetime,
    settings: Settings | None = None,
) -> CreatorPersonaVersion:
    if not verification.accepted or any(not item.accepted for item in verification.claims):
        raise ValueError("Deterministic publisher accepts only fully verified typed output")
    settings = settings or get_settings()
    creator = await session.scalar(
        select(Creator).where(Creator.id == proposal.creator_id).with_for_update()
    )
    if creator is None:
        raise ValueError("Creator does not exist")
    maximum = await session.scalar(
        select(func.max(CreatorPersonaVersion.version)).where(
            CreatorPersonaVersion.creator_id == proposal.creator_id
        )
    )
    version = int(maximum or 0) + 1
    first_version = version == 1
    requires_review = first_version or verification.requires_human_review
    record = CreatorPersonaVersion(
        workspace_id=creator.workspace_id,
        creator_id=creator.id,
        version=version,
        status=proposal.status,
        is_latest=not requires_review,
        requires_review=requires_review,
        review_status=ReviewStatus.PENDING if requires_review else ReviewStatus.APPROVED,
        summary=proposal.summary,
        persona=proposal.model_dump(mode="json"),
        verification=verification.model_dump(mode="json"),
        evidence_cutoff=evidence_cutoff,
        approved_at=None if requires_review else datetime.now(UTC),
    )
    if not requires_review:
        await session.execute(
            update(CreatorPersonaVersion)
            .where(CreatorPersonaVersion.creator_id == creator.id)
            .values(is_latest=False)
        )
        creator.latest_persona_version = version
    session.add(record)
    await session.flush()
    for facet in ["content", "communication", "visual", "commercial"]:
        for key, value in getattr(proposal, f"{facet}_fingerprint").items():
            session.add(
                CreatorPersonaTrait(
                    workspace_id=creator.workspace_id,
                    persona_version_id=record.id,
                    facet=facet,
                    key=key,
                    value={"share": value},
                    confidence=proposal.confidence.overall,
                    evidence=[],
                )
            )
    embedder = configured_embedder(settings)
    facets = ["content", "visual", "communication", "commercial"]
    vectors = embedder.embed([_facet_text(proposal, facet) for facet in facets])
    for facet, vector in zip(facets, vectors, strict=True):
        session.add(
            CreatorFacetEmbedding(
                workspace_id=creator.workspace_id,
                creator_id=creator.id,
                persona_version_id=record.id,
                facet=facet,
                model_version=embedder.model_version,
                embedding=vector,
            )
        )
    await session.flush()
    if record.is_latest:
        await recompute_relationships(session, creator.workspace_id, record.id)
    return record


async def review_persona(
    session: AsyncSession,
    workspace_id: UUID,
    persona_version_id: UUID,
    review: PersonaReviewCreate,
) -> PersonaReview:
    persona = await session.scalar(
        select(CreatorPersonaVersion)
        .where(
            CreatorPersonaVersion.id == persona_version_id,
            CreatorPersonaVersion.workspace_id == workspace_id,
        )
        .with_for_update()
    )
    if persona is None:
        raise ValueError("Persona version not found")
    status = ReviewStatus(review.status)
    record = PersonaReview(
        workspace_id=workspace_id,
        persona_version_id=persona.id,
        status=status,
        reviewer=review.reviewer,
        notes=review.notes,
    )
    session.add(record)
    persona.review_status = status
    if status == ReviewStatus.APPROVED:
        await session.execute(
            update(CreatorPersonaVersion)
            .where(CreatorPersonaVersion.creator_id == persona.creator_id)
            .values(is_latest=False)
        )
        persona.is_latest = True
        persona.approved_at = datetime.now(UTC)
        creator = await session.scalar(
            select(Creator).where(Creator.id == persona.creator_id).with_for_update()
        )
        if creator:
            creator.latest_persona_version = persona.version
        await session.flush()
        await recompute_relationships(session, workspace_id, persona.id)
    await session.flush()
    return record
