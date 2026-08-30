from __future__ import annotations

import math
from collections import defaultdict
from uuid import UUID

from sqlalchemy import delete, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from brandfit_core.db.models import (
    CreatorFacetEmbedding,
    CreatorPersonaVersion,
    CreatorRelationship,
)


def _cosine(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right, strict=True))
    denominator = math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right))
    return numerator / denominator if denominator else 0.0


def _terms(persona: dict[str, object]) -> set[str]:
    values: set[str] = set()
    for facet in [
        "content_fingerprint",
        "communication_fingerprint",
        "visual_fingerprint",
        "commercial_fingerprint",
    ]:
        fingerprint = persona.get(facet)
        if isinstance(fingerprint, dict):
            values.update(str(key) for key in fingerprint)
    return values


async def recompute_relationships(
    session: AsyncSession, workspace_id: UUID, persona_version_id: UUID
) -> int:
    source = await session.scalar(
        select(CreatorPersonaVersion).where(CreatorPersonaVersion.id == persona_version_id)
    )
    if not source or not source.is_latest:
        return 0
    personas = list(
        (
            await session.scalars(
                select(CreatorPersonaVersion).where(
                    CreatorPersonaVersion.workspace_id == workspace_id,
                    CreatorPersonaVersion.is_latest.is_(True),
                    CreatorPersonaVersion.id != source.id,
                )
            )
        ).all()
    )
    embeddings = (
        list(
            (
                await session.scalars(
                    select(CreatorFacetEmbedding).where(
                        CreatorFacetEmbedding.persona_version_id.in_(
                            [source.id] + [item.id for item in personas]
                        )
                    )
                )
            ).all()
        )
        if personas
        else []
    )
    by_persona: dict[UUID, dict[str, list[float]]] = defaultdict(dict)
    for item in embeddings:
        by_persona[item.persona_version_id][item.facet] = list(item.embedding)
    candidates: list[tuple[float, CreatorPersonaVersion, dict[str, float]]] = []
    weights = {"content": 0.35, "visual": 0.25, "communication": 0.2, "commercial": 0.2}
    for target in personas:
        similarities = {
            facet: _cosine(
                by_persona[source.id].get(facet, []), by_persona[target.id].get(facet, [])
            )
            if by_persona[source.id].get(facet) and by_persona[target.id].get(facet)
            else 0.0
            for facet in weights
        }
        overall = sum(similarities[facet] * weight for facet, weight in weights.items())
        candidates.append((overall, target, similarities))
    candidates.sort(key=lambda item: item[0], reverse=True)
    await session.execute(
        delete(CreatorRelationship).where(
            or_(
                CreatorRelationship.persona_version_id == source.id,
                CreatorRelationship.related_persona_version_id == source.id,
            )
        )
    )
    source_terms = _terms(source.persona)
    for overall, target, values in candidates[:20]:
        target_terms = _terms(target.persona)
        shared = sorted(source_terms & target_terms)[:30]
        differences = sorted(source_terms ^ target_terms)[:30]
        common = dict(
            workspace_id=workspace_id,
            overall_similarity=overall,
            content_similarity=values["content"],
            visual_similarity=values["visual"],
            communication_similarity=values["communication"],
            commercial_similarity=values["commercial"],
            shared_terms=shared,
            differences=differences,
        )
        session.add(
            CreatorRelationship(
                **common,
                creator_id=source.creator_id,
                related_creator_id=target.creator_id,
                persona_version_id=source.id,
                related_persona_version_id=target.id,
            )
        )
        session.add(
            CreatorRelationship(
                **common,
                creator_id=target.creator_id,
                related_creator_id=source.creator_id,
                persona_version_id=target.id,
                related_persona_version_id=source.id,
            )
        )
    await session.flush()
    return min(len(candidates), 20)
