from __future__ import annotations

from collections import defaultdict
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from brandfit_core.config import Settings, get_settings
from brandfit_core.db.models import Post, PostObservationRecord
from brandfit_core.domain.persona import CreatorAnalysisPlan, PlannedPost


def _sample_posts(posts: list[Post], observations: dict[UUID, PostObservationRecord]) -> list[Post]:
    if len(posts) <= 50:
        return posts
    selected: dict[UUID, Post] = {post.id: post for post in posts[:20]}
    selected.update({post.id: post for post in posts if post.is_pinned})
    strata: dict[str, list[Post]] = defaultdict(list)
    for post in posts[20:]:
        observation = observations.get(post.id)
        topic = "unknown"
        if observation:
            topics = observation.observation.get("content", {}).get("topics", [])
            if topics:
                topic = topics[0].get("key", "unknown")
        quarter = f"{post.published_at.year}-q{(post.published_at.month - 1) // 3 + 1}"
        strata[f"{post.media_type}:{quarter}:{topic}"].append(post)
    while len(selected) < 50 and any(strata.values()):
        for key in sorted(strata):
            if strata[key] and len(selected) < 50:
                post = strata[key].pop(len(strata[key]) // 2)
                selected[post.id] = post
    return sorted(selected.values(), key=lambda item: item.published_at, reverse=True)[:50]


async def create_analysis_plan(
    session: AsyncSession,
    creator_id: UUID,
    *,
    settings: Settings | None = None,
) -> CreatorAnalysisPlan:
    settings = settings or get_settings()
    posts = list(
        (
            await session.scalars(
                select(Post)
                .where(Post.creator_id == creator_id, Post.is_quarantined.is_(False))
                .order_by(Post.published_at.desc())
            )
        ).all()
    )
    records = (
        list(
            (
                await session.scalars(
                    select(PostObservationRecord).where(
                        PostObservationRecord.post_id.in_([post.id for post in posts])
                    )
                )
            ).all()
        )
        if posts
        else []
    )
    observations = {record.post_id: record for record in records}
    selected = _sample_posts(posts, observations)
    planned: list[PlannedPost] = []
    missing: list[str] = []
    for post in selected:
        observation = observations.get(post.id)
        if observation is None:
            action = "baseline"
            reason = "No approved post observation exists."
        elif observation.confidence < settings.analysis_confidence_threshold:
            action = "deep_inspection"
            reason = "Existing extraction confidence is below the configured threshold."
        else:
            action = "reuse"
            reason = "An unchanged, sufficiently confident observation is available."
        planned.append(PlannedPost(post_id=post.id, action=action, reason=reason))
    if not posts:
        missing.append("No analyzable posts are stored for this creator.")
    elif len(posts) < 5:
        missing.append("Fewer than five analyzable posts; persona status will be insufficient.")
    return CreatorAnalysisPlan(
        creator_id=creator_id,
        selected_posts=planned,
        missing_evidence=missing,
        requires_review=len(posts) < 10,
        max_react_tool_calls=settings.max_react_tool_calls,
    )
