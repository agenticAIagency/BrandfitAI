from __future__ import annotations

import hashlib
import json
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from brandfit_core.agents.react import BoundedReActInvestigator
from brandfit_core.config import Settings, get_settings
from brandfit_core.db.models import (
    AnalysisRun,
    CreatorProfileSnapshot,
    ExtractionArtifact,
    MediaAsset,
    Post,
    PostFacetEmbedding,
    PostMetricSnapshot,
    PostObservationRecord,
    PostTaxonomyAssignment,
    TaxonomyTerm,
)
from brandfit_core.domain.analysis import EvidenceReference, PostAnalysisDraft, PostObservation
from brandfit_core.domain.enums import ArtifactType, JobStatus
from brandfit_core.providers.base import MediaPart, ProviderFailure, ProviderPayload
from brandfit_core.providers.embedding import configured_embedder
from brandfit_core.providers.router import ProviderRouter
from brandfit_core.services.extraction import extract_post_evidence, get_artifacts
from brandfit_core.storage import ObjectStore
from brandfit_core.taxonomy import TAXONOMY_VERSION, resolve_term

PROMPT_VERSION = "post-analysis-1.0"
SCHEMA_VERSION = "1.0"


def _fingerprint(
    post: Post,
    assets: list[MediaAsset],
    settings: Settings,
    *,
    metrics: dict[str, object] | None,
    profile: dict[str, object] | None,
    analysis_mode: str = "baseline",
) -> str:
    metadata = {
        # Evidence references inside PostObservation are post-specific. A
        # reusable cross-post vision cache would need a separate unbound
        # artifact contract; never return another post's observation directly.
        "post_id": str(post.id),
        "caption": post.caption,
        "published_at": post.published_at.isoformat(),
        "media_type": post.media_type,
        "checksums": sorted(asset.sha256 for asset in assets),
        "metrics": metrics or {},
        "profile": profile or {},
        "provider_chain": settings.provider_chain,
        "prompt": PROMPT_VERSION,
        "schema": SCHEMA_VERSION,
        "taxonomy": TAXONOMY_VERSION,
        "analysis_mode": analysis_mode,
    }
    return hashlib.sha256(json.dumps(metadata, sort_keys=True, default=str).encode()).hexdigest()


def _facet_text(observation: PostObservation, facet: str) -> str:
    value = getattr(observation, facet)
    return json.dumps(value.model_dump(mode="json"), sort_keys=True, ensure_ascii=False)


def _available_references(
    post: Post,
    artifacts: list[ExtractionArtifact],
    metrics: PostMetricSnapshot | None,
) -> list[EvidenceReference]:
    values: list[EvidenceReference] = []
    if post.caption:
        values.append(
            EvidenceReference(
                post_id=post.id,
                kind="caption",
                locator="caption:full",
                excerpt=post.caption[:2000],
            )
        )
    values.append(EvidenceReference(post_id=post.id, kind="profile", locator="profile:latest"))
    if metrics:
        values.append(EvidenceReference(post_id=post.id, kind="metrics", locator="metrics:latest"))
    mapping = {
        ArtifactType.KEYFRAME: "keyframe",
        ArtifactType.TRANSCRIPT: "transcript",
        ArtifactType.OCR: "ocr",
    }
    for artifact in artifacts:
        kind = mapping.get(artifact.artifact_type)
        if kind:
            excerpt = str((artifact.content or {}).get("text", ""))[:2000] or None
            values.append(
                EvidenceReference(
                    post_id=post.id,
                    artifact_id=artifact.id,
                    kind=kind,
                    locator=artifact.locator,
                    excerpt=excerpt,
                )
            )
    return values


def _validate_generated_evidence(
    draft: PostAnalysisDraft, allowed: list[EvidenceReference]
) -> None:
    allowed_keys = {(item.artifact_id, item.kind, item.locator) for item in allowed}
    references = list(draft.evidence)
    for signal in draft.safety:
        references.extend(signal.evidence)
    invalid = [
        item
        for item in references
        if (item.artifact_id, item.kind, item.locator) not in allowed_keys
    ]
    if invalid:
        raise ProviderFailure(
            "Provider invented or altered evidence references",
            retryable=False,
            code="invalid_evidence_reference",
        )


async def analyze_post(
    session: AsyncSession,
    post_id: UUID,
    *,
    router: ProviderRouter | None = None,
    store: ObjectStore | None = None,
    settings: Settings | None = None,
    force_deep_inspection: bool = False,
) -> PostObservationRecord:
    settings = settings or get_settings()
    router = router or ProviderRouter(settings)
    store = store or ObjectStore(settings)
    post = await session.scalar(select(Post).where(Post.id == post_id))
    if post is None or post.is_quarantined:
        raise ValueError("Post does not exist or is quarantined")
    assets = list(
        (await session.scalars(select(MediaAsset).where(MediaAsset.post_id == post.id))).all()
    )
    metrics = await session.scalar(
        select(PostMetricSnapshot)
        .where(PostMetricSnapshot.post_id == post.id)
        .order_by(PostMetricSnapshot.captured_at.desc())
        .limit(1)
    )
    profile = await session.scalar(
        select(CreatorProfileSnapshot)
        .where(CreatorProfileSnapshot.creator_id == post.creator_id)
        .order_by(CreatorProfileSnapshot.captured_at.desc())
        .limit(1)
    )
    fingerprint = _fingerprint(
        post,
        assets,
        settings,
        metrics=metrics.metrics if metrics else None,
        profile=profile.profile if profile else None,
        analysis_mode="deep" if force_deep_inspection else "baseline",
    )
    cached = await session.scalar(
        select(PostObservationRecord)
        .join(AnalysisRun, AnalysisRun.id == PostObservationRecord.analysis_run_id)
        .where(
            PostObservationRecord.post_id == post.id,
            AnalysisRun.fingerprint == fingerprint,
            AnalysisRun.status == JobStatus.COMPLETED,
        )
    )
    if cached:
        return cached

    run = AnalysisRun(
        workspace_id=post.workspace_id,
        post_id=post.id,
        fingerprint=fingerprint,
        status=JobStatus.RUNNING,
        prompt_version=PROMPT_VERSION,
        schema_version=SCHEMA_VERSION,
        taxonomy_version=TAXONOMY_VERSION,
    )
    session.add(run)
    await session.flush()
    try:
        artifacts = await get_artifacts(session, post.id)
        if not artifacts:
            artifacts = await extract_post_evidence(session, post, store=store, settings=settings)
        references = _available_references(post, artifacts, metrics)
        has_frames = any(item.artifact_type == ArtifactType.KEYFRAME for item in artifacts)
        has_transcript = any(
            item.artifact_type == ArtifactType.TRANSCRIPT and (item.content or {}).get("text")
            for item in artifacts
        )
        has_ocr = any(
            item.artifact_type == ArtifactType.OCR and (item.content or {}).get("text")
            for item in artifacts
        )
        coverage_parts = [bool(post.caption), has_frames, has_ocr]
        if post.media_type.value in {"reel", "video"}:
            coverage_parts.append(has_transcript)
        coverage = sum(coverage_parts) / len(coverage_parts)
        context: dict[str, Any] = {
            "post_id": str(post.id),
            "caption": post.caption,
            "published_at": post.published_at.isoformat(),
            "media_type": post.media_type,
            "profile": profile.profile if profile else {},
            "metrics": metrics.metrics if metrics else {},
            "extraction_coverage": coverage,
            "available_evidence": [item.model_dump(mode="json") for item in references],
            "transcript": [
                item.content for item in artifacts if item.artifact_type == ArtifactType.TRANSCRIPT
            ],
            "ocr": [item.content for item in artifacts if item.artifact_type == ArtifactType.OCR],
        }
        media: list[MediaPart] = []
        for artifact in artifacts:
            if artifact.artifact_type == ArtifactType.KEYFRAME and artifact.object_key:
                media.append(
                    MediaPart(
                        mime_type=str(
                            (artifact.content or {}).get("mime_type", "image/jpeg")
                        ),
                        data=await __import__("asyncio").to_thread(
                            store.get_bytes, artifact.object_key
                        ),
                        locator=artifact.locator,
                    )
                )
        policy = (
            "Describe only observable creator content. Do not infer demographics, personality, "
            "mental state, health, religion, politics, sexuality, ethnicity, caste, or other "
            "protected traits. "
            "Every material output must cite an available evidence reference exactly."
        )
        response = await router.generate(
            PostAnalysisDraft,
            ProviderPayload(
                prompt=f"TASK=post_analysis\n{policy}\nINPUT_JSON="
                + json.dumps(context, default=str),
                media=media[:12],
            ),
        )
        attempts = [item.model_dump(mode="json") for item in response.attempts]
        draft = response.value
        if (
            coverage < settings.analysis_confidence_threshold
            or draft.confidence.overall < settings.analysis_confidence_threshold
        ):
            missing: list[str] = []
            if not has_frames:
                missing.append("inspect_frame")
            if not has_transcript:
                missing.append("read_transcript")
            if not has_ocr:
                missing.append("read_ocr")
            missing.append("compare_evidence")
            investigation, react_attempts = await BoundedReActInvestigator(
                router, settings.max_react_tool_calls
            ).run(artifacts, missing)
            attempts.extend(react_attempts)
            context["react_investigation"] = [item.output for item in investigation]
            second = await router.generate(
                PostAnalysisDraft,
                ProviderPayload(
                    prompt=f"TASK=post_analysis\n{policy}\nINPUT_JSON="
                    + json.dumps(context, default=str),
                    media=media[:12],
                ),
            )
            attempts.extend(item.model_dump(mode="json") for item in second.attempts)
            draft = second.value
            response = second
            if (
                draft.confidence.overall < settings.analysis_confidence_threshold
                and len(settings.provider_targets) > 1
            ):
                escalated = await router.generate(
                    PostAnalysisDraft,
                    ProviderPayload(
                        prompt=f"TASK=post_analysis\n{policy}\nINPUT_JSON="
                        + json.dumps(context, default=str),
                        media=media[:12],
                    ),
                    start_target_index=1,
                )
                attempts.extend(item.model_dump(mode="json") for item in escalated.attempts)
                draft = escalated.value
                response = escalated
        _validate_generated_evidence(draft, references)
        observation = PostObservation(
            **draft.model_dump(),
            post_id=post.id,
            analysis_run_id=run.id,
            prompt_version=PROMPT_VERSION,
            taxonomy_version=TAXONOMY_VERSION,
            provider=response.provider,
            model=response.model,
            requires_human_review=(
                draft.confidence.overall < settings.analysis_confidence_threshold
            ),
        )
        record = PostObservationRecord(
            workspace_id=post.workspace_id,
            post_id=post.id,
            analysis_run_id=run.id,
            observation=observation.model_dump(mode="json"),
            confidence=observation.confidence.overall,
        )
        session.add(record)
        await session.flush()
        embedder = configured_embedder(settings)
        facets = ["content", "visual", "communication", "commercial"]
        vectors = embedder.embed([_facet_text(observation, facet) for facet in facets])
        for facet, vector in zip(facets, vectors, strict=True):
            session.add(
                PostFacetEmbedding(
                    workspace_id=post.workspace_id,
                    post_id=post.id,
                    observation_id=record.id,
                    facet=facet,
                    model_version=embedder.model_version,
                    embedding=vector,
                )
            )
        labels = (
            observation.content.topics
            + observation.content.formats
            + observation.content.intents
            + observation.communication.tones
            + observation.communication.speaking_styles
            + observation.visual.composition_styles
            + observation.visual.production_styles
            + observation.commercial.integration_styles
        )
        for label in labels:
            resolved = resolve_term(label.key)
            if not resolved:
                continue
            term = await session.scalar(
                select(TaxonomyTerm).where(
                    TaxonomyTerm.version == TAXONOMY_VERSION, TaxonomyTerm.key == resolved.key
                )
            )
            if term:
                session.add(
                    PostTaxonomyAssignment(
                        workspace_id=post.workspace_id,
                        observation_id=record.id,
                        term_id=term.id,
                        confidence=label.confidence,
                        evidence=[item.model_dump(mode="json") for item in label.evidence],
                    )
                )
        run.status = JobStatus.COMPLETED
        run.provider = response.provider
        run.model = response.model
        run.attempts = attempts
        await session.flush()
        return record
    except Exception as exc:
        run.status = JobStatus.FAILED
        run.error = {"type": type(exc).__name__, "message": str(exc)}
        await session.flush()
        raise
