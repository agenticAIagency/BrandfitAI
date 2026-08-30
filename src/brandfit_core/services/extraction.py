from __future__ import annotations

import asyncio
import hashlib
from uuid import UUID

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from brandfit_core.config import Settings, get_settings
from brandfit_core.db.models import ExtractionArtifact, MediaAsset, Post
from brandfit_core.domain.enums import ArtifactType
from brandfit_core.media.extractor import extract_media
from brandfit_core.media.ocr import MultilingualOCR
from brandfit_core.media.transcription import FasterWhisperTranscriber
from brandfit_core.storage import ObjectStore


async def extract_post_evidence(
    session: AsyncSession,
    post: Post,
    *,
    store: ObjectStore | None = None,
    settings: Settings | None = None,
) -> list[ExtractionArtifact]:
    settings = settings or get_settings()
    store = store or ObjectStore(settings)
    assets = list(
        (
            await session.scalars(
                select(MediaAsset)
                .where(MediaAsset.post_id == post.id)
                .order_by(MediaAsset.position)
            )
        ).all()
    )
    # Derived extraction is reproducible; replace only derived artifacts, never source evidence.
    await session.execute(delete(ExtractionArtifact).where(ExtractionArtifact.post_id == post.id))
    records: list[ExtractionArtifact] = []
    ocr = MultilingualOCR()
    transcriber = FasterWhisperTranscriber(settings.transcription_model)
    for asset in assets:
        prefix = f"derived/{post.workspace_id}/{post.id}/{asset.sha256[:16]}"
        extracted = await asyncio.to_thread(
            extract_media,
            store,
            object_key=asset.object_key,
            media_type=post.media_type,
            mime_type=asset.mime_type,
            output_prefix=prefix,
        )
        for item in extracted:
            artifact_type = ArtifactType(item.kind)
            record = ExtractionArtifact(
                workspace_id=post.workspace_id,
                post_id=post.id,
                artifact_type=artifact_type,
                object_key=str(item.path).replace("\\", "/"),
                locator=item.locator,
                content={"mime_type": item.mime_type},
            )
            session.add(record)
            records.append(record)
            if artifact_type == ArtifactType.KEYFRAME:
                if record.object_key is None:
                    raise ValueError("Keyframe extraction did not produce an object key")
                if settings.enable_ocr:
                    try:
                        content = await asyncio.to_thread(ocr.extract, store, record.object_key)
                    except Exception as exc:
                        content = {"text": "", "lines": [], "unavailable": str(exc)}
                else:
                    content = {"text": "", "lines": [], "unavailable": "OCR disabled"}
                ocr_record = ExtractionArtifact(
                    workspace_id=post.workspace_id,
                    post_id=post.id,
                    artifact_type=ArtifactType.OCR,
                    content=content,
                    locator=f"ocr:{item.locator}",
                    checksum=hashlib.sha256(str(content).encode()).hexdigest(),
                )
                session.add(ocr_record)
                records.append(ocr_record)
            elif artifact_type == ArtifactType.AUDIO:
                if record.object_key is None:
                    raise ValueError("Audio extraction did not produce an object key")
                if settings.enable_transcription:
                    try:
                        content = await asyncio.to_thread(
                            transcriber.transcribe, store, record.object_key
                        )
                    except Exception as exc:
                        content = {"text": "", "segments": [], "unavailable": str(exc)}
                else:
                    content = {
                        "text": "",
                        "segments": [],
                        "unavailable": "Transcription disabled",
                    }
                transcript = ExtractionArtifact(
                    workspace_id=post.workspace_id,
                    post_id=post.id,
                    artifact_type=ArtifactType.TRANSCRIPT,
                    content=content,
                    locator="transcript:all",
                    checksum=hashlib.sha256(str(content).encode()).hexdigest(),
                )
                session.add(transcript)
                records.append(transcript)
    await session.flush()
    return records


async def get_artifacts(session: AsyncSession, post_id: UUID) -> list[ExtractionArtifact]:
    return list(
        (
            await session.scalars(
                select(ExtractionArtifact)
                .where(ExtractionArtifact.post_id == post_id)
                .order_by(ExtractionArtifact.created_at)
            )
        ).all()
    )
