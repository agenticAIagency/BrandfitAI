from __future__ import annotations

from datetime import timedelta
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Body, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from brandfit_core.api.dependencies import workspace_id
from brandfit_core.db.models import ProcessingJob
from brandfit_core.db.session import get_session
from brandfit_core.domain.ingestion import IngestionBatchRead
from brandfit_core.services.ingestion import ManifestEnvelopeError, get_batch, ingest_manifest
from brandfit_core.storage import ObjectStore

router = APIRouter(prefix="/v1/ingestion", tags=["ingestion"])


class UploadRequest(BaseModel):
    object_key: str = Field(min_length=1, max_length=1024)
    expires_seconds: int = Field(3600, ge=60, le=86_400)


class UploadResponse(BaseModel):
    object_key: str
    upload_url: str


@router.post("/uploads", response_model=UploadResponse)
async def create_upload(request: UploadRequest) -> UploadResponse:
    store = ObjectStore()
    url = await __import__("asyncio").to_thread(
        store.presigned_upload, request.object_key, timedelta(seconds=request.expires_seconds)
    )
    return UploadResponse(object_key=request.object_key, upload_url=url)


@router.post("/manifests", response_model=IngestionBatchRead, status_code=status.HTTP_202_ACCEPTED)
async def submit_manifest(
    payload: dict[str, Any] = Body(...),
    session: AsyncSession = Depends(get_session),
) -> IngestionBatchRead:
    try:
        batch, created = await ingest_manifest(session, payload)
        await session.commit()
    except ManifestEnvelopeError as exc:
        await session.rollback()
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if created and batch.accepted_posts:
        from brandfit_core.workers.tasks import process_ingestion_batch

        task = process_ingestion_batch.delay(str(batch.id))
        job = await session.scalar(select(ProcessingJob).where(ProcessingJob.id == batch.job_id))
        if job:
            job.celery_task_id = task.id
            await session.commit()
    return batch


@router.get("/batches/{batch_id}", response_model=IngestionBatchRead)
async def read_batch(
    batch_id: UUID,
    workspace: UUID = Depends(workspace_id),
    session: AsyncSession = Depends(get_session),
) -> IngestionBatchRead:
    batch = await get_batch(session, workspace, batch_id)
    if not batch:
        raise HTTPException(status_code=404, detail="Ingestion batch not found")
    return batch
