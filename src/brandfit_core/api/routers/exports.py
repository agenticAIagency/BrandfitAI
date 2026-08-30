from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from brandfit_core.api.dependencies import workspace_id
from brandfit_core.db.models import DatasetExport, ProcessingJob
from brandfit_core.db.session import get_session
from brandfit_core.domain.enums import JobStatus, WorkflowKind
from brandfit_core.domain.jobs import ExportCreate, ExportRead

router = APIRouter(prefix="/v1/exports", tags=["exports"])


def _read(record: DatasetExport) -> ExportRead:
    return ExportRead.model_validate(record, from_attributes=True)


@router.post("", response_model=ExportRead, status_code=status.HTTP_202_ACCEPTED)
async def create_export(
    request: ExportCreate,
    workspace: UUID = Depends(workspace_id),
    session: AsyncSession = Depends(get_session),
) -> ExportRead:
    formats = sorted(set(request.formats))
    if not formats or not set(formats) <= {"jsonl", "parquet"}:
        raise HTTPException(status_code=422, detail="formats must contain jsonl and/or parquet")
    job = ProcessingJob(
        workspace_id=workspace,
        kind=WorkflowKind.EXPORT,
        status=JobStatus.QUEUED,
        progress_total=1,
    )
    session.add(job)
    await session.flush()
    record = DatasetExport(
        workspace_id=workspace,
        job_id=job.id,
        status=JobStatus.QUEUED,
        schema_version="1.0",
        selection={
            "creator_ids": [str(value) for value in request.creator_ids]
            if request.creator_ids
            else None,
            "include_relationships": request.include_relationships,
            "formats": formats,
        },
    )
    session.add(record)
    await session.commit()
    await session.refresh(record)
    from brandfit_core.workers.tasks import generate_dataset_export

    task = generate_dataset_export.delay(str(record.id))
    job.celery_task_id = task.id
    await session.commit()
    return _read(record)


@router.get("/{export_id}", response_model=ExportRead)
async def read_export(
    export_id: UUID,
    workspace: UUID = Depends(workspace_id),
    session: AsyncSession = Depends(get_session),
) -> ExportRead:
    record = await session.scalar(
        select(DatasetExport).where(
            DatasetExport.id == export_id, DatasetExport.workspace_id == workspace
        )
    )
    if not record:
        raise HTTPException(status_code=404, detail="Export not found")
    return _read(record)
