from __future__ import annotations

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from brandfit_core.db.models import ProcessingJob
from brandfit_core.domain.jobs import JobRead


async def get_job(session: AsyncSession, workspace_id: UUID, job_id: UUID) -> JobRead | None:
    record = await session.scalar(
        select(ProcessingJob).where(
            ProcessingJob.id == job_id, ProcessingJob.workspace_id == workspace_id
        )
    )
    if record is None:
        return None
    return JobRead.model_validate(record, from_attributes=True)
