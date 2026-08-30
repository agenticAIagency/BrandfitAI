from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from brandfit_core.api.dependencies import workspace_id
from brandfit_core.db.session import get_session
from brandfit_core.domain.jobs import JobRead
from brandfit_core.services.jobs import get_job

router = APIRouter(prefix="/v1/jobs", tags=["jobs"])


@router.get("/{job_id}", response_model=JobRead)
async def read_job(
    job_id: UUID,
    workspace: UUID = Depends(workspace_id),
    session: AsyncSession = Depends(get_session),
) -> JobRead:
    job = await get_job(session, workspace, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job
