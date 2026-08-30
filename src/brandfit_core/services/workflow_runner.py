from __future__ import annotations

from uuid import UUID

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from sqlalchemy import select

from brandfit_core.config import get_settings
from brandfit_core.db.models import (
    Creator,
    IngestionBatch,
    Post,
    ProcessingJob,
    WorkflowRun,
)
from brandfit_core.db.session import SessionFactory
from brandfit_core.domain.enums import JobStatus, WorkflowKind
from brandfit_core.workflows.creator_persona import build_creator_persona_graph


def _checkpoint_url(url: str) -> str:
    return url.replace("postgresql+psycopg://", "postgresql://")


async def run_creator_job(job_id: UUID, creator_id: UUID) -> dict[str, object]:
    settings = get_settings()
    async with SessionFactory() as session:
        job = await session.get(ProcessingJob, job_id)
        creator = await session.get(Creator, creator_id)
        if not job or not creator:
            raise ValueError("Job or creator does not exist")
        job.status = JobStatus.RUNNING
        workflow = WorkflowRun(
            workspace_id=job.workspace_id,
            kind=WorkflowKind.CREATOR_PERSONA,
            subject_id=creator.id,
            status=JobStatus.RUNNING,
            state={"node": "starting"},
        )
        session.add(workflow)
        await session.commit()
        try:
            async with AsyncPostgresSaver.from_conn_string(
                _checkpoint_url(settings.database_url_sync)
            ) as checkpointer:
                await checkpointer.setup()
                graph = build_creator_persona_graph(
                    session, settings=settings, checkpointer=checkpointer
                )
                result = await graph.ainvoke(
                    {
                        "creator_id": str(creator.id),
                        "workflow_run_id": str(workflow.id),
                    },
                    config={"configurable": {"thread_id": str(workflow.id)}},
                )
            await session.refresh(workflow)
            await session.refresh(job)
            job.status = workflow.status
            job.progress_current = job.progress_total
            job.result = {
                "creator_id": str(creator.id),
                "workflow_run_id": str(workflow.id),
                "persona_version_id": result.get("persona_version_id"),
            }
            await session.commit()
            return job.result
        except Exception as exc:
            await session.rollback()
            failed_workflow = await session.get(WorkflowRun, workflow.id)
            failed_job = await session.get(ProcessingJob, job.id)
            error = {"type": type(exc).__name__, "message": str(exc)}
            if failed_workflow:
                failed_workflow.status = JobStatus.FAILED
                failed_workflow.state = {"node": "failed", "error": error}
            if failed_job:
                failed_job.status = JobStatus.FAILED
                failed_job.error = error
            await session.commit()
            raise


async def run_ingestion_job(batch_id: UUID) -> dict[str, object]:
    async with SessionFactory() as session:
        batch = await session.get(IngestionBatch, batch_id)
        if not batch or not batch.job_id:
            raise ValueError("Ingestion batch does not exist")
        creator_id = await session.scalar(
            select(Post.creator_id).where(Post.ingestion_batch_id == batch.id).limit(1)
        )
        if creator_id is None:
            job = await session.get(ProcessingJob, batch.job_id)
            if job is None:
                raise ValueError("Ingestion job does not exist")
            job.status = JobStatus.COMPLETED
            job.result = {"accepted_posts": 0}
            await session.commit()
            return job.result
        job_id = batch.job_id
    return await run_creator_job(job_id, creator_id)
