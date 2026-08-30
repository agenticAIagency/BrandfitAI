from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field

from brandfit_core.domain.enums import JobStatus, WorkflowKind


class JobRead(BaseModel):
    id: UUID
    workspace_id: UUID
    kind: WorkflowKind
    status: JobStatus
    progress_current: int
    progress_total: int
    result: dict[str, Any] | None
    error: dict[str, Any] | None
    created_at: datetime
    updated_at: datetime


class ExportCreate(BaseModel):
    creator_ids: list[UUID] | None = None
    include_relationships: bool = True
    formats: list[str] = Field(default_factory=lambda: ["jsonl", "parquet"])


class ExportRead(BaseModel):
    id: UUID
    status: JobStatus
    schema_version: str
    object_keys: dict[str, str]
    row_counts: dict[str, int]
    created_at: datetime
