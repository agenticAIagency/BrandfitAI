from __future__ import annotations

import io
from uuid import UUID

import orjson
import pyarrow as pa
import pyarrow.parquet as pq
from sqlalchemy import select

from brandfit_core.db.models import (
    CreatorPersonaVersion,
    CreatorRelationship,
    DatasetExport,
    ProcessingJob,
)
from brandfit_core.db.session import SessionFactory
from brandfit_core.domain.enums import JobStatus
from brandfit_core.storage import ObjectStore


async def generate_export(export_id: UUID) -> dict[str, str]:
    async with SessionFactory() as session:
        export = await session.get(DatasetExport, export_id)
        if not export:
            raise ValueError("Export does not exist")
        job = await session.get(ProcessingJob, export.job_id)
        if job is None:
            raise ValueError("Export job does not exist")
        export.status = JobStatus.RUNNING
        job.status = JobStatus.RUNNING
        await session.commit()
        selection = export.selection or {}
        query = select(CreatorPersonaVersion).where(
            CreatorPersonaVersion.workspace_id == export.workspace_id,
            CreatorPersonaVersion.is_latest.is_(True),
        )
        creator_ids = selection.get("creator_ids")
        if creator_ids:
            query = query.where(
                CreatorPersonaVersion.creator_id.in_([UUID(value) for value in creator_ids])
            )
        personas = list(
            (await session.scalars(query.order_by(CreatorPersonaVersion.creator_id))).all()
        )
        relationships: list[CreatorRelationship] = []
        if selection.get("include_relationships", True) and personas:
            relationships = list(
                (
                    await session.scalars(
                        select(CreatorRelationship)
                        .where(
                            CreatorRelationship.persona_version_id.in_(
                                [item.id for item in personas]
                            )
                        )
                        .order_by(
                            CreatorRelationship.creator_id,
                            CreatorRelationship.overall_similarity.desc(),
                        )
                    )
                ).all()
            )
        relation_map: dict[UUID, list[dict[str, object]]] = {}
        for relation in relationships:
            relation_map.setdefault(relation.persona_version_id, []).append(
                {
                    "related_creator_id": str(relation.related_creator_id),
                    "overall_similarity": relation.overall_similarity,
                    "content_similarity": relation.content_similarity,
                    "visual_similarity": relation.visual_similarity,
                    "communication_similarity": relation.communication_similarity,
                    "commercial_similarity": relation.commercial_similarity,
                    "shared_terms": relation.shared_terms,
                    "differences": relation.differences,
                }
            )
        rows = [
            {
                "schema_version": export.schema_version,
                "creator_id": str(item.creator_id),
                "persona_version_id": str(item.id),
                "version": item.version,
                "status": item.status.value,
                "summary": item.summary,
                "persona": item.persona,
                "verification": item.verification,
                "evidence_cutoff": item.evidence_cutoff.isoformat(),
                "relationships": relation_map.get(item.id, []),
            }
            for item in personas
        ]
        store = ObjectStore()
        keys: dict[str, str] = {}
        prefix = f"exports/{export.workspace_id}/{export.id}/v{export.schema_version}"
        formats = selection.get("formats", ["jsonl", "parquet"])
        if "jsonl" in formats:
            data = b"".join(orjson.dumps(row, option=orjson.OPT_SORT_KEYS) + b"\n" for row in rows)
            key = f"{prefix}/creator-personas.jsonl"
            store.put_bytes(key, data, "application/x-ndjson")
            keys["jsonl"] = key
        if "parquet" in formats:
            parquet_rows = [
                {
                    "schema_version": row["schema_version"],
                    "creator_id": row["creator_id"],
                    "persona_version_id": row["persona_version_id"],
                    "version": row["version"],
                    "status": row["status"],
                    "summary": row["summary"],
                    "persona_json": orjson.dumps(
                        row["persona"], option=orjson.OPT_SORT_KEYS
                    ).decode(),
                    "verification_json": orjson.dumps(
                        row["verification"], option=orjson.OPT_SORT_KEYS
                    ).decode(),
                    "relationships_json": orjson.dumps(
                        row["relationships"], option=orjson.OPT_SORT_KEYS
                    ).decode(),
                    "evidence_cutoff": row["evidence_cutoff"],
                }
                for row in rows
            ]
            table = pa.Table.from_pylist(parquet_rows)
            buffer = io.BytesIO()
            pq.write_table(table, buffer, compression="zstd", version="2.6")
            key = f"{prefix}/creator-personas.parquet"
            store.put_bytes(key, buffer.getvalue(), "application/vnd.apache.parquet")
            keys["parquet"] = key
        export.object_keys = keys
        export.row_counts = {"personas": len(rows), "relationships": len(relationships)}
        export.status = JobStatus.COMPLETED
        job.status = JobStatus.COMPLETED
        job.progress_current = 1
        job.result = {"export_id": str(export.id), "object_keys": keys}
        await session.commit()
        return keys
