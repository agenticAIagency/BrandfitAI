from __future__ import annotations

import asyncio
import json
from pathlib import Path

from sqlalchemy import select

from brandfit_core.agents.verifier import PersonaVerifier
from brandfit_core.db.models import CreatorPersonaVersion, Post, PostObservationRecord
from brandfit_core.db.session import SessionFactory
from brandfit_core.domain.analysis import PostObservation
from brandfit_core.domain.persona import PersonaProposal


async def run() -> dict[str, object]:
    benchmark_path = Path("fixtures/generated/benchmark_v1.json")
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    expected_posts = {item["platform_post_id"]: item for item in benchmark["posts"]}
    results: dict[str, object] = {}
    async with SessionFactory() as session:
        posts = list((await session.scalars(select(Post))).all())
        observations = list((await session.scalars(select(PostObservationRecord))).all())
        latest_by_post: dict[object, PostObservationRecord] = {}
        for record in observations:
            current = latest_by_post.get(record.post_id)
            if not current or record.created_at > current.created_at:
                latest_by_post[record.post_id] = record
        matches = 0
        reviewed = 0
        schema_valid_observations = 0
        for post in posts:
            expected = expected_posts.get(post.platform_post_id)
            record = latest_by_post.get(post.id)
            if not record:
                continue
            value = PostObservation.model_validate(record.observation)
            schema_valid_observations += 1
            if expected:
                reviewed += 1
                topics = {item.key for item in value.content.topics}
                if expected["expected_primary_topic"] in topics:
                    matches += 1
        personas = list((await session.scalars(select(CreatorPersonaVersion))).all())
        valid_personas = 0
        supported_personas = 0
        prohibited_personas = 0
        for record in personas:
            proposal = PersonaProposal.model_validate(record.persona)
            valid_personas += 1
            values = [
                PostObservation.model_validate(item.observation)
                for item in observations
                if item.post_id in proposal.evidence_post_ids
            ]
            verification = PersonaVerifier().verify(proposal, values)
            if not verification.unsupported_claims:
                supported_personas += 1
            if verification.policy_violations:
                prohibited_personas += 1
        agreement = matches / reviewed if reviewed else 0.0
        results = {
            "reviewed_posts": reviewed,
            "topic_agreement": agreement,
            "schema_valid_observations": schema_valid_observations,
            "schema_valid_personas": valid_personas,
            "supported_personas": supported_personas,
            "prohibited_personas": prohibited_personas,
            "gates": {
                "at_least_50_reviewed_posts": reviewed >= 50,
                "at_least_90_percent_topic_agreement": agreement >= 0.9,
                "all_personas_schema_valid": valid_personas == len(personas),
                "all_claims_resolve": supported_personas == len(personas),
                "zero_prohibited_inferences": prohibited_personas == 0,
            },
        }
    return results


def main() -> None:
    result = asyncio.run(run())
    print(json.dumps(result, indent=2))
    if not all(result["gates"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
