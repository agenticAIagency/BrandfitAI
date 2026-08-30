from __future__ import annotations

import json
from typing import Any, cast

from brandfit_core.domain.persona import PersonaNarrativeDraft, PersonaProposal
from brandfit_core.providers.base import ProviderFailure, ProviderPayload
from brandfit_core.providers.router import ProviderRouter
from brandfit_core.services.aggregation import AggregateResult


class PersonaSynthesizer:
    def __init__(self, router: ProviderRouter) -> None:
        self.router = router

    async def synthesize(
        self, *, creator_id: str, username: str, aggregate: AggregateResult
    ) -> tuple[PersonaProposal, list[dict[str, object]]]:
        context = {
            "creator_id": creator_id,
            "username": username,
            "post_count": len(aggregate.evidence_post_ids),
            "content": aggregate.content,
            "communication": aggregate.communication,
            "visual": aggregate.visual,
            "commercial": aggregate.commercial,
            "performance": aggregate.performance,
            "recent_90d": aggregate.recent_90d,
            "trends": aggregate.trends,
            "evidence_examples": aggregate.evidence_examples,
        }
        response = await self.router.generate(
            PersonaNarrativeDraft,
            ProviderPayload(
                prompt=(
                    "TASK=persona_narrative\nWrite only observable, evidence-grounded claims. "
                    "Never infer demographics, protected traits, psychology, personality, "
                    "health, beliefs, or private "
                    "facts. Preserve evidence references exactly.\nINPUT_JSON="
                    + json.dumps(context, default=str)
                )
            ),
        )
        allowed = {
            (
                str(reference["post_id"]),
                reference.get("artifact_id"),
                reference["kind"],
                reference["locator"],
            )
            for item in aggregate.evidence_examples
            for reference in cast(list[dict[str, Any]], item.get("evidence", []))
        }
        for claim in response.value.claims:
            if not claim.evidence or not any(
                (
                    str(ref.post_id),
                    str(ref.artifact_id) if ref.artifact_id else None,
                    ref.kind,
                    ref.locator,
                )
                in allowed
                for ref in claim.evidence
            ):
                raise ProviderFailure(
                    "Persona synthesis introduced an unsupported claim",
                    retryable=False,
                    code="unsupported_persona_claim",
                )
        proposal = PersonaProposal(
            creator_id=creator_id,
            status=aggregate.status,
            summary=response.value.summary,
            claims=response.value.claims,
            content_fingerprint=aggregate.content,
            communication_fingerprint=aggregate.communication,
            visual_fingerprint=aggregate.visual,
            commercial_fingerprint=aggregate.commercial,
            performance_context=aggregate.performance,
            safety_observations=aggregate.safety,
            recent_90d=aggregate.recent_90d,
            trends=aggregate.trends,
            confidence=aggregate.confidence,
            evidence_post_ids=aggregate.evidence_post_ids,
        )
        return proposal, [item.model_dump(mode="json") for item in response.attempts]
