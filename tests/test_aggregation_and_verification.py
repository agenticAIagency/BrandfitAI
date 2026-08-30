from datetime import UTC, datetime

from conftest import observation

from brandfit_core.agents.verifier import PersonaVerifier
from brandfit_core.domain.enums import PersonaStatus
from brandfit_core.domain.persona import PersonaClaim, PersonaProposal
from brandfit_core.services.aggregation import aggregate_observations


def test_aggregate_is_deterministic_and_uses_status_thresholds() -> None:
    values = [observation() for _ in range(10)]
    first = aggregate_observations(values, available_post_count=10, now=datetime.now(UTC))
    second = aggregate_observations(values, available_post_count=10, now=datetime.now(UTC))
    assert first.status == PersonaStatus.ESTABLISHED
    assert first.content == second.content
    assert first.content["topic.tech"] == 1.0
    assert first.confidence.sample_size == 1.0


def test_verifier_rejects_prohibited_inference() -> None:
    value = observation()
    aggregate = aggregate_observations([value] * 5, available_post_count=5)
    proposal = PersonaProposal(
        creator_id=value.post_id,
        status=aggregate.status,
        summary="The creator's personality is outgoing.",
        claims=[
            PersonaClaim(
                category="content",
                statement="Technology content is observable.",
                confidence=0.9,
                evidence=[value.evidence[0]],
            )
        ],
        content_fingerprint=aggregate.content,
        communication_fingerprint=aggregate.communication,
        visual_fingerprint=aggregate.visual,
        commercial_fingerprint=aggregate.commercial,
        performance_context=aggregate.performance,
        safety_observations=[],
        recent_90d=aggregate.recent_90d,
        trends={},
        confidence=aggregate.confidence,
        evidence_post_ids=[value.post_id],
    )
    result = PersonaVerifier().verify(proposal, [value])
    assert not result.accepted
    assert result.policy_violations == ["personality"]
