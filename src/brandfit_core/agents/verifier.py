from __future__ import annotations

from brandfit_core.domain.analysis import PostObservation
from brandfit_core.domain.persona import (
    ClaimVerification,
    PersonaProposal,
    PersonaVerificationReport,
)

PROHIBITED_INFERENCES = {
    "personality",
    "mental health",
    "ethnicity",
    "religion",
    "political affiliation",
    "sexual orientation",
    "caste",
    "medical condition",
    "household income",
}


class PersonaVerifier:
    def verify(
        self, proposal: PersonaProposal, observations: list[PostObservation]
    ) -> PersonaVerificationReport:
        allowed_refs = {
            (ref.post_id, ref.artifact_id, ref.kind, ref.locator)
            for observation in observations
            for ref in observation.evidence
        }
        decisions: list[ClaimVerification] = []
        unsupported: list[int] = []
        for index, claim in enumerate(proposal.claims):
            invalid = [
                ref
                for ref in claim.evidence
                if (ref.post_id, ref.artifact_id, ref.kind, ref.locator) not in allowed_refs
            ]
            accepted = not invalid
            reasons = [] if accepted else ["One or more evidence references do not resolve."]
            decisions.append(
                ClaimVerification(claim_index=index, accepted=accepted, reasons=reasons)
            )
            if not accepted:
                unsupported.append(index)
        combined = " ".join(
            [proposal.summary] + [claim.statement for claim in proposal.claims]
        ).lower()
        violations = sorted(term for term in PROHIBITED_INFERENCES if term in combined)
        unresolved_observations = any(item.requires_human_review for item in observations)
        return PersonaVerificationReport(
            accepted=not unsupported and not violations,
            claims=decisions,
            unsupported_claims=unsupported,
            policy_violations=violations,
            requires_human_review=bool(
                unsupported or violations or proposal.trends or unresolved_observations
            ),
        )
