from uuid import uuid4

import pytest
from conftest import observation
from pydantic import ValidationError

from brandfit_core.domain.analysis import EvidenceReference


def test_post_observation_rejects_cross_post_evidence() -> None:
    value = observation().model_dump()
    value["evidence"] = [
        EvidenceReference(post_id=uuid4(), kind="caption", locator="caption:full").model_dump()
    ]
    with pytest.raises(ValidationError, match="analyzed post"):
        type(observation()).model_validate(value)
