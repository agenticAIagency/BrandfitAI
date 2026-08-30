import pytest

from brandfit_core.agents.react import BoundedReActInvestigator
from brandfit_core.config import Settings
from brandfit_core.providers.fake import FakeProvider
from brandfit_core.providers.router import ProviderRouter


@pytest.mark.asyncio
async def test_react_enforces_six_tool_call_budget() -> None:
    fake = FakeProvider()
    for _ in range(10):
        fake.queue({"thought": "inspect", "action": "inspect_frame", "arguments": {}})
    router = ProviderRouter(
        Settings(provider_chain="fake:fixture"),
        providers={"fake": fake},
        attempts_per_provider=1,
    )
    results, _attempts = await BoundedReActInvestigator(router, max_calls=6).run([], [])
    assert len(results) == 6
