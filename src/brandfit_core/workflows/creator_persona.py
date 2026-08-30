from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, TypedDict
from uuid import UUID

from langgraph.graph import END, START, StateGraph
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from brandfit_core.agents.planner import create_analysis_plan
from brandfit_core.agents.synthesizer import PersonaSynthesizer
from brandfit_core.agents.verifier import PersonaVerifier
from brandfit_core.config import Settings, get_settings
from brandfit_core.db.models import (
    AgentStep,
    Creator,
    Post,
    PostObservationRecord,
    WorkflowRun,
)
from brandfit_core.domain.analysis import PostObservation
from brandfit_core.domain.enums import JobStatus
from brandfit_core.domain.persona import (
    CreatorAnalysisPlan,
    PersonaProposal,
    PersonaVerificationReport,
)
from brandfit_core.providers.router import ProviderRouter
from brandfit_core.services.aggregation import aggregate_observations
from brandfit_core.services.post_analysis import analyze_post
from brandfit_core.services.publisher import publish_persona
from brandfit_core.storage import ObjectStore


class CreatorWorkflowState(TypedDict, total=False):
    creator_id: str
    workflow_run_id: str
    plan: dict[str, Any]
    observation_ids: list[str]
    proposal: dict[str, Any]
    verification: dict[str, Any]
    persona_version_id: str
    error: dict[str, Any]


async def _log_step(
    session: AsyncSession,
    *,
    workspace_id: UUID,
    run_id: UUID,
    agent: str,
    number: int,
    kind: str,
    input_value: dict[str, Any],
    output_value: dict[str, Any],
) -> None:
    session.add(
        AgentStep(
            workspace_id=workspace_id,
            workflow_run_id=run_id,
            agent=agent,
            step_number=number,
            kind=kind,
            input=input_value,
            output=output_value,
        )
    )


def build_creator_persona_graph(
    session: AsyncSession,
    *,
    router: ProviderRouter | None = None,
    store: ObjectStore | None = None,
    settings: Settings | None = None,
    checkpointer: Any = None,
) -> Any:
    settings = settings or get_settings()
    router = router or ProviderRouter(settings)
    store = store or ObjectStore(settings)

    async def require_run(state: CreatorWorkflowState) -> WorkflowRun:
        run = await session.get(WorkflowRun, UUID(state["workflow_run_id"]))
        if run is None:
            raise ValueError("Workflow run does not exist")
        return run

    async def require_creator(state: CreatorWorkflowState) -> Creator:
        creator = await session.get(Creator, UUID(state["creator_id"]))
        if creator is None:
            raise ValueError("Creator does not exist")
        return creator

    async def plan_node(state: CreatorWorkflowState) -> dict[str, Any]:
        creator_id = UUID(state["creator_id"])
        run = await require_run(state)
        plan = await create_analysis_plan(session, creator_id, settings=settings)
        await _log_step(
            session,
            workspace_id=run.workspace_id,
            run_id=run.id,
            agent="creator_analysis_planner",
            number=1,
            kind="plan",
            input_value={"creator_id": str(creator_id)},
            output_value=plan.model_dump(mode="json"),
        )
        run.state = {"node": "planned", "plan": plan.model_dump(mode="json")}
        await session.commit()
        return {"plan": plan.model_dump(mode="json")}

    async def analyze_node(state: CreatorWorkflowState) -> dict[str, Any]:
        plan = CreatorAnalysisPlan.model_validate(state["plan"])
        run = await require_run(state)
        observation_ids: list[str] = []
        failures: list[dict[str, str]] = []
        for item in plan.selected_posts:
            try:
                async with session.begin_nested():
                    observation = await analyze_post(
                        session,
                        item.post_id,
                        router=router,
                        store=store,
                        settings=settings,
                        force_deep_inspection=item.action == "deep_inspection",
                    )
                    observation_ids.append(str(observation.id))
            except Exception as exc:
                failures.append(
                    {"post_id": str(item.post_id), "type": type(exc).__name__, "message": str(exc)}
                )
        await _log_step(
            session,
            workspace_id=run.workspace_id,
            run_id=run.id,
            agent="post_analyst",
            number=2,
            kind="analysis_batch",
            input_value={"selected_posts": len(plan.selected_posts)},
            output_value={"observation_ids": observation_ids, "failures": failures},
        )
        run.state = {"node": "analyzed", "failures": failures}
        await session.commit()
        return {
            "observation_ids": observation_ids,
            "error": {"post_failures": failures} if failures else {},
        }

    async def _aggregate(state: CreatorWorkflowState) -> tuple[Any, list[PostObservation], Creator]:
        creator = await require_creator(state)
        records = (
            list(
                (
                    await session.scalars(
                        select(PostObservationRecord)
                        .join(Post, Post.id == PostObservationRecord.post_id)
                        .where(
                            PostObservationRecord.id.in_(
                                [UUID(value) for value in state.get("observation_ids", [])]
                            ),
                            Post.creator_id == creator.id,
                        )
                    )
                ).all()
            )
            if state.get("observation_ids")
            else []
        )
        observations = [PostObservation.model_validate(item.observation) for item in records]
        posts = list(
            (
                await session.scalars(
                    select(Post).where(
                        Post.creator_id == creator.id, Post.is_quarantined.is_(False)
                    )
                )
            ).all()
        )
        published = {post.id: post.published_at for post in posts}
        return (
            aggregate_observations(
                observations, available_post_count=len(posts), published_at=published
            ),
            observations,
            creator,
        )

    async def synthesize_node(state: CreatorWorkflowState) -> dict[str, Any]:
        aggregate, _observations, creator = await _aggregate(state)
        proposal, attempts = await PersonaSynthesizer(router).synthesize(
            creator_id=str(creator.id), username=creator.current_username, aggregate=aggregate
        )
        run = await require_run(state)
        await _log_step(
            session,
            workspace_id=run.workspace_id,
            run_id=run.id,
            agent="persona_synthesizer",
            number=3,
            kind="synthesis",
            input_value={"observation_ids": state.get("observation_ids", [])},
            output_value={"proposal": proposal.model_dump(mode="json"), "attempts": attempts},
        )
        run.state = {"node": "synthesized"}
        await session.commit()
        return {"proposal": proposal.model_dump(mode="json")}

    async def verify_node(state: CreatorWorkflowState) -> dict[str, Any]:
        _aggregate_value, observations, _creator = await _aggregate(state)
        proposal = PersonaProposal.model_validate(state["proposal"])
        report = PersonaVerifier().verify(proposal, observations)
        run = await require_run(state)
        await _log_step(
            session,
            workspace_id=run.workspace_id,
            run_id=run.id,
            agent="persona_verifier",
            number=4,
            kind="verification",
            input_value={"claim_count": len(proposal.claims)},
            output_value=report.model_dump(mode="json"),
        )
        if not report.accepted:
            run.status = JobStatus.REVIEW_REQUIRED
        run.state = {"node": "verified", "accepted": report.accepted}
        await session.commit()
        return {"verification": report.model_dump(mode="json")}

    async def publish_node(state: CreatorWorkflowState) -> dict[str, Any]:
        proposal = PersonaProposal.model_validate(state["proposal"])
        report = PersonaVerificationReport.model_validate(state["verification"])
        record = await publish_persona(
            session,
            proposal,
            report,
            evidence_cutoff=datetime.now(UTC),
            settings=settings,
        )
        run = await require_run(state)
        await _log_step(
            session,
            workspace_id=run.workspace_id,
            run_id=run.id,
            agent="deterministic_publisher",
            number=5,
            kind="publication",
            input_value={"verified": True},
            output_value={
                "persona_version_id": str(record.id),
                "version": record.version,
                "requires_review": record.requires_review,
            },
        )
        run.status = JobStatus.REVIEW_REQUIRED if record.requires_review else JobStatus.COMPLETED
        run.state = {"node": "published", "persona_version_id": str(record.id)}
        await session.commit()
        return {"persona_version_id": str(record.id)}

    def after_verify(state: CreatorWorkflowState) -> str:
        report = PersonaVerificationReport.model_validate(state["verification"])
        return "publish" if report.accepted else "end"

    builder = StateGraph(CreatorWorkflowState)
    builder.add_node("plan", plan_node)
    builder.add_node("analyze", analyze_node)
    builder.add_node("synthesize", synthesize_node)
    builder.add_node("verify", verify_node)
    builder.add_node("publish", publish_node)
    builder.add_edge(START, "plan")
    builder.add_edge("plan", "analyze")
    builder.add_edge("analyze", "synthesize")
    builder.add_edge("synthesize", "verify")
    builder.add_conditional_edges("verify", after_verify, {"publish": "publish", "end": END})
    builder.add_edge("publish", END)
    return builder.compile(checkpointer=checkpointer)
