from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

from brandfit_core.db.models import ExtractionArtifact
from brandfit_core.domain.analysis import ReActAction
from brandfit_core.domain.enums import ArtifactType
from brandfit_core.providers.base import ProviderPayload
from brandfit_core.providers.router import ProviderRouter
from brandfit_core.taxonomy import TERMS, resolve_term


@dataclass(slots=True)
class ToolResult:
    action: str
    output: dict[str, Any]


class EvidenceToolbox:
    """Read-only, post-scoped tools exposed to the ReAct investigator."""

    def __init__(self, artifacts: list[ExtractionArtifact]) -> None:
        self.artifacts = artifacts

    def execute(self, action: ReActAction) -> ToolResult:
        args = action.arguments
        if action.action == "inspect_frame":
            frames = [
                {
                    "artifact_id": str(item.id),
                    "locator": item.locator,
                    "object_key": item.object_key,
                }
                for item in self.artifacts
                if item.artifact_type == ArtifactType.KEYFRAME
            ]
            return ToolResult(action.action, {"frames": frames[:12]})
        if action.action == "read_transcript":
            transcripts = [
                item.content
                for item in self.artifacts
                if item.artifact_type == ArtifactType.TRANSCRIPT
            ]
            return ToolResult(action.action, {"transcripts": transcripts})
        if action.action == "read_ocr":
            ocr_values = [
                item.content for item in self.artifacts if item.artifact_type == ArtifactType.OCR
            ]
            return ToolResult(action.action, {"ocr": ocr_values})
        if action.action == "resolve_taxonomy":
            key = str(args.get("key", ""))
            term = resolve_term(key)
            output = asdict(term) if term else {"allowed_terms": TERMS}
            return ToolResult(action.action, output)
        if action.action == "compare_evidence":
            transcripts = [
                item.content
                for item in self.artifacts
                if item.artifact_type == ArtifactType.TRANSCRIPT
            ]
            ocr = [
                item.content for item in self.artifacts if item.artifact_type == ArtifactType.OCR
            ]
            return ToolResult(action.action, {"transcript": transcripts, "ocr": ocr})
        return ToolResult("finish", {})


class BoundedReActInvestigator:
    def __init__(self, router: ProviderRouter, max_calls: int = 6) -> None:
        self.router = router
        self.max_calls = max_calls

    async def run(
        self, artifacts: list[ExtractionArtifact], missing_tools: list[str]
    ) -> tuple[list[ToolResult], list[dict[str, Any]]]:
        toolbox = EvidenceToolbox(artifacts)
        results: list[ToolResult] = []
        attempts: list[dict[str, Any]] = []
        for calls_used in range(self.max_calls):
            context = {
                "calls_used": calls_used,
                "calls_remaining": self.max_calls - calls_used,
                "missing_tools": missing_tools,
                "observations": [item.output for item in results],
            }
            response = await self.router.generate(
                ReActAction,
                ProviderPayload(
                    prompt=(
                        "TASK=react_action\nChoose one read-only evidence tool. Never infer "
                        "demographic, psychological, health, religious, political, "
                        "sexual-orientation, or other protected "
                        "traits.\nINPUT_JSON=" + json.dumps(context, default=str)
                    )
                ),
            )
            attempts.extend(item.model_dump(mode="json") for item in response.attempts)
            if response.value.action == "finish":
                break
            result = toolbox.execute(response.value)
            results.append(result)
            if response.value.action in missing_tools:
                missing_tools.remove(response.value.action)
        return results, attempts
