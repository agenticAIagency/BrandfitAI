from __future__ import annotations

import json
from typing import Any, cast

from brandfit_core.providers.base import ProviderPayload


def _input(payload: ProviderPayload) -> dict[str, Any]:
    marker = "INPUT_JSON="
    return cast(dict[str, Any], json.loads(payload.prompt.split(marker, 1)[1]))


def _label(key: str, label: str, confidence: float = 0.82) -> dict[str, Any]:
    return {"key": key, "label": label, "confidence": confidence, "evidence": []}


def fixture_response(payload: ProviderPayload) -> dict[str, Any]:
    """Schema-valid deterministic responses for tests, demos, and fixture benchmarks."""

    data = _input(payload)
    if payload.prompt.startswith("TASK=post_analysis"):
        caption = str(data.get("caption") or "")
        lowered = caption.lower()
        topic_terms = [
            ("beauty", ["beauty", "makeup", "skincare"]),
            ("tech", ["tech", "phone", "laptop", "ai tool"]),
            ("food", ["food", "recipe", "खाना"]),
            ("fitness", ["fitness", "workout"]),
            ("fashion", ["fashion", "styling"]),
            ("travel", ["travel", "प्रवास"]),
            ("education", ["education", "शिक्षा"]),
            ("gaming", ["gaming"]),
            ("entertainment", ["entertainment"]),
            ("lifestyle", ["lifestyle", "routine"]),
        ]
        topic_key = next(
            (key for key, words in topic_terms if any(word in lowered for word in words)),
            "lifestyle",
        )
        topic = _label(f"topic.{topic_key}", topic_key.title(), 0.82)
        if any("\u0b80" <= char <= "\u0bff" for char in caption):
            language = "ta"
        elif "प्रवास" in caption or "आहे" in caption:
            language = "mr"
        elif any("\u0900" <= char <= "\u097f" for char in caption):
            language = "hi"
        elif any(word in lowered for word in [" hai", "aaj", "kaise"]):
            language = "hi-Latn"
        else:
            language = "en"
        evidence = data["available_evidence"][:3] or [
            {"post_id": data["post_id"], "kind": "profile", "locator": "profile:latest"}
        ]
        metrics = data.get("metrics") or {}
        followers = data.get("profile", {}).get("followers")
        likes = metrics.get("likes")
        comments = metrics.get("comments")
        engagement = None
        if followers and (likes is not None or comments is not None):
            engagement = ((likes or 0) + (comments or 0)) / followers
        coverage = float(data.get("extraction_coverage", 0.5))
        return {
            "original_language": language,
            "normalized_summary": caption[:500] or "Visual post with no supplied caption.",
            "content": {
                "topics": [topic],
                "formats": [_label("format.vlog", "Vlog", 0.68)],
                "intents": [_label("intent.inform", "Inform", 0.7)],
                "open_tags": [],
            },
            "communication": {
                "languages": [language],
                "tones": [_label("tone.conversational", "Conversational", 0.75)],
                "speaking_styles": [],
                "calls_to_action": [],
            },
            "visual": {
                "composition_styles": [_label("composition.medium_shot", "Medium shot", 0.65)],
                "production_styles": [_label("production.casual", "Casual", 0.7)],
                "settings": [],
                "dominant_colors": [],
                "product_presence": False,
            },
            "commercial": {
                "brand_mentions": [],
                "sponsorship_disclosed": None,
                "integration_styles": [],
                "affiliate_or_coupon_signal": False,
            },
            "safety": [],
            "performance": {
                "likes": likes,
                "comments": comments,
                "views": metrics.get("views"),
                "followers_at_capture": followers,
                "engagement_rate": engagement,
                "relative_engagement_percentile": None,
            },
            "confidence": {
                "extraction_coverage": coverage,
                "schema_completeness": 0.9,
                "evidence_grounding": 0.9,
                "overall": min(0.9, 0.45 + coverage * 0.5),
            },
            "evidence": evidence,
        }
    if payload.prompt.startswith("TASK=react_action"):
        used = int(data.get("calls_used", 0))
        missing = data.get("missing_tools") or []
        if used < 2 and missing:
            return {
                "thought": "Inspect missing extraction evidence.",
                "action": missing[0],
                "arguments": {},
            }
        return {
            "thought": "Available evidence has been checked.",
            "action": "finish",
            "arguments": {},
        }
    if payload.prompt.startswith("TASK=persona_narrative"):
        examples = data.get("evidence_examples") or []
        claims: list[dict[str, Any]] = []
        for example in examples[:5]:
            claims.append(
                {
                    "category": example["category"],
                    "statement": example["statement"],
                    "confidence": example["confidence"],
                    "evidence": example["evidence"],
                }
            )
        name = data.get("username", "The creator")
        return {
            "summary": f"{name} has a persona grounded in {data['post_count']} analyzed posts.",
            "claims": claims,
        }
    raise ValueError("Unknown fixture task")
