from __future__ import annotations

from dataclasses import dataclass

TAXONOMY_VERSION = "1.0"

# This small v1 vocabulary is intentionally conservative. Open tags retain concepts that do
# not yet deserve a canonical identity.
TERMS: dict[str, dict[str, str]] = {
    "topic": {
        "topic.beauty": "Beauty",
        "topic.fashion": "Fashion",
        "topic.food": "Food",
        "topic.fitness": "Fitness",
        "topic.gaming": "Gaming",
        "topic.tech": "Technology",
        "topic.travel": "Travel",
        "topic.education": "Education",
        "topic.entertainment": "Entertainment",
        "topic.lifestyle": "Lifestyle",
    },
    "format": {
        "format.tutorial": "Tutorial",
        "format.review": "Review",
        "format.vlog": "Vlog",
        "format.demonstration": "Demonstration",
        "format.storytelling": "Storytelling",
        "format.comedy": "Comedy",
        "format.interview": "Interview",
        "format.static_editorial": "Static editorial",
    },
    "intent": {
        "intent.educate": "Educate",
        "intent.entertain": "Entertain",
        "intent.inspire": "Inspire",
        "intent.inform": "Inform",
        "intent.promote": "Promote",
        "intent.community": "Community engagement",
    },
    "tone": {
        "tone.informative": "Informative",
        "tone.conversational": "Conversational",
        "tone.humorous": "Humorous",
        "tone.enthusiastic": "Enthusiastic",
        "tone.reflective": "Reflective",
        "tone.formal": "Formal",
    },
    "speaking_style": {
        "speaking.direct_to_camera": "Direct to camera",
        "speaking.voiceover": "Voiceover",
        "speaking.dialogue": "Dialogue",
        "speaking.text_led": "Text led",
    },
    "composition": {
        "composition.close_up": "Close-up",
        "composition.medium_shot": "Medium shot",
        "composition.wide_shot": "Wide shot",
        "composition.flat_lay": "Flat lay",
        "composition.product_focus": "Product focus",
    },
    "production": {
        "production.polished": "Polished",
        "production.casual": "Casual",
        "production.cinematic": "Cinematic",
        "production.studio": "Studio",
        "production.user_generated": "User-generated style",
    },
    "integration": {
        "integration.dedicated": "Dedicated sponsorship",
        "integration.segment": "Sponsored segment",
        "integration.organic_mention": "Organic mention",
        "integration.product_placement": "Product placement",
        "integration.affiliate": "Affiliate promotion",
    },
}


@dataclass(frozen=True, slots=True)
class ResolvedTerm:
    facet: str
    key: str
    label: str


def resolve_term(key: str) -> ResolvedTerm | None:
    normalized = key.strip().lower().replace(" ", "_")
    for facet, values in TERMS.items():
        if normalized in values:
            return ResolvedTerm(facet, normalized, values[normalized])
    return None


def seed_rows() -> list[dict[str, object]]:
    return [
        {"version": TAXONOMY_VERSION, "facet": facet, "key": key, "label": label, "aliases": []}
        for facet, values in TERMS.items()
        for key, label in values.items()
    ]
