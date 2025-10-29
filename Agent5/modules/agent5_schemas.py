from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class EmailDraft(BaseModel):
    """Structured fields for the outreach email."""
    subject: str = Field(description="A professional and engaging subject line for the outreach email.")
    body: str = Field(description="The complete body of the personalized outreach email, using data points from the creator's DCPR (e.g., specific style trait, specific topic, audience intent) to justify the collaboration.")

class TopMatch(BaseModel):
    """Details for a single top-matched creator."""
    creator_id: str = Field(description="The unique ID of the matched creator.")
    final_match_score_estimated: float = Field(description="A calculated match score from 0.0 to 1.0, estimating the fit against the user's constraints and query.")
    reason_for_fit: str = Field(description="A concise, data-backed justification of why this creator is a strong fit for the query, referencing specific metrics (e.g., 'authenticity score 0.91', 'focus on reels').")
    email_draft: EmailDraft

class MatchmakerOutput(BaseModel):
    """The final structured JSON output for the UI."""
    search_criteria_applied: Dict[str, Any] = Field(description="The key filters and constraints inferred and applied by the LLM from the user's query.")
    top_matches: List[TopMatch] = Field(description="An array containing the top 3-5 best-matched creators.")
    recommendation_summary: str = Field(description="A concluding summary recommending the list and highlighting the overall quality of the match.")

# --- Helper Model for Ingestion ---
class CreatorDCPR(BaseModel):
    """A minimal Pydantic model to ingest the core DCPR fields needed for matching."""
    creator_id: str
    persona_version: str
    identity_summary: Dict[str, Any]
    content_intelligence: Dict[str, Any]
    style_intelligence: Dict[str, Any]
    performance_intelligence: Dict[str, Any]
    # Note: embedding is excluded as the LLM does the semantic comparison directly.
