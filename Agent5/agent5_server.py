import asyncio
import logging
import os
import sys
import pandas as pd
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import ast

# --- Configuration & Path Setup ---
load_dotenv()
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper(),
                    format='%(asctime)s - %(name)s [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

# --- CRITICAL PATH DEFINITIONS ---
# Set Agent5 base directory (where this script is located)
AGENT5_BASE_DIR = Path(__file__).parent.resolve()
# Assuming data directory is the parent of Agent5/
PROJECT_ROOT = AGENT5_BASE_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"

# Check local Agent5 folder first, then shared root folder
EVALUATION_CSV_PATHS = [
    AGENT5_BASE_DIR / "data" / "evaluation" / "creator_evaluation_latest.csv",
    DATA_DIR / "evaluation" / "creator_evaluation_latest.csv"
]
PERSONA_DIR = DATA_DIR / "personas"


# --- Helper Functions ---

def _safe_str_to_list(vector_str: str) -> Optional[List[float]]:
    """Safely converts a string representation of a vector list to a Python list."""
    if not isinstance(vector_str, str) or not vector_str.strip():
        return None
    try:
        # Use ast.literal_eval for safe evaluation of the string as a Python literal
        result = ast.literal_eval(vector_str)
        if isinstance(result, (list, tuple)):
            return [float(x) for x in result]
        return None
    except:
        return None

def load_data() -> Optional[pd.DataFrame]:
    """Loads and deserializes the latest evaluation results CSV."""
    csv_path = None
    for path in EVALUATION_CSV_PATHS:
        if path.exists():
            csv_path = path
            break
    
    if csv_path is None:
        logger.error("Agent 4 output CSV file not found in any expected location.")
        return None

    try:
        # Define columns that hold string representations of vectors/lists
        VECTOR_COLUMNS = ['clustering_embedding', 'brand_fit_vector']
        converters = {col: _safe_str_to_list for col in VECTOR_COLUMNS}
        
        # Use the converters when reading the CSV to parse the vector strings
        # engine='python' and on_bad_lines='skip' is used to handle internal commas
        df = pd.read_csv(
            csv_path, 
            engine='python', 
            on_bad_lines='skip',
            converters=converters
        )
        logger.info(f"Loaded {len(df)} creator records from CSV. Vectors deserialized.")
        return df
    except Exception as e:
        logger.error(f"Failed to load or parse CSV file at {csv_path}: {e}")
        return None

def get_creator_dcpr(creator_id: str) -> Optional[Dict[str, Any]]:
    """Loads the original DCPR JSON for full context (Layer 1 summary)."""
    dcpr_path = PERSONA_DIR / creator_id / "dcpr_latest.json"
    if not dcpr_path.exists():
        return None
    try:
        with open(dcpr_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


# --- Core Matching Logic ---

class MatchingEngine:
    """Performs semantic search, filtering, and weighted ranking."""

    def __init__(self, df: pd.DataFrame, config: Dict[str, Any]):
        self.df = df
        self.ranking_weights = config.get("ranking_weights", {"similarity": 0.4, "quality": 0.4, "performance": 0.2})
        self.TOP_N = config.get("top_n_candidates", 5)
        
        # Initialize embedding model (must be the same model used by Agent 4)
        self.EMBEDDING_MODEL_NAME = config.get("embedding_model", "all-MiniLM-L6-v2")
        self.model = SentenceTransformer(self.EMBEDDING_MODEL_NAME)
        logger.info(f"Matching Engine initialized. Model: {self.EMBEDDING_MODEL_NAME}")

    def _get_query_vector(self, query: str) -> np.ndarray:
        """Generates the embedding vector for the user's text query."""
        query_vector = self.model.encode(query, convert_to_numpy=True)
        return query_vector.reshape(1, -1)

    def _apply_semantic_search(self, query_vector: np.ndarray, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates cosine similarity."""
        df_valid_vectors = df[df['brand_fit_vector'].notna()].copy()
        
        if df_valid_vectors.empty:
            df['similarity_score'] = 0.0
            return df

        # Convert list of lists (read from CSV) to NumPy matrix
        creator_vectors = np.array(df_valid_vectors['brand_fit_vector'].tolist())
        
        similarity_scores = cosine_similarity(creator_vectors, query_vector).flatten()
        
        scores_series = pd.Series(similarity_scores, index=df_valid_vectors.index)
        
        df['similarity_score'] = scores_series.reindex(df.index, fill_value=0.0)
        return df

    def _apply_hard_filters(self, df: pd.DataFrame, constraints: Dict[str, Any]) -> pd.DataFrame:
        """Applies non-negotiable constraints (Followers, Quality, Authenticity)."""
        
        filtered_df = df.copy()
        
        # Max Follower Constraint
        follower_max = constraints.get('follower_max', float('inf'))
        if follower_max != float('inf'):
             filtered_df = filtered_df[filtered_df['raw_follower_count'] <= follower_max]
            
        # Quality Score Minimum (Scale 0-100)
        min_quality = constraints.get('min_quality_score', 0)
        filtered_df = filtered_df[filtered_df['quality_score'] >= min_quality]
            
        # Authenticity Minimum (Scale 0-1)
        min_auth = constraints.get('min_authenticity_score', 0.0)
        filtered_df = filtered_df[filtered_df['raw_authenticity'] >= min_auth]
        
        return filtered_df

    def run_matching_workflow(self, query: str, constraints: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Orchestrates the matching process."""
        
        df = self.df.copy() # Work on a copy of the master DataFrame

        # 1. Generate Query Vector
        query_vector = self._get_query_vector(query)

        # 2. Semantic Search (Adds 'similarity_score' column)
        df_with_sim = self._apply_semantic_search(query_vector, df)
        
        # 3. Hard Filtering (Filter-First Strategy)
        df_filtered = self._apply_hard_filters(df_with_sim, constraints)
        
        if df_filtered.empty:
            logger.warning("No candidates remain after hard filtering.")
            return []
            
        # 4. Weighted Ranking and Selection
        
        # Normalize scores (Quality and Performance are already 0-100, normalize them to 0-1)
        df_filtered['quality_norm'] = df_filtered['quality_score'] / 100.0
        df_filtered['performance_norm'] = df_filtered['performance_score'] / 100.0
        
        # Calculate Final Match Score (Weighted Rank Formula)
        final_score = (
            self.ranking_weights['similarity'] * df_filtered['similarity_score'] +
            self.ranking_weights['quality'] * df_filtered['quality_norm'] +
            self.ranking_weights['performance'] * df_filtered['performance_norm']
        )
        
        df_filtered['final_match_score'] = final_score
        
        # Select top N candidates
        top_candidates = df_filtered.sort_values(by='final_match_score', ascending=False).head(self.TOP_N)
        
        # Convert to list of dictionaries for response
        output_cols = ['creator_id', 'final_match_score', 'quality_score', 'performance_score', 'similarity_score', 'raw_follower_count', 'raw_authenticity', 'cluster_id']
        
        # Ensure all output columns exist, fill with None if missing
        for col in output_cols:
             if col not in top_candidates.columns:
                 top_candidates[col] = None 
        
        return top_candidates[output_cols].to_dict(orient='records')


# --- FastAPI Setup ---
app = FastAPI(
    title="Agent 5 - Brand Matchmaker MVP",
    description="Receives brand query, performs semantic search against Agent 4 CSV output, and returns ranked creators.",
    version="1.0.0"
)

# Load data and initialize matcher globally
# This assumes the CSV file does not change during the server's runtime.
MATCHING_DF = load_data()

if MATCHING_DF is None or MATCHING_DF.empty:
    logger.error("Server starting without required data. Matching will not function.")
    # Initialize with an empty DataFrame to prevent immediate crash
    MATCHING_ENGINE = MatchingEngine(pd.DataFrame(), {"ranking_weights": {"similarity": 0.4, "quality": 0.4, "performance": 0.2}})
else:
    # Use a mock config since we don't have a real one yet
    mock_config = {
        "matching": {
            "ranking_weights": {"similarity": 0.4, "quality": 0.4, "performance": 0.2},
            "top_n_candidates": 5
        }
    }
    MATCHING_ENGINE = MatchingEngine(MATCHING_DF, mock_config)


# --- API Models ---
class QueryRequest(BaseModel):
    query: str = Field(..., description="Free-text description of the desired creator/campaign.")
    follower_max: Optional[int] = Field(None, description="Maximum number of followers allowed.")
    min_quality_score: Optional[float] = Field(None, description="Minimum overall quality score (0-100).")
    min_authenticity_score: Optional[float] = Field(None, description="Minimum raw authenticity score (0-1).")


class MatchResult(BaseModel):
    creator_id: str
    final_match_score: float
    quality_score: float
    performance_score: float
    similarity_score: float
    raw_follower_count: int
    raw_authenticity: float
    cluster_id: Optional[int] = Field(None, description="The cluster ID assigned by Agent 4.")
    short_summary: Optional[str] = Field(None, description="1-2 sentence persona summary (from DCPR JSON).")


class MatchResponse(BaseModel):
    candidates: List[MatchResult]
    query_parsed: Dict[str, Any]


@app.post("/match_creators", response_model=MatchResponse)
async def match_creators_endpoint(request: QueryRequest):
    """
    Receives a brand query, runs the matching workflow, and returns ranked creators.
    """
    if MATCHING_DF.empty:
        raise HTTPException(status_code=503, detail="Agent 5 data source is unavailable or empty. Run Agent 4.")

    # We use the LLM (EmailDrafting) module here primarily for its query parsing ability
    # This is a temporary dependency to avoid writing a manual NLP parser.
    # We must load the LLM module/client internally if we were to use it.
    
    # --- MOCKING LLM PARSING (To avoid needing the full LLM in the server startup) ---
    # In a real system, the MatchingEngine would call the LLM service to parse the text.
    # For this MVP, we mock the parsing function results based on the query structure.
    
    # 1. Simple Keyword Parser Mock
    query_text = request.query.lower()
    
    # Extract hard constraints from the request body, prioritizing API over defaults
    constraints = {
        'follower_max': request.follower_max if request.follower_max is not None else 200000,
        'min_quality_score': request.min_quality_score if request.min_quality_score is not None else 60.0,
        'min_authenticity_score': request.min_authenticity_score if request.min_authenticity_score is not None else 0.5,
        'required_keywords': [w for w in query_text.split() if len(w) > 3],
        'campaign_goals': "Product review campaign"
    }

    # 2. Run Matching
    ranked_candidates_dicts = MATCHING_ENGINE.run_matching_workflow(
        query=request.query,
        constraints=constraints
    )

    # 3. Final Assembly (Injecting DCPR summary for UI)
    final_candidates = []
    for candidate in ranked_candidates_dicts:
        # Load DCPR to fetch the human-readable summary (Layer 1)
        dcpr_data = get_creator_dcpr(candidate['creator_id'])
        
        candidate_model = MatchResult(
            **candidate,
            short_summary=dcpr_data.get('identity_summary', {}).get('short_summary', 'Summary not available.'),
            cluster_id=candidate.get('cluster_id', -1) # Ensure cluster_id is present
        )
        final_candidates.append(candidate_model)

    return MatchResponse(
        candidates=final_candidates,
        query_parsed=constraints
    )

if __name__ == "__main__":
    import uvicorn
    # This server should run on a specific port, e.g., 8005
    uvicorn.run(app, host="0.0.0.0", port=8005)
