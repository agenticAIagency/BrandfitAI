import asyncio
import logging
import os
import sys
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from tqdm.asyncio import tqdm_asyncio
from typing import Dict, Any, List, Optional
import json
import ast # Required for safe string-to-list conversion

# --- Setup Paths and Logging ---
load_dotenv()
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper(),
                    format='%(asctime)s - %(name)s [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
# Set AGENT5_BASE_DIR to the directory containing this script (E:\AGENTIC_AI\Project_agent3\Agent5)
AGENT5_BASE_DIR = Path(__file__).parent.resolve()
# Assuming the main data directory is the parent of Agent5/ (E:\AGENTIC_AI\Project_agent3)
PROJECT_ROOT = AGENT5_BASE_DIR.parent
DATA_DIR = PROJECT_ROOT / "data" 

# --- CRITICAL PATH DEFINITIONS ---
# 1. Primary Location (Manual/Local to Agent 5) - Looks for E:\AGENTIC_AI\Project_agent3\Agent5\data\evaluation\
LOCAL_EVALUATION_CSV_PATH = AGENT5_BASE_DIR / "data" / "evaluation" / "creator_evaluation_latest.csv"

# 2. Secondary Location (Ideal Shared Root) - Looks for E:\AGENTIC_AI\Project_agent3\data\evaluation\
SHARED_EVALUATION_CSV_PATH = DATA_DIR / "evaluation" / "creator_evaluation_latest.csv"

PERSONA_DIR = DATA_DIR / "personas"
EMAIL_OUTPUT_DIR = DATA_DIR / "email_drafts"
EMAIL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
logger.info(f"Email drafts will be saved to: {EMAIL_OUTPUT_DIR}")


# --- Import Project Modules ---
try:
    # Need to add project root to path to import utils
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
        
    from utils.retry import RetryHandler
    from Agent5.agent5_matching import MatchingEngine
    from Agent5.agent5_email import EmailDrafting
except ImportError as e:
    logger.fatal(f"FATAL: Missing project module. Check path and requirements: {e}")
    sys.exit(1)
except Exception as e:
    logger.fatal(f"FATAL: Error during module initialization: {e}")
    sys.exit(1)


# Helper function to safely convert string list/array representation to a Python list
def _safe_str_to_list(vector_str: str) -> Optional[List[float]]:
    """Converts a string representation of a list of floats (e.g., '[-0.1, 0.2]') to a list."""
    if not isinstance(vector_str, str) or not vector_str.strip():
        return None
    try:
        # Use ast.literal_eval for safe evaluation of the string as a Python literal
        # Ensures that the data is treated as a list/tuple of numbers, not arbitrary code
        result = ast.literal_eval(vector_str)
        if isinstance(result, (list, tuple)):
            # Convert any tuples to lists and ensure elements are floats
            return [float(x) for x in result]
        return None
    except:
        # Log this error in the matching module if possible, here we return None
        return None


def load_data_and_check() -> Optional[pd.DataFrame]:
    """Loads the latest evaluation results CSV, checking local and shared paths."""
    
    # Prioritize the local Agent 5 folder (where you manually dragged it)
    if LOCAL_EVALUATION_CSV_PATH.exists():
        csv_path = LOCAL_EVALUATION_CSV_PATH
        logger.info(f"Using local Agent 5 CSV path: {csv_path}")
    # Check the shared data folder (where it should ideally be)
    elif SHARED_EVALUATION_CSV_PATH.exists():
        csv_path = SHARED_EVALUATION_CSV_PATH
        logger.warning(f"Using shared CSV path: {csv_path}")
    else:
        logger.error(f"Agent 4 output not found in primary location ({LOCAL_EVALUATION_CSV_PATH}) or secondary location ({SHARED_EVALUATION_CSV_PATH}). Run Agent 4 first.")
        return None
    
    try:
        # Define columns that hold string representations of vectors/lists
        VECTOR_COLUMNS = ['clustering_embedding', 'brand_fit_vector', 'embedding_text_1', 'embedding_text_2']
        
        # Create the converters dictionary using the safe function
        converters = {col: _safe_str_to_list for col in VECTOR_COLUMNS}
        
        # Use the converters when reading the CSV
        df = pd.read_csv(
            csv_path, 
            engine='python', 
            on_bad_lines='skip',
            converters=converters # Automatically converts the vector columns
        )
        logger.info(f"Loaded {len(df)} creator records from CSV. Vector columns successfully deserialized.")
        return df
    except Exception as e:
        logger.error(f"Failed to load or parse CSV file at {csv_path}: {e}")
        return None

def load_creator_dcpr(creator_id: str) -> Optional[Dict[str, Any]]:
    """Loads the original DCPR JSON for full context."""
    # Assumes DCPR is saved under E:\AGENTIC_AI\Project_agent3\data\personas\<creator_id>\
    dcpr_path = PERSONA_DIR / creator_id / "dcpr_latest.json"
    if not dcpr_path.exists():
        logger.warning(f"DCPR file not found for creator: {creator_id} at {dcpr_path}")
        return None
    try:
        with open(dcpr_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load DCPR for {creator_id}: {e}")
        return None


async def run_agent5_workflow(user_query: str):
    """Main orchestration function for Agent 5."""
    logger.info("=========================================")
    logger.info("STARTING AGENT 5: MATCHMAKER WORKFLOW")
    logger.info("=========================================")

    # Initialize components
    retry_handler = RetryHandler(max_retries=2, base_delay=1.0)
    # Matcher/Email config placeholder (using empty dict for now)
    config = {"matching": {}, "embeddings": {}, "email": {}} 
    matcher = MatchingEngine(config=config, data_dir=str(DATA_DIR))
    emailer = EmailDrafting(retry_handler=retry_handler)

    # --- Phase 1 & 2: Query Parsing, Matching, Filtering ---
    
    # 1. Load Data
    full_df = load_data_and_check()
    if full_df is None:
        return

    # 2. Parse Query (LLM call)
    constraints = await emailer.parse_query_to_constraints(user_query)
    logger.info(f"Parsed Constraints: {constraints}")
    
    # 3. Run Matching (Semantic Search, Filtering, Ranking)
    ranked_candidates = matcher.run_matching_workflow(
        query=user_query,
        results_df=full_df,
        constraints=constraints
    )
    
    if not ranked_candidates:
        logger.warning("No suitable creators found after matching and filtering.")
        return

    logger.info(f"Found {len(ranked_candidates)} candidates.")
    
    # --- Phase 3: Email Drafting (Parallel) ---
    
    async def draft_task(candidate):
        """Helper to draft email and save for a single creator."""
        creator_id = candidate['creator_id']
        # Load the original DCPR for personalization
        dcpr_data = load_creator_dcpr(creator_id)
        if dcpr_data:
            return await emailer.draft_personalized_email(
                brand_goals=constraints.get('campaign_goals', 'Collaborative content'),
                creator_data=candidate,
                creator_dcpr=dcpr_data,
                storage_path=EMAIL_OUTPUT_DIR
            )
        else:
            logger.error(f"Skipping email for {creator_id}: DCPR file missing.")
            return None

    logger.info(f"Starting parallel email drafting for {len(ranked_candidates)} creators...")
    
    # Execute drafting in parallel using tqdm for progress
    # Note: Use list comprehension for generator to ensure all tasks are created
    draft_tasks = [draft_task(c) for c in ranked_candidates]
    draft_paths = await tqdm_asyncio.gather(*draft_tasks)

    logger.info("=========================================")
    logger.info("AGENT 5 WORKFLOW COMPLETE")
    logger.info(f"Top {len(ranked_candidates)} matches found. Email drafts stored.")
    logger.info(f"Final drafts location: {EMAIL_OUTPUT_DIR}")
    logger.info("=========================================")


if __name__ == "__main__":
    # Example user query based on the fitness/tech context
    QUERY = "I need an authentic, high-quality content creator who focuses on motivational fitness reels. Max followers should be 200,000. Goal: product review campaign."
    
    logger.info(f"Running Agent 5 with example query: '{QUERY}'")
    try:
        # Run the asynchronous workflow
        asyncio.run(run_agent5_workflow(QUERY))
    except Exception as e:
        logger.fatal(f"FATAL: Unhandled error in Agent 5 main execution: {e}", exc_info=True)
