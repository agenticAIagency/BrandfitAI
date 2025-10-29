# --- Setup Logging FIRST ---
import logging
import os
import sys
import operator
from pathlib import Path

# Base directory for Agent 3
agent3_base_dir = Path(__file__).parent.resolve()

# Load .env relative to this file's location
from dotenv import load_dotenv
dotenv_path = agent3_base_dir / '.env'
load_dotenv(dotenv_path=dotenv_path)
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format='%(asctime)s - %(name)s [%(levelname)s] - %(message)s',
    handlers=[
        logging.FileHandler(agent3_base_dir / "logs" / "agent3_graph.log", encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)
logger.info(f"--- agent3_graph.py logging configured ---")


# --- Load libraries safely ---
try:
    import asyncio
    import json
    import yaml
    from typing import TypedDict, List, Dict, Any, Optional
    from langgraph.graph import StateGraph, END
    from pydantic import BaseModel, Field
    import chromadb 
    logger.info("--- Standard libraries imported successfully ---")
except ImportError as e:
    logger.exception(f"--- FATAL: Failed to import standard libraries: {e} ---")
    sys.exit(1)

# --- Import project modules safely ---
project_base_dir = agent3_base_dir
if str(project_base_dir) not in sys.path:
     sys.path.insert(0, str(project_base_dir))
     logger.info(f"--- Added {project_base_dir} to sys.path ---")

try:
    logger.info("--- Importing Agent 3 project modules ---")
    from modules.ingestion import DataIngestionModule
    from modules.dcpr_calculator import DCPRCalculator
    from modules.summarization import PersonaSummarizationModule 
    from modules.validation import ValidationModule
    from memory.store import PersonaMemoryStore
    from memory.versioning import VersionManager
    from utils.retry import RetryHandler
    logger.info("--- Agent 3 project modules imported successfully ---")
except ImportError as e:
    logger.exception(f"--- FATAL: Failed to import Agent 3 project modules or utils: {e} ---")
    sys.exit(1)

# --- FIX: Define Missing Pydantic Model (Required for state definition) ---
class ReflectionResult(BaseModel):
    """Schema for the LLM's self-reflection output (BDI Loop)."""
    critique: str = Field(description="A critique of the generated summary.")
    score: float = Field(description="A numerical score (0.0 to 1.0) for summary quality.")
    looks_good: bool = Field(description="True if the summary passes the reflection and needs no retry.")


# --- Load Configuration ---
CONFIG_PATH = agent3_base_dir / "config.yaml"
config = {}
try:
    logger.info(f"--- Loading configuration from {CONFIG_PATH} ---")
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
    logger.info("--- Configuration loaded successfully ---")
except FileNotFoundError:
    logger.error(f"--- FATAL: Configuration file not found at {CONFIG_PATH}. Exiting. ---")
    sys.exit(1)


# --- State Definition ---
class Agent3State(TypedDict):
    """Holds the state for the persona building graph."""
    creator_id: str
    all_posts: List[Dict[str, Any]]
    new_posts: List[Dict[str, Any]]
    old_stats: Optional[Dict[str, Any]]
    final_dcpr: Dict[str, Any]
    final_stats: Dict[str, Any]
    version_id: str
    build_type: str
    error_message: Optional[str]
    # Keep reflection keys for robust state initialization
    summary_critique: Optional[str]
    reflection_score: float
    reflection_attempts: int
    processed_posts: List[Dict[str, Any]]


# --- Initialize Modules ---
ingestion = DataIngestionModule()
calculator = DCPRCalculator()
# Note: Summarization module requires LLM keys in .env
summarization = PersonaSummarizationModule() 
validation = ValidationModule()
memory_store = PersonaMemoryStore()
version_manager = VersionManager()
retry_handler = RetryHandler(max_retries=2, base_delay=1.0)


# --- VECTOR DB BYPASS (Required for JSON-only save) ---
vector_db_client = None
persona_collection = None
post_collection = None
persist_dir = "" 
logger.warning("--- Vector DB storage is intentionally bypassed. Saving only to JSON. ---")


# --- GRAPH NODES ---
# NOTE: Removed unused/bypassed nodes (reflect_on_summary, store_embeddings) definitions 
# to clean up the code and prevent recursion risks.

async def check_for_stats(state: Agent3State) -> dict:
    """Checks if stats exist to decide the build path."""
    creator_id = state["creator_id"]
    logger.info(f"--- [Node: check_for_stats] Checking stats for {creator_id}")
    if memory_store.stats_exist(creator_id):
        old_stats = memory_store.load_latest_stats(creator_id)
        if old_stats:
            # FIX: Ensure all keys needed for the graph are initialized/passed
            return {"build_type": "incremental", "old_stats": old_stats, "reflection_attempts": 0, "processed_posts": []} 
        else:
            return {"build_type": "full", "all_posts": [], "new_posts": [], "reflection_attempts": 0, "processed_posts": []}
    else:
        return {"build_type": "full", "all_posts": [], "new_posts": [], "reflection_attempts": 0, "processed_posts": []}

async def load_data(state: Agent3State) -> dict:
    """Node to load all or new posts based on build_type."""
    creator_id = state["creator_id"]
    build_type = state["build_type"]
    logger.info(f"--- [Node: load_data] Loading data for {creator_id} (Type: {build_type})")

    try:
        if build_type == "full":
            all_posts = await ingestion.load_creator_data(creator_id, file_pattern="*.json")
            min_posts_required = config.get("bdi", {}).get("min_posts_threshold", 5)
            if len(all_posts) < min_posts_required:
                return {"error_message": f"Insufficient posts for full build: {len(all_posts)} < {min_posts_required}"}
            logger.info(f"--- Loaded {len(all_posts)} posts for full build.")
            return {"all_posts": all_posts, "new_posts": []}

        elif build_type == "incremental":
            old_stats = state.get("old_stats")
            last_date = old_stats.get("last_post_date")
            new_posts = await ingestion.load_new_posts_since(creator_id, last_date)
            if not new_posts:
                return {"error_message": "No new posts to process"}
            logger.info(f"--- Loaded {len(new_posts)} new posts for incremental build.")
            return {"new_posts": new_posts, "all_posts": []}
        else:
             return {"error_message": f"Unknown build_type: {build_type}"}

    except Exception as e:
        logger.error(f"--- Error during data loading for {creator_id}: {e}", exc_info=True)
        return {"error_message": f"Data loading failed: {e}"}


async def calculate_dcpr(state: Agent3State) -> dict:
    """Node to run the correct calculation logic."""
    build_type = state["build_type"]
    creator_id = state["creator_id"]

    try:
        version_id = version_manager.create_version(creator_id)
        posts_to_process = []
        
        if build_type == "full":
            posts_to_process = state.get("all_posts")
            dcpr, stats = calculator.create_from_scratch(posts_to_process)
        else: # Incremental
            posts_to_process = state.get("new_posts")
            dcpr, stats = calculator.update_incrementally(state["old_stats"], posts_to_process)

        dcpr["creator_id"] = creator_id
        dcpr["persona_version"] = version_id
        
        return {
            "final_dcpr": dcpr,
            "final_stats": stats,
            "version_id": version_id,
            "processed_posts": posts_to_process
            }
    except Exception as e:
        logger.error(f"--- Error during DCPR calculation for {creator_id}: {e}", exc_info=True)
        return {"error_message": f"DCPR calculation failed: {e}"}

async def summarize_persona(state: Agent3State) -> dict:
    creator_id = state["creator_id"]
    
    if not state.get("final_dcpr"):
        return {"error_message": "Cannot summarize persona without calculated DCPR."}

    try:
        identity_summary_dict = await retry_handler.execute(
            summarization.generate_identity_summary,
            state["final_dcpr"]
        )

        updated_dcpr = state["final_dcpr"].copy()
        updated_dcpr["identity_summary"] = identity_summary_dict
        
        # NOTE: If LLM fails consistently, it should fail via retry_handler.execute
        return {"final_dcpr": updated_dcpr, "summary_critique": None} 
    except Exception as e:
        logger.error(f"--- Summarization failed after retries for {creator_id}: {e}", exc_info=True)
        return {"error_message": f"LLM Summarization failed: {e}"}


async def validate_and_save_json(state: Agent3State) -> dict:
    """Node to validate and save the final DCPR/Stats JSON artifacts."""
    creator_id = state["creator_id"]
    version_id = state.get("version_id", "unknown")

    if not state.get("final_dcpr") or not state.get("final_stats") or not version_id:
          return {"error_message": "Cannot save JSON without complete DCPR, stats, and version ID."}

    try:
        # 1. Validation (Guardrail)
        validation_result = validation.validate_dcpr_output(state["final_dcpr"])
        if validation_result.get("errors"):
             return {"error_message": f"Validation failed: {validation_result['errors']}"}

        # 2. Save JSONs (Long-Term Memory)
        save_successful = await retry_handler.execute(
            memory_store.save_persona_and_stats,
            creator_id=creator_id,
            dcpr=state["final_dcpr"],
            stats=state["final_stats"],
            version=version_id
        )

        if not save_successful:
             return {"error_message": "Save JSON operation failed unexpectedly after retries."}

        logger.info(f"--- Successfully validated and saved JSON version {version_id}")
        return {}
    except Exception as e:
        logger.error(f"--- Validation or JSON Save failed after retries for {creator_id}: {e}", exc_info=True)
        return {"error_message": f"Validation or JSON Save failed: {e}"}

# --- UNUSED NODES (REMOVED FROM GRAPH FLOW) ---

async def handle_error(state: Agent3State) -> dict:
    """Logs errors."""
    error_msg = state.get('error_message', 'Unknown error')
    creator_id = state.get('creator_id', 'Unknown creator')
    logger.error(f"--- [Node: handle_error] Error processing {creator_id}: {error_msg}")
    return {} # Terminal node


# --- GRAPH EDGES ---
def should_continue(state: Agent3State) -> str:
    """Checks for errors before proceeding."""
    if state.get("error_message") == "No new posts to process":
        return "stop"
    elif state.get("error_message"):
        return "handle_error"
    return "continue"


# --- ASSEMBLE THE GRAPH ---
logger.info("--- Assembling Agent 3 graph ---")
workflow = StateGraph(Agent3State)

# Add Nodes
workflow.add_node("check_for_stats", check_for_stats)
workflow.add_node("load_data", load_data)
workflow.add_node("calculate_dcpr", calculate_dcpr)
workflow.add_node("summarize_persona", summarize_persona)
workflow.add_node("validate_and_save_json", validate_and_save_json)
workflow.add_node("handle_error", handle_error)

# Define Edges
workflow.set_entry_point("check_for_stats")
workflow.add_edge("check_for_stats", "load_data")

workflow.add_conditional_edges(
    "load_data",
    should_continue,
    {
        "continue": "calculate_dcpr",
        "handle_error": "handle_error",
        "stop": END
    }
)

workflow.add_conditional_edges(
    "calculate_dcpr",
     should_continue,
    {"continue": "summarize_persona", "handle_error": "handle_error"}
)

# --- CRITICAL FIX: Direct Flow to JSON Save, then END ---
workflow.add_conditional_edges(
    "summarize_persona",
     should_continue,
    {"continue": "validate_and_save_json", "handle_error": "handle_error"}
)

workflow.add_conditional_edges(
    "validate_and_save_json",
    should_continue,
    {
        "continue": END, # FINAL TERMINAL CONDITION
        "handle_error": "handle_error"
    }
)

# Error handler always goes to END
workflow.add_edge("handle_error", END)

# Compile
try:
    app = workflow.compile()
    logger.info("--- Agent 3 graph compiled successfully ---")
except Exception as e:
    logger.exception("--- FATAL: Failed to compile Agent 3 graph ---")
    sys.exit(1)

# --- Main Execution Function ---
async def run_agent3_graph():
    """Runs the graph for all dynamically found creators."""
    logger.info("=========================================")
    logger.info("STARTING AGENT 3: DCPR WORKFLOW")
    logger.info("=========================================")
    creators_to_process = ["demo_creator_001", "demo_creator_002"]

    initial_states = [
        {
            "creator_id": creator,
            "all_posts": [], "new_posts": [], "old_stats": None,
            "final_dcpr": {}, "final_stats": {}, "version_id": "",
            "build_type": "", "error_message": None,
            "reflection_attempts": 0, "reflection_score": 0.0, "summary_critique": None,
            "processed_posts": []
        } for creator in creators_to_process
    ]
    
    for initial_state in initial_states:
        creator = initial_state["creator_id"]
        logger.info(f"\n--- Running for Creator: {creator} ---")
        try:
            # The recursion limit is now handled by the fixed graph logic
            final_state = await app.ainvoke(initial_state, {"recursion_limit": 10}) # Increased limit just in case of LLM slow-down
            if final_state.get("error_message"):
                 logger.error(f"Graph FAILED for {creator}. Error: {final_state['error_message']}")
            else:
                 logger.info(f"Graph COMPLETED for {creator}.")
        except Exception as e:
            logger.exception(f"Unhandled exception for {creator}: {e}")
        logger.info(f"--- Finished for Creator: {creator} ---\n")

    logger.info("=========================================")
    logger.info("AGENT 3 WORKFLOW COMPLETE")
    logger.info("=========================================")

# Direct execution block
if __name__ == "__main__":
    logger.info("--- Running agent3_graph.py directly ---")
    # Setup directories
    Path(agent3_base_dir / "logs").mkdir(exist_ok=True)
    Path(agent3_base_dir / "data" / "creators").mkdir(parents=True, exist_ok=True)
    Path(agent3_base_dir / "data" / "personas").mkdir(parents=True, exist_ok=True)

    asyncio.run(run_agent3_graph())
    logger.info("--- Direct execution finished ---")