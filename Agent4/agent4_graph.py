# --- Setup Logging FIRST ---
import logging
import os
import sys
from pathlib import Path

# Base directory for Agent 4
agent4_base_dir = Path(__file__).parent.resolve()

# Load .env relative to this file's location
from dotenv import load_dotenv
dotenv_path = agent4_base_dir / '.env'
load_dotenv(dotenv_path=dotenv_path)
logger = logging.getLogger(__name__) # Get logger first
logger.info(f"--- agent4_graph.py: Attempting to load .env from {dotenv_path} ---") # Log path
if dotenv_path.exists():
    load_dotenv(dotenv_path=dotenv_path)
    logger.info(f"--- agent4_graph.py: Loaded .env from {dotenv_path} ---")
else:
    logger.warning(f"--- agent4_graph.py: .env file not found at {dotenv_path}. Using system environment variables. ---")


# Configure logging relative to Agent 4 directory
log_dir_relative = os.getenv("AGENT4_LOGS_DIR", "./logs") # Path relative to agent4_base_dir
log_dir_absolute = agent4_base_dir / log_dir_relative
log_dir_absolute.mkdir(parents=True, exist_ok=True)
log_file_path = log_dir_absolute / "agent4.log"

# Clear previous handlers to avoid duplicate logs if run multiple times
root_logger = logging.getLogger()
if root_logger.hasHandlers():
    root_logger.handlers.clear()
# logging.getLogger().handlers.clear() # Optional: Clears root logger handlers

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format='%(asctime)s - %(name)s [%(levelname)s] - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path, encoding='utf-8'),
        logging.StreamHandler(sys.stdout) # To console
    ]
)
# Re-get logger after basicConfig
logger = logging.getLogger(__name__)
logger.info(f"--- agent4_graph.py logging configured. Log file: {log_file_path} ---")

# --- Now load other libraries safely ---
try:
    import operator
    import asyncio
    import json
    import yaml
    from typing import TypedDict, List, Dict, Any, Optional
    import pandas as pd
    from langgraph.graph import StateGraph, END
    logger.info("--- Standard libraries imported successfully ---")
except ImportError as e:
    logger.exception(f"--- FATAL: Failed to import standard libraries: {e} ---")
    sys.exit(1)

# --- Import project modules safely ---
# Ensure the parent directory (Project_agent3) is in the Python path
# This allows importing 'utils'
project_base_dir = agent4_base_dir.parent
if str(project_base_dir) not in sys.path:
     sys.path.insert(0, str(project_base_dir))
     logger.info(f"--- Added {project_base_dir} to sys.path ---")

try:
    logger.info("--- Importing Agent 4 project modules ---")
    from Agent4.modules.ingestion import DCPR_Ingestion
    from Agent4.modules.clustering import CreatorClustering
    from Agent4.modules.scoring import CreatorScoring
    from Agent4.memory.store import EvaluationStore
    # Import utils from the parent directory structure
    from utils.retry import RetryHandler
    logger.info("--- Agent 4 project modules imported successfully ---")
except ImportError as e:
    logger.exception(f"--- FATAL: Failed to import Agent 4 project modules or utils: {e} ---")
    sys.exit(1)
except Exception as e:
     logger.exception(f"--- FATAL: Error during Agent 4 module import: {e} ---")
     sys.exit(1)


# --- Load Configuration ---
CONFIG_PATH = agent4_base_dir / "config.yaml" # Use absolute path
config = {}
try:
    logger.info(f"--- Loading configuration from {CONFIG_PATH} ---")
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
    logger.info("--- Configuration loaded successfully ---")
except FileNotFoundError:
    logger.error(f"--- FATAL: Configuration file not found at {CONFIG_PATH}. Exiting. ---")
    sys.exit(1)
except yaml.YAMLError as e:
     logger.error(f"--- FATAL: Error parsing configuration file {CONFIG_PATH}: {e}. Exiting. ---")
     sys.exit(1)
except Exception as e:
     logger.exception(f"--- FATAL: Unexpected error loading config: {e}. Exiting. ---")
     sys.exit(1)


# --- State Definition ---
class Agent4State(TypedDict):
    """Holds the state for the evaluation graph."""
    raw_dcprs: List[Dict[str, Any]]
    clustering_results: Optional[pd.DataFrame]
    clustering_metadata: Dict[str, Any]
    scoring_results: Optional[pd.DataFrame]
    final_evaluation_results: Optional[pd.DataFrame] # Can be large, consider storing path instead
    error_message: Optional[str]

# --- Initialize Modules with Config ---
try:
    logger.info("--- Initializing Agent 4 modules ---")
    paths_config = config.get("paths", {})

    # --- CORRECTED PATH LOGIC ---
    # Get the relative path from config
    dcpr_input_dir_relative = paths_config.get("dcpr_input_dir", "../Project_agent3/data/personas")
    # Construct the absolute path based on *this script's location* (agent4_base_dir)
    # Go up one level from Agent4 dir, then join the relative path components
    dcpr_input_dir_absolute = (agent4_base_dir.parent / dcpr_input_dir_relative).resolve()

    output_dir_relative = paths_config.get("output_dir", "./data/evaluation")
    # Output is relative to Agent4 dir
    output_dir_absolute = (agent4_base_dir / output_dir_relative).resolve()
    # --- END CORRECTED PATH LOGIC ---

    logger.info(f"Input DCPR directory resolved to: {dcpr_input_dir_absolute}")
    logger.info(f"Output evaluation directory resolved to: {output_dir_absolute}")

    # Ensure directories exist before initializing modules that use them
    output_dir_absolute.mkdir(parents=True, exist_ok=True)
    # Check if input dir exists *before* initializing ingestion
    if not dcpr_input_dir_absolute.exists():
         logger.warning(f"--- Input DCPR directory specified in config does not exist: {dcpr_input_dir_absolute} ---")
         # Allow initialization, but loading will fail later if dir isn't created.

    ingestion = DCPR_Ingestion(dcpr_dir=str(dcpr_input_dir_absolute)) # Pass string path
    clustering = CreatorClustering(config=config)
    scoring = CreatorScoring(config=config)
    evaluation_store = EvaluationStore(output_dir=str(output_dir_absolute)) # Pass string path
    retry_handler = RetryHandler(max_retries=2, base_delay=1.0)
    logger.info("--- Agent 4 modules initialized successfully ---")
except KeyError as e:
     logger.exception(f"--- FATAL: Missing key during module initialization, check config.yaml: {e} ---")
     sys.exit(1)
except Exception as e:
     logger.exception(f"--- FATAL: Error initializing Agent 4 modules: {e} ---")
     sys.exit(1)


# --- GRAPH NODES (Keep implementation the same as before) ---
async def load_dcprs(state: Agent4State) -> dict:
    """Loads the latest DCPR files from Agent 3's output."""
    logger.info("--- [Node: load_dcprs] Loading DCPRs...")
    try:
        # load_latest_dcprs now uses the absolute path set during initialization
        dcprs = await ingestion.load_latest_dcprs()
        if not dcprs:
            # Modify error message if the directory didn't exist initially
            if not ingestion.dcpr_root_dir.exists():
                 return {"error_message": f"Input directory does not exist: {ingestion.dcpr_root_dir}"}
            else:
                 return {"error_message": "No valid DCPRs found to process in the input directory."}
        logger.info(f"--- Loaded {len(dcprs)} DCPRs.")
        return {"raw_dcprs": dcprs}
    except Exception as e:
        logger.error(f"--- Failed to load DCPRs: {e}", exc_info=True)
        return {"error_message": f"DCPR loading failed: {e}"}

async def cluster_creators(state: Agent4State) -> dict:
    """Performs clustering on the loaded DCPRs."""
    logger.info("--- [Node: cluster_creators] Clustering creators...")
    if not state.get("raw_dcprs"):
        return {"error_message": "Cannot cluster without loaded DCPRs."}
    try:
        # Wrap synchronous CPU-bound task
        cluster_df, meta = await asyncio.to_thread(clustering.cluster_creators, state["raw_dcprs"])
        logger.info(f"--- Clustering completed. Status: {meta.get('status', 'unknown')}")
        return {"clustering_results": cluster_df, "clustering_metadata": meta}
    except Exception as e:
        logger.error(f"--- Clustering failed: {e}", exc_info=True)
        return {"error_message": f"Clustering failed: {e}"}

async def calculate_scores(state: Agent4State) -> dict:
    """Calculates quality and performance scores."""
    logger.info("--- [Node: calculate_scores] Calculating scores...")
    if not state.get("raw_dcprs"):
        return {"error_message": "Cannot calculate scores without loaded DCPRs."}
    try:
        # Wrap synchronous CPU-bound task
        scores_df = await asyncio.to_thread(scoring.calculate_scores, state["raw_dcprs"])
        logger.info(f"--- Score calculation complete for {len(scores_df)} creators.")
        return {"scoring_results": scores_df}
    except Exception as e:
        logger.error(f"--- Score calculation failed: {e}", exc_info=True)
        return {"error_message": f"Score calculation failed: {e}"}

async def combine_and_save_results(state: Agent4State) -> dict:
    """Combines clustering and scoring results and saves them."""
    logger.info("--- [Node: combine_and_save] Combining and saving results...")
    cluster_df = state.get("clustering_results")
    scores_df = state.get("scoring_results")
    cluster_meta = state.get("clustering_metadata", {})

    final_df = None # Initialize

    if scores_df is None:
         return {"error_message": "Missing scoring results to save."}

    if cluster_df is None:
        # This case handles when clustering was skipped (e.g., too few creators)
        logger.warning("--- Clustering was skipped or failed; saving scores only.")
        final_df = scores_df.copy()
        # Ensure 'cluster_id' column exists even if skipped
        if 'cluster_id' not in final_df.columns:
            final_df['cluster_id'] = -1
    else:
        # Merge results on creator_id
        try:
             logger.info("--- Merging scoring and clustering results...")
             # Ensure creator_id columns have the same type if needed before merge
             cluster_df['creator_id'] = cluster_df['creator_id'].astype(str)
             scores_df['creator_id'] = scores_df['creator_id'].astype(str)
             # Use outer merge to keep all creators even if one step failed for them
             final_df = pd.merge(scores_df, cluster_df, on="creator_id", how="outer")
             # Handle cases where a creator might be missing from one result
             final_df['cluster_id'] = final_df['cluster_id'].fillna(-1).astype(int)
             # Fill missing score columns with a default (e.g., 0 or NaN) if needed
             for col in ['quality_score', 'performance_score']:
                  if col not in final_df.columns: final_df[col] = 0.0 # Or np.nan
                  else: final_df[col] = final_df[col].fillna(0.0) # Or np.nan
             logger.info(f"--- Merge complete. Final DataFrame shape: {final_df.shape}")

        except Exception as e:
            logger.error(f"--- Failed to merge clustering and scoring results: {e}", exc_info=True)
            return {"error_message": f"Failed to combine results: {e}"}

    # Save using the EvaluationStore (wrap sync I/O)
    try:
        if final_df is None: # Should not happen if scores_df existed, but check
             return {"error_message": "Final DataFrame is None, cannot save."}

        logger.info(f"--- Attempting to save results for {len(final_df)} creators...")
        await asyncio.to_thread(evaluation_store.save_results, final_df, cluster_meta)
        logger.info("--- Results saved successfully.")
        # Store confirmation in state instead of the potentially large DataFrame
        return {"final_evaluation_results": "saved"}
    except Exception as e:
        logger.error(f"--- Failed to save results: {e}", exc_info=True)
        return {"error_message": f"Failed to save results: {e}"}


async def handle_error(state: Agent4State) -> dict:
    """Logs errors."""
    error_msg = state.get('error_message', 'Unknown error')
    logger.error(f"--- [Node: handle_error] Workflow failed: {error_msg}")
    return {} # Terminal node


# --- GRAPH EDGES ---
def should_continue(state: Agent4State) -> str:
    """Checks for errors before proceeding."""
    if state.get("error_message"):
        # Log the specific error before routing
        logger.warning(f"--- Stopping workflow at node '{state.get('_node_', 'unknown')}' due to error: {state['error_message']}")
        return "handle_error"
    return "continue"

# --- ASSEMBLE THE GRAPH ---
logger.info("--- Assembling Agent 4 graph ---")
workflow = StateGraph(Agent4State)

# Add Nodes
workflow.add_node("load_dcprs", load_dcprs)
workflow.add_node("cluster_creators", cluster_creators)
workflow.add_node("calculate_scores", calculate_scores)
workflow.add_node("combine_and_save_results", combine_and_save_results)
workflow.add_node("handle_error", handle_error)

# Define Workflow Edges (Sequential for now)
workflow.set_entry_point("load_dcprs")

workflow.add_conditional_edges(
    "load_dcprs",
    should_continue,
    {"continue": "cluster_creators", "handle_error": "handle_error"}
)
workflow.add_conditional_edges(
    "cluster_creators",
    should_continue, # Even if clustering skipped, continue to scoring
    {"continue": "calculate_scores", "handle_error": "handle_error"}
)
workflow.add_conditional_edges(
    "calculate_scores",
    should_continue,
    {"continue": "combine_and_save_results", "handle_error": "handle_error"}
)
workflow.add_conditional_edges(
    "combine_and_save_results",
    should_continue,
    {"continue": END, "handle_error": "handle_error"}
)

workflow.add_edge("handle_error", END)

# Compile the graph
try:
    app = workflow.compile()
    logger.info("--- Agent 4 graph compiled successfully ---")
except Exception as e:
    logger.exception("--- FATAL: Failed to compile Agent 4 graph ---")
    sys.exit(1)


# --- Main Execution Function (used by run_agent4.py) ---
# This function must exist for the import in run_agent4.py to work
async def run_evaluation():
    """Executes the Agent 4 evaluation graph."""
    logger.info("=========================================")
    logger.info("STARTING AGENT 4: EVALUATION WORKFLOW")
    logger.info("=========================================")

    initial_state: Agent4State = {
        "raw_dcprs": [],
        "clustering_results": None,
        "clustering_metadata": {},
        "scoring_results": None,
        "final_evaluation_results": None, # Storing path/status instead of full df
        "error_message": None
    }

    try:
        final_state = await app.ainvoke(initial_state, {"recursion_limit": 5}) # Add recursion limit

        if final_state.get("error_message"):
            logger.error(f"Agent 4 workflow finished with an error: {final_state['error_message']}")
        else:
            logger.info("Agent 4 workflow finished successfully.")
            final_results_status = final_state.get("final_evaluation_results")
            if final_results_status == "saved":
                 logger.info("Evaluation results were saved.")
                 # Optionally load and show head:
                 # latest_df = evaluation_store.load_latest_results()
                 # if latest_df is not None: logger.info(f"Results head:\n{latest_df.head()}")

    except Exception as e:
        logger.exception(f"Unhandled exception during Agent 4 graph execution: {e}")

    logger.info("=========================================")
    logger.info("AGENT 4 WORKFLOW COMPLETE")
    logger.info("=========================================")

# --- Direct Execution Block (for testing agent4_graph.py directly) ---
if __name__ == "__main__":
    logger.info("--- Running agent4_graph.py directly for testing ---")
    # Ensure necessary directories exist relative to this script file
    # This might be redundant if initialization already does it, but safe to include
    if 'output_dir_absolute' in locals() and output_dir_absolute:
         output_dir_absolute.mkdir(parents=True, exist_ok=True)
         logger.info(f"Direct run: Ensured output directory exists: {output_dir_absolute}")
    else:
         logger.error("Direct run: Could not determine output directory.")

    if 'log_dir_absolute' in locals() and log_dir_absolute:
         log_dir_absolute.mkdir(parents=True, exist_ok=True)
         logger.info(f"Direct run: Ensured log directory exists: {log_dir_absolute}")
    else:
         logger.error("Direct run: Could not determine log directory.")

    asyncio.run(run_evaluation())
    logger.info("--- Direct execution finished ---")

