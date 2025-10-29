import logging
import asyncio
import os
from datetime import datetime
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import pandas as pd

# Load .env file
load_dotenv()

# --- IMPORT THE LANGGRAPH APP and State ---
try:
    from agent4_graph import app as langgraph_app, Agent4State
except ImportError:
    print("Error: Could not import langgraph_app from agent4_graph.py.")
    langgraph_app = None
    Agent4State = None # type: ignore

# --- Import memory store for reading results ---
try:
     from memory.store import EvaluationStore
except ImportError:
     print("Error: Could not import EvaluationStore from memory.store")
     EvaluationStore = None # type: ignore

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Agent 4 - Creator Evaluator",
    description="Clusters creators and calculates quality/performance scores using LangGraph.",
    version="1.0.0"
)

# Initialize components needed by API
if EvaluationStore:
     # Use path from config if possible, else default
     # This requires loading config here too, or passing path explicitly
     # For simplicity, using env var or default directly.
     output_dir = os.getenv("AGENT4_OUTPUT_DIR", "./data/evaluation")
     evaluation_store = EvaluationStore(output_dir=output_dir)
else:
     evaluation_store = None


# --- Request/Response Models ---
class TriggerEvaluationResponse(BaseModel):
    status: str
    message: str
    job_id: Optional[str] = None # For potential background task tracking

class EvaluationResultRecord(BaseModel):
    creator_id: str
    cluster_id: int
    quality_score: float
    performance_score: float
    persona_version: Optional[str] = None
    total_posts_analyzed: Optional[int] = None
    # Add other fields from the output CSV as needed

class GetEvaluationResponse(BaseModel):
    results: List[EvaluationResultRecord]
    total_creators: int
    timestamp: Optional[str] = None # When the results were generated

# --- API Endpoints ---

@app.post("/trigger_evaluation", response_model=TriggerEvaluationResponse, status_code=202) # 202 Accepted for background tasks
async def trigger_evaluation(background_tasks: BackgroundTasks):
    """
    Triggers the Agent 4 evaluation workflow asynchronously.
    """
    if langgraph_app is None or Agent4State is None:
         raise HTTPException(status_code=500, detail="LangGraph application not loaded.")

    logger.info("Received request to trigger evaluation workflow.")

    # Define the initial state for the graph run
    initial_state: Agent4State = {
        "raw_dcprs": [],
        "clustering_results": None,
        "clustering_metadata": {},
        "scoring_results": None,
        "final_evaluation_results": None,
        "error_message": None
    }

    async def run_graph_task():
        logger.info("Background task started: Running Agent 4 graph...")
        try:
            # Invoke the graph
            final_state = await langgraph_app.ainvoke(initial_state)
            if final_state.get("error_message"):
                 logger.error(f"Agent 4 graph execution failed in background: {final_state['error_message']}")
            else:
                 logger.info("Agent 4 graph execution completed successfully in background.")
        except Exception as e:
            logger.exception(f"Unhandled exception during background graph execution: {e}")

    # Add the graph execution as a background task
    background_tasks.add_task(run_graph_task)

    # Return immediately
    return TriggerEvaluationResponse(
        status="triggered",
        message="Agent 4 evaluation workflow started in the background."
    )


@app.get("/evaluation_results", response_model=GetEvaluationResponse)
async def get_evaluation_results():
    """
    Retrieves the latest evaluation results (clustering and scores).
    """
    if evaluation_store is None:
        raise HTTPException(status_code=500, detail="EvaluationStore not initialized.")

    logger.info("Request received for latest evaluation results.")
    try:
        latest_results_df = evaluation_store.load_latest_results()

        if latest_results_df is None:
            raise HTTPException(status_code=404, detail="No evaluation results found. Trigger the evaluation first.")

        # Convert DataFrame to list of Pydantic models for response
        # Ensure columns match EvaluationResultRecord fields, handle missing cols
        latest_results_df.rename(columns={'meta_timestamp': 'timestamp'}, inplace=True, errors='ignore') # Example rename if needed
        results_list = latest_results_df.to_dict(orient='records')

        # Basic validation/conversion before creating Pydantic models
        validated_results = []
        for record in results_list:
             # Ensure required fields have default if missing before creating model
             record['persona_version'] = record.get('persona_version', None)
             record['total_posts_analyzed'] = record.get('total_posts_analyzed', None)
             try:
                validated_results.append(EvaluationResultRecord(**record))
             except Exception as pydantic_error:
                 logger.warning(f"Skipping record due to validation error: {pydantic_error}. Record: {record}")


        # Try to get timestamp from metadata if saved, else file mod time
        timestamp = None
        latest_filepath = evaluation_store.output_dir / "creator_evaluation_latest.csv"
        if latest_filepath.exists():
             timestamp = datetime.fromtimestamp(latest_filepath.stat().st_mtime).isoformat() + "Z"

        return GetEvaluationResponse(
            results=validated_results,
            total_creators=len(validated_results),
            timestamp=timestamp
        )

    except HTTPException as http_exc:
         raise http_exc
    except Exception as e:
        logger.exception(f"Error retrieving evaluation results: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")


# --- Health check (Optional, similar to Agent 3) ---
@app.get("/health", status_code=200)
async def health_check():
     # Simple health check
     # Could add checks for directory access, etc.
     return {"status": "healthy", "timestamp": datetime.utcnow().isoformat() + "Z"}


# --- Run FastAPI Server ---
if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8004")) # Different port for Agent 4
    logger.info(f"Starting Agent 4 FastAPI server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)
