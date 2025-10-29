"""
Agent 3: Creator Persona Builder (REFACTORED for LangGraph)
Main FastAPI Application using LangGraph for core logic.
"""

import logging
import asyncio
from datetime import datetime
from typing import Optional, List, Dict, Any

import os
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load .env file for configuration if needed by modules during import
load_dotenv()

# --- IMPORT THE LANGGRAPH APP ---
# Assumes agent3_graph.py is in the same directory or accessible via Python path
try:
    from agent3_graph import app as langgraph_app, Agent3State
except ImportError:
    print("Error: Could not import langgraph_app from agent3_graph.py.")
    print("Ensure agent3_graph.py is in the same directory or Python path.")
    # Fallback if needed, or exit
    langgraph_app = None
    Agent3State = None # type: ignore

# Import only necessary modules for API layer (primarily memory store for GET/DELETE)
from memory.store import PersonaMemoryStore
from memory.versioning import VersionManager # Keep for version generation if needed outside graph

# Configure logging
# Consider using Loguru or structured logging for better FastAPI logs
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Agent 3 - Creator Persona Builder (LangGraph)",
    description="Builds DCPRs using a LangGraph workflow with retries and reflection.",
    version="3.0.0" # Version bump reflecting LangGraph integration
)

# Initialize components needed directly by API endpoints
memory_store = PersonaMemoryStore(storage_dir=os.getenv("STORAGE_DIR", "./data/personas"))
# version_manager = VersionManager() # Might not be needed if graph handles all versioning


# --- Request/Response Models ---
class BuildRequest(BaseModel):
    creator_id: str = Field(..., description="Unique creator identifier")
    force_rebuild: bool = Field(False, description="Force full rebuild even if latest stats/persona exist")

# Simplified Update Request - we just need the creator_id to trigger the graph
class UpdateRequest(BaseModel):
     creator_id: str = Field(..., description="Unique creator identifier")

class DCPRResponse(BaseModel):
    creator_id: str
    persona_version: str
    dcpr: Dict[str, Any]
    stats_saved: bool
    created_at: str
    operation: str # e.g., "full_build", "incremental_update", "cached", "error"

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    storage_stats: Dict[str, Any]


# --- API Endpoints ---

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check system health and storage statistics."""
    storage_stats = {}
    status = "healthy"
    try:
        storage_stats = memory_store.get_storage_stats()
    except Exception as e:
        logger.error(f"Failed to get storage stats: {e}")
        status = "degraded" # Or "unhealthy" if storage is critical

    return HealthResponse(
        status=status,
        timestamp=datetime.utcnow().isoformat() + "Z",
        storage_stats=storage_stats
    )


# Combined Build/Update Endpoint using LangGraph
@app.post("/build_or_update_persona", response_model=DCPRResponse)
async def build_or_update_persona(request: BuildRequest):
    """
    Triggers the LangGraph workflow to build or update a creator's DCPR.
    The graph automatically determines if a full build or incremental update is needed
    based on the presence of stats, unless force_rebuild is True.
    """
    if langgraph_app is None or Agent3State is None:
         raise HTTPException(status_code=500, detail="LangGraph application not loaded.")

    creator_id = request.creator_id
    force_rebuild = request.force_rebuild
    logger.info(f"LangGraph build/update requested for creator: {creator_id}. Force rebuild: {force_rebuild}")

    # --- Cache Check (Skip if forcing rebuild) ---
    if not force_rebuild:
        try:
            # Check if stats exist - if they do, an incremental update *might* happen
            stats_exist = memory_store.stats_exist(creator_id)
            if stats_exist:
                 # Optional: More sophisticated cache check - e.g., check if latest persona
                 # is very recent and there are likely no new posts yet.
                 # For now, we rely on the graph's "No new posts" exit condition.
                 pass # Let the graph handle the check

            # Check if a latest persona *already* exists
            if memory_store.persona_exists(creator_id):
                # If NOT forcing rebuild AND persona exists, we might return cached
                # But let's check if stats exist first to decide if an update attempt is warranted
                if not stats_exist: # Persona exists but no stats? Something is wrong, force rebuild logic
                     logger.warning(f"Persona exists for {creator_id} but stats missing. Forcing graph execution.")
                else:
                    # Both exist. The graph will check for new posts.
                    # We could add logic here to return cached immediately if the last update was very recent.
                    # For simplicity now, we let the graph run. It will exit quickly if no new posts.
                    pass

        except Exception as e:
             logger.warning(f"Error during cache check for {creator_id}: {e}. Proceeding with graph execution.")


    # --- Define Initial State for LangGraph ---
    # The graph nodes will populate most of these fields
    initial_state: Agent3State = { # Type hint for clarity
        "creator_id": creator_id,
        "all_posts": [],
        "new_posts": [],
        "old_stats": None, # Graph's check_for_stats node will load this if needed
        "final_dcpr": {},
        "final_stats": {},
        "version_id": "",
        "build_type": "full" if force_rebuild else "", # Let check_for_stats override if not forced
        "error_message": None,
        "reflection_attempts": 0
    }
    # If forcing rebuild, explicitly clear old_stats to ensure full build path
    if force_rebuild:
         initial_state["old_stats"] = None
         initial_state["build_type"] = "full"
         logger.info(f"Forcing full rebuild path for {creator_id}.")


    # --- Invoke LangGraph ---
    try:
        logger.info(f"Invoking LangGraph for {creator_id}...")
        # Use ainvoke for non-blocking call within FastAPI endpoint
        final_state = await langgraph_app.ainvoke(initial_state, {"recursion_limit": 5}) # Add recursion limit

        # Check for graph execution errors
        error_msg = final_state.get("error_message")
        if error_msg:
            # Handle specific known non-fatal errors differently if needed
            if "No new posts to process" in error_msg:
                 logger.info(f"Graph finished for {creator_id}: No new posts found for incremental update.")
                 # Load the *existing* latest persona to return it
                 existing_dcpr = memory_store.load_latest_dcpr(creator_id)
                 if existing_dcpr:
                      return DCPRResponse(
                          creator_id=creator_id,
                          persona_version=existing_dcpr.get("persona_version", "unknown"),
                          dcpr=existing_dcpr,
                          stats_saved=True,
                          created_at=existing_dcpr.get("metadata", {}).get("last_updated", ""),
                          operation="no_update_needed"
                      )
                 else: # Should not happen if stats existed, but handle defensively
                      raise HTTPException(status_code=404, detail="No new posts, and failed to load existing persona.")
            else:
                 # A real error occurred during the graph execution
                 logger.error(f"LangGraph execution failed for {creator_id}: {error_msg}")
                 raise HTTPException(
                     status_code=500,
                     detail=f"Persona generation failed: {error_msg}"
                 )

        # --- Success Case ---
        final_dcpr = final_state.get("final_dcpr")
        version_id = final_state.get("version_id")
        build_type = final_state.get("build_type", "unknown") # Get the actual build type executed

        if not final_dcpr or not version_id:
             logger.error(f"Graph finished for {creator_id} but final DCPR or version_id missing in state.")
             raise HTTPException(status_code=500, detail="Graph finished successfully but result state is incomplete.")

        logger.info(f"LangGraph execution complete for {creator_id}, version: {version_id}, type: {build_type}")

        return DCPRResponse(
            creator_id=creator_id,
            persona_version=version_id,
            dcpr=final_dcpr,
            stats_saved=True, # Reaching here means save was successful (or graph errored before)
            created_at=final_dcpr.get("metadata", {}).get("last_updated", ""),
            operation=build_type
        )

    except HTTPException as http_exc:
         raise http_exc # Re-raise FastAPI specific exceptions
    except Exception as e:
        # Catch any other unexpected errors during graph invocation
        logger.exception(f"Unhandled error invoking LangGraph for {creator_id}: {e}") # Log full traceback
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")


# --- Read Endpoints (Remain Largely Unchanged) ---

@app.get("/persona/{creator_id}", response_model=Dict[str, Any]) # Return the raw DCPR dict
async def get_persona(
    creator_id: str,
    version: str = "latest"
):
    """Retrieve the latest or a specific version of a creator's DCPR."""
    logger.info(f"Request received for persona: {creator_id}, version: {version}")
    try:
        if version == "latest":
            dcpr = memory_store.load_latest_dcpr(creator_id)
        else:
            # Add basic validation for version format if needed
            if not version.startswith("v"):
                 raise HTTPException(status_code=400, detail="Invalid version format. Must start with 'v'.")
            dcpr = memory_store.load_dcpr_by_version(creator_id, version)

        if not dcpr:
            logger.warning(f"DCPR not found for creator: {creator_id}, version: {version}")
            raise HTTPException(
                status_code=404,
                detail=f"DCPR not found for creator: {creator_id}, version: {version}"
            )

        logger.info(f"Successfully retrieved DCPR for {creator_id}, version: {version}")
        return dcpr # Return the dictionary directly

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.exception(f"Error retrieving persona for {creator_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error retrieving persona: {e}")


@app.get("/persona/{creator_id}/versions", response_model=Dict[str, Any])
async def get_persona_versions(creator_id: str):
    """Get all available version identifiers for a creator's DCPR."""
    logger.info(f"Request received for versions: {creator_id}")
    try:
        versions = memory_store.get_all_versions(creator_id)
        logger.info(f"Found {len(versions)} versions for {creator_id}")
        return {
            "creator_id": creator_id,
            "versions": versions,
            "total_versions": len(versions)
        }
    except Exception as e:
        logger.exception(f"Error retrieving versions for {creator_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error retrieving versions: {e}")


@app.delete("/persona/{creator_id}", status_code=200) # Use 200 or 204 No Content
async def delete_persona(
    creator_id: str,
    version: Optional[str] = None # Query parameter to delete specific version
):
    """Delete all or a specific version of a creator's persona and stats."""
    logger.info(f"Delete request received for: {creator_id}, version: {version or 'all'}")
    try:
        success = memory_store.delete_persona(creator_id, version)

        if not success:
            # Could be because it didn't exist or deletion failed
            logger.warning(f"Persona/version not found or deletion failed for {creator_id}, version: {version or 'all'}")
            raise HTTPException(
                status_code=404,
                detail=f"Persona or specified version not found for creator: {creator_id}"
            )

        logger.info(f"Successfully deleted persona data for {creator_id}, version: {version or 'all'}")
        return {
            "status": "deleted",
            "creator_id": creator_id,
            "version_deleted": version or "all"
        }

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.exception(f"Error deleting persona data for {creator_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error during deletion: {e}")


# --- Run FastAPI Server ---
if __name__ == "__main__":
    import uvicorn
    # Read host/port from environment variables or config if needed
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8003"))
    logger.info(f"Starting Agent 3 FastAPI server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)
