import asyncio
import logging # Keep import for logger usage
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load .env variables first - may be needed for paths
load_dotenv()

# Get the root logger (configuration will happen in agent4_graph.py upon import)
logger = logging.getLogger() # Get root logger initially

async def main():
    """Imports and runs the Agent 4 graph execution function."""
    global logger # Allow updating logger reference if agent4_graph reconfigures it
    try:
        # --- Import Attempt ---
        print("Attempting to import run_evaluation from agent4_graph...") # Use print for pre-logging setup
        # Logging inside agent4_graph should be set up during this import
        from agent4_graph import run_evaluation, logger as graph_logger
        logger = graph_logger # Use the logger configured by the graph file
        logger.info("Import successful. Starting run_evaluation...")
        await run_evaluation()
        logger.info("run_evaluation completed.")

    except ImportError as e:
        # This error likely means agent4_graph.py failed during its own execution/import phase
        print(f"FATAL: Failed to import run_evaluation from agent4_graph.py: {e}") # Use print
        print("Check agent4_graph.py for syntax errors or errors during its module imports/initializations.")
        # Attempt to get logger if it exists, otherwise use print
        try:
            logger.error("Failed to import run_evaluation from agent4_graph.py.")
            logger.error("Ensure agent4_graph.py exists and has no import errors.")
            logger.debug(f"ImportError details: {e}", exc_info=True)
        except NameError: # logger might not be defined if import failed early
             pass
        sys.exit(1) # Exit with error code

    except Exception as e:
        # Catch any other unexpected errors during graph execution
        logger.exception(f"An error occurred during Agent 4 execution: {e}")
        sys.exit(1) # Exit with error code

if __name__ == "__main__":
    # Get the directory where run_agent4.py is located
    agent4_base_dir = Path(__file__).parent.resolve()
    print(f"run_agent4.py base directory: {agent4_base_dir}")

    # Ensure necessary output directories exist RELATIVE TO AGENT4's LOCATION
    # These paths should ideally align perfectly with config.yaml
    output_dir_relative = os.getenv("AGENT4_OUTPUT_DIR", "./data/evaluation")
    output_dir_absolute = agent4_base_dir / output_dir_relative
    output_dir_absolute.mkdir(parents=True, exist_ok=True)
    print(f"Ensured output directory exists: {output_dir_absolute}")

    log_dir_relative = os.getenv("AGENT4_LOGS_DIR", "./logs")
    log_dir_absolute = agent4_base_dir / log_dir_relative
    log_dir_absolute.mkdir(parents=True, exist_ok=True)
    print(f"Ensured log directory exists: {log_dir_absolute}")

    print("Starting Agent 4 execution via run_agent4.py...")
    # Change CWD to Agent4 directory temporarily for relative paths inside modules
    original_cwd = Path.cwd()
    os.chdir(agent4_base_dir)
    print(f"Changed CWD to: {agent4_base_dir}")

    try:
        asyncio.run(main())
        print("Agent 4 execution finished.") # Use print
        # Use logger if available, otherwise ignore
        try:
             logger.info("Agent 4 execution finished.")
        except NameError:
             pass

    finally:
        # Change back to original CWD
        os.chdir(original_cwd)
        print(f"Restored CWD to: {original_cwd}")

