import logging
import os
import sys
import json
import time
from pathlib import Path
from multiprocessing import Pool, Manager, cpu_count
from typing import List, Tuple, Optional, Dict, Any
import yaml
from tqdm import tqdm
from dotenv import load_dotenv
import asyncio
from pydantic import ValidationError

# --- Setup logging early ---
# Load .env relative to this file's location first
agent2_base_dir = Path(__file__).parent.resolve()
dotenv_path = agent2_base_dir / '.env'
load_dotenv(dotenv_path=dotenv_path)

# Configure logging
log_dir = agent2_base_dir / os.getenv("AGENT2_LOG_DIR", "logs")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / "agent2.log"
log_level = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(processName)s - %(name)s [%(levelname)s] - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler(sys.stdout) # To console
    ]
)
logger = logging.getLogger(__name__)
logger.info("--- Agent 2 Processor Starting ---")

# --- Import project modules after logging is set up ---
try:
    from modules.analysis import analyze_post_with_gemini
    from modules.embedding import initialize_embedding_model, generate_embedding, get_text_for_embedding
    from modules.schemas import FinalOutputSchema, FinalOutputMetadata, EngagementFeatures # Import necessary schemas
    logger.info("Project modules imported successfully.")
except ImportError as e:
    logger.exception(f"FATAL: Failed to import project modules: {e}")
    sys.exit(1)


# --- Load Configuration ---
CONFIG_PATH = agent2_base_dir / "config.yaml"
config = {}
try:
    logger.info(f"Loading configuration from {CONFIG_PATH}")
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
    logger.info("Configuration loaded successfully.")
except FileNotFoundError:
    logger.error(f"FATAL: Configuration file not found at {CONFIG_PATH}. Exiting.")
    sys.exit(1)
except Exception as e:
    logger.exception(f"FATAL: Error loading configuration: {e}")
    sys.exit(1)

# --- Global Config Variables ---
INPUT_BASE_DIR = Path(config.get("paths", {}).get("input_base_dir", "../Agent1/Outputs"))
OUTPUT_BASE_DIR = Path(config.get("paths", {}).get("output_base_dir", "../agent3_draft/data/creators"))
MAX_WORKERS = config.get("processing", {}).get("max_workers", 0)
DEBUG_MAX_POSTS = config.get("processing", {}).get("debug_max_posts", 0)

# --- Ensure absolute paths ---
if not INPUT_BASE_DIR.is_absolute():
    INPUT_BASE_DIR = (agent2_base_dir / INPUT_BASE_DIR).resolve()
if not OUTPUT_BASE_DIR.is_absolute():
     OUTPUT_BASE_DIR = (agent2_base_dir / OUTPUT_BASE_DIR).resolve()

logger.info(f"Input base directory: {INPUT_BASE_DIR}")
logger.info(f"Output base directory: {OUTPUT_BASE_DIR}")


# --- Core Processing Function (executed by worker processes) ---
def process_post_folder(folder_path_str: str) -> Tuple[str, bool, Optional[str]]:
    """
    Processes a single post folder: Load, Analyze, Embed, Save.
    Returns (folder_path, success_status, error_message).
    """
    folder_path = Path(folder_path_str)
    post_id = folder_path.name
    creator_id = folder_path.parent.name
    log_prefix = f"[{creator_id}/{post_id}]"
    logger.info(f"{log_prefix} Starting processing.")

    try:
        # 1. Load Input
        metadata_path = folder_path / "metadata.json"
        media_file = None
        media_ext = None

        if not metadata_path.exists():
            return folder_path_str, False, "metadata.json not found"

        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        # Find media file (handle common extensions)
        supported_image_exts = ['.jpg', '.jpeg', '.png', '.webp']
        supported_video_exts = ['.mp4', '.mov', '.avi', '.mkv'] # Add more as needed
        found_media = False
        for item in folder_path.iterdir():
            ext = item.suffix.lower()
            if ext in supported_image_exts or ext in supported_video_exts:
                media_file = item
                media_ext = ext
                found_media = True
                break
        if not found_media:
            return folder_path_str, False, "Media file (image/video) not found"

        is_reel = media_ext in supported_video_exts
        media_type = "reel" if is_reel else "post"
        logger.info(f"{log_prefix} Found media: {media_file.name}, Type: {media_type}")

        # --- Check if output already exists (optional skip) ---
        output_dir = OUTPUT_BASE_DIR / creator_id
        output_path = output_dir / f"{post_id}.json"
        # Uncomment below to skip already processed files
        if output_path.exists():
            logger.info(f"{log_prefix} Output JSON already exists. Skipping.")
            return folder_path_str, True, "Skipped (already exists)"


        # 2. Multimodal Analysis (Gemini)
        logger.info(f"{log_prefix} Starting Gemini analysis...")
        # Run analyze_post_with_gemini asynchronously in the worker's event loop
        loop = asyncio.get_event_loop()
        analysis_result = loop.run_until_complete(
             analyze_post_with_gemini(media_file, metadata, is_reel)
        )
        # analysis_result = asyncio.run(analyze_post_with_gemini(media_file, metadata, is_reel)) # Simpler if no other async needed


        if analysis_result is None:
            # Error logged within analyze_post_with_gemini
            return folder_path_str, False, "Gemini analysis failed after retries"
        logger.info(f"{log_prefix} Gemini analysis successful.")


        # 3. Generate Text for Embedding
        logger.info(f"{log_prefix} Generating text for embedding...")
        text_to_embed = get_text_for_embedding(analysis_result, metadata)
        if not text_to_embed:
             return folder_path_str, False, "Failed to construct text for embedding"

        # 4. Generate Embedding (Gemma/Fallback)
        logger.info(f"{log_prefix} Generating embedding...")
        embedding = generate_embedding(text_to_embed)
        if embedding is None:
            return folder_path_str, False, "Embedding generation failed"
        logger.info(f"{log_prefix} Embedding generated successfully.")


        # 5. Final JSON Assembly
        logger.info(f"{log_prefix} Assembling final JSON...")
        # Extract necessary metadata fields
        original_metadata = {
            "creator_id": creator_id,
            "post_id": post_id,
            "upload_time": metadata.get("upload_time"),
            "caption": metadata.get("caption"),
            "hashtags": metadata.get("hashtags"),
            "likes": metadata.get("likes"),
            "comments": metadata.get("comments"),
            "views": metadata.get("views"),
            "shares": metadata.get("shares"),
            "saves": metadata.get("saves"),
            "duration_seconds": metadata.get("duration_seconds"),
            "media_type": media_type,
            "is_promotional": metadata.get("is_promotional"),
            "geo_tag": metadata.get("geo_tag"),
            "creator_follower_count": metadata.get("followers") # Assuming Agent 1 adds this
        }

        # Calculate engagement rate
        likes = original_metadata.get("likes", 0) or 0
        comments = original_metadata.get("comments", 0) or 0
        views = original_metadata.get("views") # Could be None or 0
        followers = original_metadata.get("creator_follower_count") # Could be None or 0
        engagement_rate = None

        if views and views > 0:
            engagement_rate = (likes + comments) / views
        elif followers and followers > 0:
             engagement_rate = (likes + comments) / followers
             logger.debug(f"{log_prefix} Calculating engagement rate based on followers.")
        else:
            logger.warning(f"{log_prefix} Cannot calculate engagement rate (views={views}, followers={followers}). Setting to 0.0.")
            engagement_rate = 0.0

        # Assemble using Pydantic model for structure (optional but good practice)
        try:
            final_output = FinalOutputSchema(
                # Top level convenience fields
                post_id=post_id,
                creator_id=creator_id,
                # Metadata block
                metadata=FinalOutputMetadata(**original_metadata),
                # Analysis blocks (handle potential missing keys gracefully)
                content_overview=analysis_result.get("content_overview"),
                visual_analysis=analysis_result.get("visual_analysis"),
                audio_analysis=analysis_result.get("audio_analysis"), # Will be None for posts
                linguistic_analysis=analysis_result.get("linguistic_analysis"),
                timeline_analysis=analysis_result.get("timeline_analysis"), # Will be None for posts
                # Embedding and Engagement
                embedding=embedding,
                engagement_features=EngagementFeatures(
                    engagement_rate=engagement_rate,
                    comment_sentiment=analysis_result.get("comment_sentiment_estimated")
                )
            )
            final_json_dict = final_output.model_dump(exclude_none=True) # Exclude None fields for cleaner JSON
        except ValidationError as e:
             logger.error(f"{log_prefix} Failed to assemble final JSON due to Pydantic validation: {e}")
             return folder_path_str, False, f"Final JSON assembly validation error: {e}"


        # 6. Save Output
        logger.info(f"{log_prefix} Saving output JSON...")
        output_dir = OUTPUT_BASE_DIR / creator_id
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{post_id}.json"

        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(final_json_dict, f, indent=2, ensure_ascii=False)
            logger.info(f"{log_prefix} Successfully saved output to {output_path}")
            return folder_path_str, True, None # Success
        except IOError as e:
            logger.error(f"{log_prefix} Failed to write output JSON to {output_path}: {e}", exc_info=True)
            return folder_path_str, False, f"File write error: {e}"

    except Exception as e:
        logger.exception(f"{log_prefix} UNHANDLED EXCEPTION during processing: {e}")
        return folder_path_str, False, f"Unhandled exception: {e}"


# --- Main Orchestration ---
def main():
    """Finds post folders and processes them in parallel."""
    logger.info("--- Starting Main Orchestration ---")
    start_time = time.time()

    # --- Find all post folders ---
    post_folders = []
    if not INPUT_BASE_DIR.exists():
        logger.error(f"Input directory not found: {INPUT_BASE_DIR}")
        sys.exit(1)

    logger.info(f"Scanning for creator folders in {INPUT_BASE_DIR}...")
    for creator_dir in INPUT_BASE_DIR.iterdir():
        if creator_dir.is_dir():
            logger.debug(f"Scanning creator: {creator_dir.name}")
            for post_dir in creator_dir.iterdir():
                if post_dir.is_dir(): # Basic check, could add more validation
                    post_folders.append(str(post_dir.resolve())) # Pass absolute path string

    if not post_folders:
        logger.warning("No post folders found to process.")
        sys.exit(0)

    logger.info(f"Found {len(post_folders)} post folders to process.")

    # Apply debug limit if set
    if DEBUG_MAX_POSTS and DEBUG_MAX_POSTS > 0:
        logger.warning(f"DEBUG: Limiting processing to {DEBUG_MAX_POSTS} posts.")
        post_folders = post_folders[:DEBUG_MAX_POSTS]

    # --- Setup Multiprocessing Pool ---
    num_workers = MAX_WORKERS if MAX_WORKERS > 0 else cpu_count()
    logger.info(f"Initializing multiprocessing pool with {num_workers} workers.")

    # Use Manager for shared state if needed (e.g., counters), though tqdm handles progress well.
    # manager = Manager()
    # results_list = manager.list() # Example if collecting detailed results

    try:
        # Use initializer to load the embedding model once per worker process
        with Pool(processes=num_workers, initializer=initialize_embedding_model) as pool:
            results = []
            # Use imap_unordered for potentially better performance and memory usage
            # Wrap with tqdm for progress bar
            with tqdm(total=len(post_folders), desc="Processing Posts") as pbar:
                for result in pool.imap_unordered(process_post_folder, post_folders):
                    results.append(result)
                    pbar.update(1)

        # --- Process Results ---
        success_count = 0
        failed_posts = []
        for path, success, error_msg in results:
            if success:
                success_count += 1
            else:
                failed_posts.append((Path(path).name, error_msg))

        end_time = time.time()
        duration = end_time - start_time

        logger.info("--- Processing Complete ---")
        logger.info(f"Total time: {duration:.2f} seconds")
        logger.info(f"Processed: {len(results)} posts")
        logger.info(f"Successful: {success_count}")
        logger.info(f"Failed: {len(failed_posts)}")
        if failed_posts:
            logger.warning("Failed posts:")
            for post_id, reason in failed_posts:
                logger.warning(f"  - {post_id}: {reason}")

    except Exception as e:
         logger.exception(f"An error occurred during multiprocessing: {e}")
    finally:
        logger.info("--- Agent 2 Processor Finished ---")


if __name__ == "__main__":
    # Ensure output base directory exists before starting
    OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Ensured output base directory exists: {OUTPUT_BASE_DIR}")
    main()
