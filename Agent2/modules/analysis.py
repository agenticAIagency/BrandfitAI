import google.generativeai as genai
import google.ai.generativelanguage as glm # For File API if needed
import os
import logging
import json
from pathlib import Path
from pydantic import ValidationError
from typing import Dict, Any, Optional, Tuple, Union
import time # For File API polling
import asyncio # For running sync SDK call in thread

from .schemas import PostAnalysisResult, ReelAnalysisResult # Import Pydantic models

# --- Configuration --- (Ideally load from a shared config object/dict)
# It's better practice to pass config values or load them here than rely solely on env vars
# For simplicity now, using environment variables directly
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-flash-latest")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MAX_RETRIES = int(os.getenv("GEMINI_MAX_RETRIES", "2"))
REQUEST_TIMEOUT = int(os.getenv("GEMINI_REQUEST_TIMEOUT", "300"))

logger = logging.getLogger(__name__)

# --- Configure Gemini Client ---
genai_model = None
if not GEMINI_API_KEY:
    logger.error("GEMINI_API_KEY environment variable not set. Cannot initialize Gemini client.")
    # Consider raising an error to stop agent2_processor.py from starting
    # raise ValueError("GEMINI_API_KEY must be set in the environment.")
else:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        # TODO: Add specific generation config if needed (temperature, safety settings)
        genai_model = genai.GenerativeModel(GEMINI_MODEL_NAME)
        logger.info(f"Gemini client configured for model: {GEMINI_MODEL_NAME}")
    except Exception as e:
        logger.exception(f"FATAL: Failed to configure Gemini client: {e}")
        # raise # Re-raise to stop startup


# --- Helper to upload file for Gemini ---
# Reference: https://ai.google.dev/tutorials/python_quickstart#file_api
def upload_media_if_needed(media_path: Path) -> Optional[glm.File]:
    """Uploads media file to Gemini File API and waits for it to be ready."""
    if not media_path.exists():
        logger.error(f"Media file not found: {media_path}")
        return None
    logger.info(f"Uploading media file: {media_path}...")
    try:
        # Check file size (File API might have limits, or different methods needed for huge files)
        file_size = media_path.stat().st_size
        logger.info(f"Media file size: {file_size / (1024*1024):.2f} MB")
        # Add size check logic if needed based on Gemini API limits

        media_file = genai.upload_file(path=media_path)
        logger.info(f"Started upload for '{media_file.display_name}' as: {media_file.uri}")

        # Wait for the file to be processed
        attempts = 0
        max_attempts = 15 # Wait up to ~ 1.5 min (adjust as needed)
        sleep_time = 6
        while media_file.state.name == "PROCESSING" and attempts < max_attempts:
            attempts += 1
            logger.info(f"Waiting for media processing... (Attempt {attempts}/{max_attempts}) URI: {media_file.uri}")
            time.sleep(sleep_time)
            try:
                media_file = genai.get_file(media_file.name) # Refresh status
            except Exception as get_err:
                 logger.warning(f"Error checking file status {media_file.name}: {get_err}. Retrying check.")
                 # Add a small delay before retrying get_file
                 time.sleep(2)
                 if attempts < max_attempts -1: # Avoid sleep on last check attempt
                     continue
                 else:
                     logger.error(f"Could not get file status for {media_file.name} after multiple checks.")
                     raise # Re-raise the exception


        if media_file.state.name == "FAILED":
            logger.error(f"Media file processing failed: {media_file.uri}. Reason: {media_file.error if hasattr(media_file, 'error') else 'Unknown'}")
            # Consider deleting the failed file to avoid costs/clutter
            try:
                genai.delete_file(media_file.name)
                logger.info(f"Deleted failed media file: {media_file.name}")
            except Exception as del_err:
                logger.warning(f"Could not delete failed media file {media_file.name}: {del_err}")
            return None
        elif media_file.state.name != "ACTIVE":
             logger.error(f"Media file did not become active. Final state: {media_file.state.name}. URI: {media_file.uri}")
             # Consider deleting the inactive file
             try:
                genai.delete_file(media_file.name)
                logger.info(f"Deleted inactive media file: {media_file.name}")
             except Exception as del_err:
                logger.warning(f"Could not delete inactive media file {media_file.name}: {del_err}")
             return None

        logger.info(f"Media file is active and ready: {media_file.uri}")
        return media_file
    except Exception as e:
        logger.error(f"Error uploading or processing media file {media_path}: {e}", exc_info=True)
        return None


# --- Core Analysis Function ---
async def analyze_post_with_gemini(
    media_path: Path,
    metadata: Dict[str, Any],
    is_reel: bool
) -> Optional[Dict[str, Any]]:
    """
    Analyzes a post/reel using Gemini, performs validation, and handles retries.

    Args:
        media_path: Path to the image or video file.
        metadata: Dictionary containing metadata (caption, etc.).
        is_reel: Boolean indicating if it's a video reel.

    Returns:
        Validated analysis dictionary or None if analysis fails.
    """
    if not genai_model:
        logger.error("Gemini client not initialized. Cannot perform analysis.")
        return None

    # --- 1. Construct the Prompt ---
    caption = metadata.get("caption", "")
    prompt_parts = []
    target_schema = None # Define target schema

    # System Instruction / Role
    prompt_parts.append("You are an expert social media analyst specializing in Instagram content.")
    prompt_parts.append(f"Analyze the provided {'reel (video)' if is_reel else 'post (image)'} and its caption.")

    # Specific Instructions based on type
    if is_reel:
        prompt_parts.append(
            "Provide a detailed analysis including these MANDATORY fields:\n"
            "- content_overview (object): Contains primary_theme (string, required), secondary_theme (string, optional), mood (string, optional), tone (string, required), audience_emotion_targeted (list of strings, optional), setting (string, optional), brand_presence (boolean, optional).\n"
            "- timeline_analysis (array): Break video into logical segments. For each segment (object): timestamp_start (float, required), timestamp_end (float, required), scene_summary (string, required), speech_summary (string, optional), objects_detected (list of strings, optional), actions (list of strings, optional), facial_expression (string, optional), tone (string, optional), audio_energy (float 0-1, optional), visual_intensity (float 0-1, optional), engagement_likelihood (string 'Low'/'Medium'/'High', optional).\n"
            "- visual_analysis (object): Contains camera_style (string, optional), lighting (string, optional), motion_density (float 0-1, optional), scene_changes (integer, optional), dominant_colors (list of strings, optional), composition_type (string, optional), objects_detected (list of strings, optional), focus_subject (string, optional), aesthetic_score (float 0.0-10.0, required).\n"
            "- audio_analysis (object): Contains music_type (string, optional), speech_style (string, optional), sound_effects_present (boolean, optional).\n"
            "- linguistic_analysis (object): Contains caption_tone (string, optional), emotion_keywords (list of strings, optional), hashtag_intent (list of strings, optional), text_overlay_presence (boolean, optional), speech_summary_overall (string, optional).\n"
            "- comment_sentiment_estimated (string, optional): Your estimate ('Positive', 'Neutral', or 'Negative') based on the content."
        )
        target_schema = ReelAnalysisResult # Pydantic model for validation
    else: # Image Post
        prompt_parts.append(
             "Provide a detailed analysis including these MANDATORY fields:\n"
            "- content_overview (object): Contains primary_theme (string, required), secondary_theme (string, optional), mood (string, optional), tone (string, required), audience_emotion_targeted (list of strings, optional), setting (string, optional), brand_presence (boolean, optional).\n"
            "- visual_analysis (object): Contains dominant_colors (list of strings, optional), composition_type (string, optional), objects_detected (list of strings, optional), focus_subject (string, optional), aesthetic_score (float 0.0-10.0, required).\n"
            "- linguistic_analysis (object): Contains caption_tone (string, optional), emotion_keywords (list of strings, optional), hashtag_intent (list of strings, optional), text_overlay_presence (boolean, optional).\n"
            "- image_style (string, optional): Specific style if identifiable.\n"
            "- comment_sentiment_estimated (string, optional): Your estimate ('Positive', 'Neutral', or 'Negative') based on the content."
        )
        target_schema = PostAnalysisResult # Pydantic model for validation

    prompt_parts.append(f"\nCaption provided for context: '{caption}'")
    # Crucial instruction for JSON output
    prompt_parts.append("\nOutput ONLY a single, valid JSON object strictly matching the requested structure with all mandatory fields. Do not include markdown formatting like ```json or ```, explanations, or apologies.")

    final_prompt = "\n".join(prompt_parts)
    # logger.debug(f"Final Gemini Prompt:\n{final_prompt}") # Log prompt if needed

    # --- 2. Upload Media ---
    uploaded_file = None
    try:
        uploaded_file = upload_media_if_needed(media_path)
        if not uploaded_file:
            return None # Upload failed, error logged in helper
    except Exception as upload_err:
        logger.error(f"Critical error during media upload setup: {upload_err}", exc_info=True)
        return None # Cannot proceed without media

    # --- 3. Gemini API Call with Retries ---
    analysis_json: Optional[Dict[str, Any]] = None # Explicitly define type
    last_error: Optional[str] = None
    prompt_for_api = final_prompt # Store original prompt

    for attempt in range(MAX_RETRIES + 1):
        try:
            logger.info(f"Calling Gemini ({GEMINI_MODEL_NAME}) - Attempt {attempt + 1}/{MAX_RETRIES + 1}")
            content_parts = [prompt_for_api, uploaded_file]

            response = await asyncio.to_thread(
                genai_model.generate_content,
                contents=content_parts,
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json"
                    # Consider adding temperature=0.1 or similar for deterministic analysis
                ),
                 request_options={"timeout": REQUEST_TIMEOUT} # Add timeout
            )

            # --- 4. Validation ---
            # Check for empty or blocked response
            if not response.parts:
                 prompt_feedback = response.prompt_feedback if hasattr(response, 'prompt_feedback') else 'No feedback available.'
                 logger.warning(f"Attempt {attempt + 1}: Gemini returned an empty or blocked response. Feedback: {prompt_feedback}")
                 last_error = f"Gemini Safety Block or Empty Response. Feedback: {prompt_feedback}"
                 if attempt < MAX_RETRIES: await asyncio.sleep(2 ** attempt); continue # Retry
                 else: break # Failed after retries

            raw_response_text = response.text
            logger.debug(f"Raw Gemini Response (Attempt {attempt+1}):\n{raw_response_text[:500]}...") # Log beginning

            try:
                # Basic cleanup of potential markdown fences
                if raw_response_text.strip().startswith("```json"):
                    raw_response_text = raw_response_text.strip()[7:]
                if raw_response_text.strip().endswith("```"):
                     raw_response_text = raw_response_text.strip()[:-3]

                parsed_json = json.loads(raw_response_text)
                validated_data = target_schema.model_validate(parsed_json)
                analysis_json = validated_data.model_dump()
                logger.info("Gemini response parsed and validated successfully.")
                last_error = None
                break # Success

            except json.JSONDecodeError as e:
                logger.warning(f"Attempt {attempt + 1}: Failed to parse Gemini JSON response: {e}")
                last_error = f"JSON Parse Error: {e}\nResponse Snippet: {raw_response_text[:200]}..."
                if attempt < MAX_RETRIES:
                    prompt_for_api = final_prompt + "\nREMINDER: Output ONLY valid JSON."
                    await asyncio.sleep(2 ** attempt)

            except ValidationError as e:
                logger.warning(f"Attempt {attempt + 1}: Gemini response failed Pydantic validation")
                # Log specific validation errors
                for error in e.errors():
                    logger.warning(f"  - Field: {'.'.join(map(str, error['loc']))}, Error: {error['msg']}")

                last_error = f"Pydantic Validation Error: {e.errors()}\nResponse Snippet: {raw_response_text[:200]}..."
                if attempt < MAX_RETRIES:
                    prompt_for_api = final_prompt + "\nREMINDER: Ensure all required JSON fields are present and types are correct."
                    await asyncio.sleep(2 ** attempt)

        except Exception as e:
            # Catch broader API call errors (timeouts, connection issues, etc.)
            logger.error(f"Attempt {attempt + 1}: Gemini API call failed: {e}", exc_info=False) # Keep log concise
            logger.debug("Full exception:", exc_info=True) # Log full trace only in debug
            last_error = f"API Call Error: {type(e).__name__} - {e}"
            if attempt < MAX_RETRIES:
                await asyncio.sleep(2 ** attempt) # Exponential backoff

    # --- 5. Cleanup uploaded file ---
    # Ensure cleanup happens even if analysis fails after upload
    if uploaded_file:
        try:
            logger.info(f"Deleting uploaded media file: {uploaded_file.uri}")
            genai.delete_file(uploaded_file.name)
        except Exception as e:
            logger.warning(f"Could not delete uploaded file {uploaded_file.name}: {e}")
    else:
        logger.warning("No uploaded file object to delete (upload might have failed).")


    # --- 6. Handle Final Failure ---
    if analysis_json is None:
        logger.error(f"Analysis failed for {media_path.parent.name} after {MAX_RETRIES + 1} attempts.")
        if last_error:
            logger.error(f"Last error detail: {last_error}")
        return None # Indicate failure

    return analysis_json

