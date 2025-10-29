import logging
import numpy as np
from sentence_transformers import SentenceTransformer
import torch # Import torch to check for CUDA
import os
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

# --- Global variable to hold the loaded model ---
# This is safe for multiprocessing if using initializer
embedding_model_instance: Optional[SentenceTransformer] = None

# --- Configuration --- (Ideally load from a shared config object/dict)
PRIMARY_MODEL = os.getenv("EMBEDDING_PRIMARY_MODEL", "all-MiniLM-L6-v2")
FALLBACK_MODEL = os.getenv("EMBEDDING_FALLBACK_MODEL", "embeddinggemma-300m")
DEVICE = os.getenv("EMBEDDING_DEVICE", "auto")
NORMALIZE = os.getenv("EMBEDDING_NORMALIZE", "true").lower() == "true"

def initialize_embedding_model():
    """Loads the embedding model. Intended to be called once per worker process."""
    global embedding_model_instance
    if embedding_model_instance is not None:
        logger.info("Embedding model already loaded in this process.")
        return

    model_name_to_load = PRIMARY_MODEL
    logger.info(f"Attempting to load primary embedding model: {model_name_to_load}")

    # Determine device
    selected_device = None
    if DEVICE == "auto":
        if torch.cuda.is_available():
            selected_device = "cuda"
        elif torch.backends.mps.is_available(): # Check for Apple Silicon GPU
             selected_device = "mps"
        else:
            selected_device = "cpu"
        logger.info(f"Auto-selected device: {selected_device}")
    else:
        selected_device = DEVICE
        logger.info(f"Using specified device: {selected_device}")


    try:
        # Load the primary model
        # May need trust_remote_code=True depending on the Gemma model version/source
        embedding_model_instance = SentenceTransformer(model_name_to_load, device=selected_device, trust_remote_code=True)
        logger.info(f"Successfully loaded primary model '{model_name_to_load}' onto {selected_device}")
    except Exception as primary_error:
        logger.warning(f"Failed to load primary embedding model '{model_name_to_load}': {primary_error}")
        logger.info(f"Attempting to load fallback model: {FALLBACK_MODEL}")
        try:
            # Load the fallback model
            embedding_model_instance = SentenceTransformer(FALLBACK_MODEL, device=selected_device)
            logger.info(f"Successfully loaded fallback model '{FALLBACK_MODEL}' onto {selected_device}")
        except Exception as fallback_error:
            logger.error(f"Failed to load fallback embedding model '{FALLBACK_MODEL}': {fallback_error}", exc_info=True)
            # Cannot proceed without an embedding model
            raise RuntimeError("Could not load any embedding model.") from fallback_error


def generate_embedding(text: str) -> Optional[List[float]]:
    """Generates an embedding for the given text using the globally loaded model."""
    global embedding_model_instance
    if embedding_model_instance is None:
        logger.error("Embedding model is not initialized in this process.")
        # Attempt to initialize? Or rely on initializer. Let's rely for now.
        # initialize_embedding_model() # Could try this, but might cause issues
        # if embedding_model_instance is None: # Check again
        #      return None
        return None # Fail if not initialized by pool initializer

    try:
        logger.debug(f"Generating embedding for text: '{text[:100]}...'")
        # Ensure text is not empty
        if not text or not text.strip():
             logger.warning("Input text for embedding is empty. Returning None.")
             return None

        # Generate embedding
        embedding_vector = embedding_model_instance.encode(text, normalize_embeddings=NORMALIZE)

        # Ensure it's a list of floats
        if isinstance(embedding_vector, np.ndarray):
            embedding_list = embedding_vector.tolist()
        else: # Should already be list or tensor -> list
            embedding_list = list(embedding_vector)

        logger.debug(f"Generated embedding of dimension {len(embedding_list)}")
        return embedding_list

    except Exception as e:
        logger.error(f"Error generating embedding: {e}", exc_info=True)
        return None


def get_text_for_embedding(
    analysis_data: Dict[str, Any],
    metadata: Dict[str, Any]
) -> str:
    """Constructs a text string from analysis and metadata for embedding."""

    parts = []

    # Key metadata
    caption = metadata.get("caption")
    if caption:
        parts.append(f"Caption: {caption}")

    # Key analysis parts (adjust based on schema accuracy)
    if overview := analysis_data.get("content_overview"):
        parts.append(f"Theme: {overview.get('primary_theme', 'N/A')}")
        if secondary := overview.get('secondary_theme'): parts.append(f"Secondary Theme: {secondary}")
        parts.append(f"Tone: {overview.get('tone', 'N/A')}")
        if mood := overview.get('mood'): parts.append(f"Mood: {mood}")
        if setting := overview.get('setting'): parts.append(f"Setting: {setting}")

    if visual := analysis_data.get("visual_analysis"):
        if colors := visual.get('dominant_colors'): parts.append(f"Colors: {', '.join(colors[:3])}")
        if comp := visual.get('composition_type'): parts.append(f"Composition: {comp}")
        if objs := visual.get('objects_detected'): parts.append(f"Objects: {', '.join(objs[:5])}")
        if style := analysis_data.get('image_style'): parts.append(f"Image Style: {style}") # Post only
        if cam_style := visual.get('camera_style'): parts.append(f"Camera: {cam_style}") # Reel only

    if audio := analysis_data.get("audio_analysis"): # Reel only
        if music := audio.get('music_type'): parts.append(f"Music: {music}")
        if speech := audio.get('speech_style'): parts.append(f"Speech: {speech}")

    if ling := analysis_data.get("linguistic_analysis"):
        if cap_tone := ling.get('caption_tone'): parts.append(f"Caption Tone: {cap_tone}")
        if speech_sum := ling.get('speech_summary_overall'): parts.append(f"Speech Summary: {speech_sum[:100]}...") # Truncate

    # Timeline Summary (Reels) - Keep concise
    if timeline := analysis_data.get("timeline_analysis"):
        timeline_summary = "; ".join([seg.get('scene_summary', '') for seg in timeline[:3]]) # First 3 scenes
        if timeline_summary:
             parts.append(f"Key Moments: {timeline_summary}")

    # Combine parts
    text = ". ".join(filter(None, parts))
    logger.debug(f"Constructed text for embedding ({len(text)} chars): {text[:200]}...")
    return text
