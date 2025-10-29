import json
import os
import sys
import asyncio
import glob
from pathlib import Path
from typing import List, Dict, Any, Optional

# --- Libraries we need ---
import google.generativeai as genai
from pydantic import ValidationError
from dotenv import load_dotenv
import instructor # Used for Pydantic enforcement

# --- Import schemas ---
try:
    # Assuming Agent5 folder structure is set up
    sys.path.append(str(Path(__file__).parent / 'modules'))
    from agent5_schemas import MatchmakerOutput, CreatorDCPR
except ImportError:
    # Fallback/Error if schemas are not accessible
    print("FATAL: Could not import Pydantic schemas. Ensure agent5_schemas.py is in Agent5/modules.")
    sys.exit(1)

# --- Configuration & Initialization ---
load_dotenv()

# Configuration constants
UI_DATA_DIR = Path("E:/AGENTIC_AI/Project_agent3/Agent5/UI_DATA") # Use the exact path provided by the user
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash") # Use flash for speed/cost
MAX_MATCHES = 5
DEFAULT_AUTH_MIN = 0.7
DEFAULT_FOLLOWER_MAX = 200000

# Initialize Gemini Client
client = None
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        # Use instructor to patch the model for structured output
        base_model = genai.GenerativeModel(GEMINI_MODEL)
        client = instructor.patch(base_model)
        print(f"Gemini Client initialized for model: {GEMINI_MODEL}")
    except Exception as e:
        print(f"Error initializing Gemini client: {e}")
        client = None
else:
    print("WARNING: GEMINI_API_KEY not found. LLM matching will fail.")


def load_creator_data(data_dir: Path = UI_DATA_DIR) -> List[Dict[str, Any]]:
    """
    Loads all DCPR JSON files from the specified directory.
    """
    creator_data = []
    print(f"Loading DCPR files from: {data_dir}...")
    
    if not data_dir.exists():
        print(f"Error: Data directory not found at {data_dir}")
        return []

    # Look for files ending with .json in the directory
    for file_path in glob.glob(str(data_dir / '*.json')):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
            
            # Use Pydantic model to verify key fields are present before processing
            validated_data = CreatorDCPR.model_validate(raw_data)
            creator_data.append(validated_data.model_dump())
            
        except FileNotFoundError:
            print(f"Skipped: File not found: {file_path}")
        except json.JSONDecodeError:
            print(f"Skipped: Invalid JSON in file: {file_path}")
        except ValidationError as e:
            print(f"Skipped: DCPR schema validation failed for {file_path}. Error: {e}")
            
    print(f"Successfully loaded {len(creator_data)} valid DCPR files.")
    return creator_data


def construct_matching_prompt(
    user_query: str,
    creator_data: List[Dict[str, Any]]
) -> str:
    """
    Constructs the detailed prompt for the LLM to perform filtering, ranking,
    and drafting in one step.
    """
    
    # 1. Summarize available creator data for the prompt
    creator_summaries = []
    for creator in creator_data:
        # Extract only the fields the LLM needs to reason about
        summary = {
            "creator_id": creator['creator_id'],
            "persona_keywords": creator.get('identity_summary', {}).get('persona_keywords', []),
            "dominant_topics": creator.get('content_intelligence', {}).get('dominant_topics', []),
            "authenticity_score": creator.get('performance_intelligence', {}).get('authenticity_score', 0.0),
            "avg_aesthetic_score": creator.get('performance_intelligence', {}).get('avg_aesthetic_score', 0.0),
            "avg_engagement_rate": creator.get('performance_intelligence', {}).get('avg_engagement_rate', 0.0),
            "follower_count": creator.get('performance_intelligence', {}).get('follower_count', 0),
            "short_summary": creator.get('identity_summary', {}).get('short_summary', ''),
            "audience_intent": creator.get('content_intelligence', {}).get('audience_intent_targeted', [])
        }
        creator_summaries.append(summary)

    # 2. Construct the full instruction prompt
    
    # Define constraints clearly for the LLM
    constraints_text = f"""
    * HARD FILTER (MANDATORY):
        - Follower Count MUST NOT exceed {DEFAULT_FOLLOWER_MAX}.
        - Authenticity Score MUST NOT be below {DEFAULT_AUTH_MIN}.
    """
    
    prompt = f"""
    You are a Senior AI Matchmaking Analyst and Copywriter. Your task is to analyze a user's brand query against a list of Digital Creator Persona Reports (DCPRs).

    **INSTRUCTIONS:**
    1. **INFER CONSTRAINTS:** Analyze the User Query to infer semantic and numeric constraints (e.g., target niche, max followers, required formats like 'reels'). Use the default constraints provided below if the query does not explicitly override them.
    2. **HARD FILTER:** Eliminate any creator who fails the mandatory follower or authenticity score constraints.
    3. **SEMANTIC RANK:** Rank the remaining creators based on the semantic match between their `persona_keywords`/`dominant_topics` and the User Query.
    4. **DRAFT EMAILS:** For the **Top {MAX_MATCHES}** creators, draft a highly personalized, professional outreach email, ensuring the body references specific data points from their profile (e.g., mentioning their exact aesthetic score, dominant topic, or audience intent) to justify the collaboration.

    **USER QUERY:**
    {user_query}

    **MANDATORY CONSTRAINTS:**
    {constraints_text}

    **CREATOR POOL DATA (Analyze these for filtering and drafting):**
    {json.dumps(creator_summaries, indent=2)}
    
    **OUTPUT REQUIREMENT:**
    Return ONLY a single JSON object strictly matching the MatchmakerOutput schema.
    """
    return prompt


async def execute_matchmaker_chain(user_query: str) -> Dict[str, Any]:
    """
    Executes the entire LLM chain from data loading to final structured output.
    """
    if client is None:
        raise ConnectionError("Gemini client is not initialized due to missing API key or configuration errors.")
        
    # 1. Load Data
    creator_data = load_creator_data()
    if not creator_data:
        return {
            "search_criteria_applied": {},
            "top_matches": [],
            "recommendation_summary": "No valid creator data available to perform matching."
        }

    # 2. Construct Prompt
    full_prompt = construct_matching_prompt(user_query, creator_data)
    
    # 3. Execute LLM Call (with Pydantic Guardrail)
    print("Executing LLM Matchmaking and Drafting...")
    try:
        # Use a higher temperature for creative tasks like email drafting
        match_result: MatchmakerOutput = await asyncio.to_thread(
            client.generate_content,
            contents=[full_prompt],
            response_model=MatchmakerOutput,
            generation_config=genai.GenerationConfig(
                temperature=0.7, # Higher temp for creativity
                timeout=120 # Give it plenty of time
            )
        )
        
        print("LLM Matchmaking complete and validated.")
        return match_result.model_dump()

    except Exception as e:
        print(f"Fatal Error during LLM execution or validation: {e}")
        return {
            "error": "LLM_EXECUTION_FAILURE",
            "message": str(e),
            "detail": "Failed to get structured JSON from LLM after multiple attempts."
        }


# --- Example Execution Block ---
async def main():
    if not os.path.exists(UI_DATA_DIR):
        print(f"\nNOTE: Creating placeholder directory: {UI_DATA_DIR}. Please place DCPR JSON files here.")
        UI_DATA_DIR.mkdir(parents=True, exist_ok=True)
        return

    # Example Query
    query = "I need an authentic, high-quality content creator who focuses on motivational fitness reels. Max followers should be 200,000. Campaign goal is a product review."
    
    # Ensure all original DCPR fields are simulated in the UI_DATA folder for a successful test
    # (The actual testing relies on the user providing data.)
    
    results = await execute_matchmaker_chain(query)
    
    print("\n" + "="*80)
    print("AGENT 5 MATCHMAKER MVP RESULTS")
    print("="*80)
    print(json.dumps(results, indent=2))
    print("="*80)


if __name__ == "__main__":
    asyncio.run(main())
