import logging
import asyncio
import json
import os
from typing import Dict, Any, List
from pathlib import Path

# Configuration for API calls
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-flash-latest")
LLM_API_KEY = os.getenv("GEMINI_API_KEY", "")
LLM_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent?key={}".format(LLM_MODEL, LLM_API_KEY)
logger = logging.getLogger(__name__)

class EmailDrafting:
    """Uses LLM to parse user query into constraints and draft personalized emails."""

    def __init__(self, retry_handler):
        self.retry_handler = retry_handler
        logger.info(f"Email Drafting initialized with LLM: {LLM_MODEL}")

    async def _call_llm(self, contents: List[Dict[str, Any]], system_instruction: str = None, response_schema: Dict[str, Any] = None) -> str:
        """Internal helper for making LLM API calls with retries."""
        
        payload = {
            "contents": contents,
            "generationConfig": {}
        }
        
        headers = {'Content-Type': 'application/json'}
        
        if system_instruction:
            payload['systemInstruction'] = {"parts": [{"text": system_instruction}]}

        if response_schema:
            payload['generationConfig']['responseMimeType'] = "application/json"
            payload['generationConfig']['responseSchema'] = response_schema
            
        async def fetch_llm_response():
            # Use asyncio.to_thread for synchronous fetch within an async function
            import httpx # Use httpx for asynchronous requests if possible, otherwise rely on to_thread(fetch)
            
            # Since we are using an empty API key and expecting Canvas to inject it, 
            # we rely on the standard synchronous fetch pattern inside asyncio.to_thread
            
            # Note: Using python standard 'requests' inside to_thread for simplicity
            import requests
            
            # Reconstruct the URL for synchronous requests if necessary, otherwise use standard fetch setup
            # Since the environment is controlled, we rely on the standard API URL defined globally
            
            response = requests.post(
                LLM_API_URL, 
                headers=headers, 
                json=payload
            )
            response.raise_for_status()
            
            result = response.json()
            # Extract text from standard Gemini response structure
            if result.get('candidates') and result['candidates'][0].get('content'):
                return result['candidates'][0]['content']['parts'][0]['text']
            
            raise Exception("LLM response missing content or candidate.")

        # Execute with retry handler
        try:
            return await self.retry_handler.execute(fetch_llm_response)
        except Exception as e:
            logger.error(f"LLM API call failed after all retries: {e}")
            return ""

    async def parse_query_to_constraints(self, query: str) -> Dict[str, Any]:
        """
        Uses LLM to convert a simple text query into structured filter constraints.
        """
        logger.info("Parsing user query to structured constraints...")
        
        schema = {
            "type": "OBJECT",
            "properties": {
                "top_n": {"type": "INTEGER", "description": "Number of creators to return (default 5)."},
                "follower_min": {"type": "INTEGER", "description": "Minimum required follower count (default 0)."},
                "follower_max": {"type": "INTEGER", "description": "Maximum required follower count (default 1000000)."},
                "min_quality_score": {"type": "NUMBER", "description": "Minimum overall quality score (0-100)."},
                "min_authenticity_score": {"type": "NUMBER", "description": "Minimum raw authenticity score (0-1)."},
                "required_keywords": {"type": "ARRAY", "items": {"type": "STRING"}, "description": "List of required topics or style keywords."},
                "campaign_goals": {"type": "STRING", "description": "Brief description of the brand's campaign goals."}
            }
        }
        
        system_instruction = (
            "You are a sophisticated AI query parser. Your task is to extract all numerical and thematic "
            "constraints from the user's free-text request and format them into a strict JSON object. "
            "For keywords, look for required topics, tones, or specific aesthetic criteria (e.g., 'keto', 'authentic', 'bright'). "
            "Do NOT invent constraints; use defaults if not specified."
        )
        
        contents = [{"parts": [{"text": f"User Query: {query}"}]}]
        
        json_string = await self._call_llm(contents, system_instruction, schema)
        
        try:
            constraints = json.loads(json_string)
            # Apply defaults if not present
            constraints['top_n'] = constraints.get('top_n', 5)
            constraints['follower_min'] = constraints.get('follower_min', 0)
            constraints['follower_max'] = constraints.get('follower_max', 1000000)
            constraints['required_keywords'] = constraints.get('required_keywords', [])
            return constraints
        except Exception as e:
            logger.error(f"Failed to parse LLM constraint JSON: {e}")
            return {'top_n': 5, 'follower_min': 0, 'follower_max': 1000000, 'required_keywords': []}


    async def draft_personalized_email(
        self,
        brand_goals: str,
        creator_data: Dict[str, Any],
        creator_dcpr: Dict[str, Any],
        storage_path: Path
    ) -> str:
        """
        Drafts a personalized outreach email to a creator.
        """
        creator_id = creator_data['creator_id']
        logger.info(f"Drafting email for creator: {creator_id}")
        
        # Extract personalization points from DCPR
        summary = creator_dcpr.get('identity_summary', {}).get('short_summary', 'their unique content style.')
        top_topic = creator_dcpr.get('content_intelligence', {}).get('dominant_topics', ['content'])[0]
        authenticity = creator_data.get('raw_authenticity', 'high')
        
        system_instruction = (
            "You are a professional brand outreach manager. Draft a concise, personalized, and engaging email "
            "to the creator asking for a partnership. The tone must be respectful and enthusiastic. "
            "Do NOT include placeholders like [BRAND NAME] or [MANAGER NAME]. "
            "The content MUST reference the creator's specific style and metrics to prove the match is genuine."
        )

        prompt = (
            f"Brand Campaign Goal: {brand_goals}\n"
            f"Creator ID: {creator_id}\n"
            f"Creator Persona Summary: {summary}\n"
            f"Key Match Point: The creator's high focus on '{top_topic}' and high raw authenticity score ({authenticity:.2f}).\n\n"
            "Task: Write a complete email draft."
        )

        contents = [{"parts": [{"text": prompt}]}]
        
        email_body = await self._call_llm(contents, system_instruction)
        
        # Save the draft
        try:
            filename = f"email_draft_{creator_id}.txt"
            filepath = storage_path / filename
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(email_body)
            
            logger.info(f"Saved email draft for {creator_id} to {filepath}")
            return str(filepath)
        except Exception as e:
            logger.error(f"Failed to save email draft for {creator_id}: {e}")
            return "Failed to save email."
