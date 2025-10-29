import asyncio
import json
import os
import requests # Make sure requests is imported
import logging
from typing import Dict, Any, List

# --- NEW IMPORTS ---
import instructor
from pydantic import BaseModel, Field

# Import the correct Ollama client
from ollama import AsyncClient

logger = logging.getLogger(__name__)

# --- STEP 1: Define the Guardrail/Pydantic Model ---
class IdentitySummary(BaseModel):
    """The required JSON structure for the identity summary."""
    short_summary: str = Field(..., description="A concise 1-2 sentence description of this creator's persona")
    persona_keywords: List[str] = Field(..., description="A list of 5-7 keywords that capture their essence")

class PersonaSummarizationModule:
    """
    Generates ONLY identity_summary using LLM from pre-calculated DCPR
    Supports: Gemini API or Local Ollama
    """
    
    def __init__(
        self,
        model: str = None,
        temperature: float = 0.3,
        api_key: str = None,
        use_ollama: bool = None,
        ollama_base_url: str = None
    ):
        """
        Initialize with Gemini or Ollama
        """
        # Determine LLM backend
        self.use_ollama = use_ollama if use_ollama is not None else os.getenv("USE_OLLAMA", "false").lower() == "true"
        self.temperature = temperature
        
        if self.use_ollama:
            # --- FIX: DO NOT PATCH THE OLLAMA CLIENT ---
            # We will call it manually to avoid the patch error.
            self.client = AsyncClient(
                host=ollama_base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            )
            self.model_name = model or os.getenv("OLLAMA_MODEL", "llama3.2")
            self.use_mock = False
            logger.info(f"PersonaSummarizationModule initialized with Ollama (Manual Mode): {self.model_name}")
        
        else:
            # Configure Gemini
            self.model_name = model or os.getenv("LLM_MODEL", "gemini-pro")
            api_key = api_key or os.getenv("GEMINI_API_KEY")
            
            if not api_key:
                logger.warning("No GEMINI_API_KEY found. Using mock responses.")
                self.use_mock = True
                self.client = None # No client if mocking
            else:
                import google.generativeai as genai
                genai.configure(api_key=api_key)
                
                # Patching the Gemini client works fine.
                self.client = instructor.patch(
                    genai.GenerativeModel(self.model_name)
                )
                self.use_mock = False
                logger.info(f"PersonaSummarizationModule initialized with Gemini: {self.model_name}")

    
    async def generate_identity_summary(
        self,
        dcpr: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate ONLY the identity_summary from pre-calculated DCPR
        """
        logger.info("Generating identity summary")
        
        # We need to add JSON format instructions for the manual Ollama call
        add_json_rules = self.use_ollama
        prompt = self._build_identity_prompt(dcpr, add_json_format_rules=add_json_rules)
        
        if self.use_mock or (self.client is None and not self.use_ollama):
            logger.warning("Using mock response")
            summary_model = self._get_mock_summary()
            return summary_model.model_dump() # Convert Pydantic to dict

        try:
            if self.use_ollama:
                # --- FIX: Call Ollama manually ---
                response = await self.client.chat(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    format="json" # Tell Ollama to return a JSON string
                )
                
                # Extract the JSON string from the response
                response_text = response['message']['content']
                logger.info("Ollama response received, parsing...")
                
                # Manually parse the JSON
                parsed_data = json.loads(response_text)
                # Manually validate with the Pydantic model
                summary_model = IdentitySummary(**parsed_data)
            
            else: 
                # --- Gemini path (with instructor) ---
                import google.generativeai as genai # Ensure genai is available
                summary_model = await self.client.generate_content(
                    prompt,
                    response_model=IdentitySummary,
                    generation_config=genai.types.GenerationConfig(
                        temperature=self.temperature
                    )
                )
            
            logger.info("Identity summary generated and validated.")
            # Convert the Pydantic model back to a dict for the DCPR
            return summary_model.model_dump() 
            
        except Exception as e:
            logger.error(f"Failed to generate or validate LLM response: {e}", exc_info=True)
            fallback_model = self._create_fallback_summary()
            return fallback_model.model_dump()


    def _build_identity_prompt(self, dcpr: Dict[str, Any], add_json_format_rules: bool = False) -> str:
        """
        Build simplified prompt.
        """
        content_intel = dcpr.get("content_intelligence", {})
        style_intel = dcpr.get("style_intelligence", {})
        performance = dcpr.get("performance_intelligence", {})
        
        # Build the prompt as a list of strings
        prompt_lines = [
            "You are an expert social media analyst. Based on the following PRE-CALCULATED metrics for a creator, write a concise identity summary.",
            "\n**PRE-CALCULATED CONTENT INTELLIGENCE:**",
            f"- Dominant Topics: {content_intel.get('dominant_topics', [])}",
            f"- Topic Distribution: {content_intel.get('topic_distribution', {})} \n... (rest of metrics) ...",
            f"- Total Posts Analyzed: {performance.get('total_posts_analyzed', 0)}",
            "\n**TASK:**",
            "Based ONLY on the above metrics, generate:",
            "1. A concise 1-2 sentence description of this creator's persona",
            "2. A list of 5-7 keywords that capture their essence",
            "\n**RULES:**",
            "- Be specific and descriptive",
            "- ... (rest of rules) ...",
        ]

        # --- FIX: Conditionally add JSON format instructions ---
        if add_json_format_rules:
            prompt_lines.extend([
                "\n**OUTPUT FORMAT (JSON only, no markdown):**",
                '{',
                '  "short_summary": "Your 1-2 sentence summary here",',
                '  "persona_keywords": ["keyword1", "keyword2", "keyword3", "keyword4", "keyword5"]',
                '}'
            ])
        
        # Re-join all lines into the full prompt
        # (I'm using a placeholder for the full metric list for brevity)
        full_metrics_prompt = f"""You are an expert social media analyst. Based on the following PRE-CALCULATED metrics for a creator, write a concise identity summary.

**PRE-CALCULATED CONTENT INTELLIGENCE:**
- Dominant Topics: {content_intel.get('dominant_topics', [])}
- Topic Distribution: {content_intel.get('topic_distribution', {})}
- Dominant Emotions: {content_intel.get('emotional_profile', {}).get('dominant_emotions', [])}
- Emotional Intensity: {content_intel.get('emotional_profile', {}).get('average_emotional_intensity', 0)}
- Audience Intent: {content_intel.get('audience_intent_targeted', [])}

**PRE-CALCULATED STYLE INTELLIGENCE:**
- Visual Signatures:
  - Lighting: {style_intel.get('visual_signatures', {}).get('lighting', [])}
  - Camera Styles: {style_intel.get('visual_signatures', {}).get('camera_styles', [])}
  - Composition: {style_intel.get('visual_signatures', {}).get('composition_styles', [])}
- Audio Signatures:
  - Music Types: {style_intel.get('audio_signatures', {}).get('music_types', [])}
  - Speech Style: {style_intel.get('audio_signatures', {}).get('speech_style', '')}
- Linguistic Signatures:
  - Caption Tones: {style_intel.get('linguistic_signatures', {}).get('caption_tones', [])}
  - Common Keywords: {style_intel.get('linguistic_signatures', {}).get('common_keywords', [])[:5]}

**PRE-CALCULATED PERFORMANCE:**
- Posting Frequency: {performance.get('posting_frequency_per_week', 0)} posts/week
- Avg Engagement Rate: {performance.get('avg_engagement_rate', 0):.2%}
- Authenticity Score: {performance.get('authenticity_score', 0):.2f}/1.0
- Consistency Score: {performance.get('consistency_score', 0):.2f}/1.0
- Audience Sentiment: {performance.get('audience_sentiment', 'Unknown')}
- Total Posts Analyzed: {performance.get('total_posts_analyzed', 0)}

**TASK:**
Based ONLY on the above metrics, generate:
1. A concise 1-2 sentence description of this creator's persona
2. A list of 5-7 keywords that capture their essence

**RULES:**
- Be specific and descriptive
- Focus on what makes this creator unique
- Use the actual topics, styles, and metrics provided
- Keywords should be single words or short phrases
"""
        
        if add_json_format_rules:
             full_metrics_prompt += """
**OUTPUT FORMAT (JSON only, no markdown):**
{
  "short_summary": "Your 1-2 sentence summary here",
  "persona_keywords": ["keyword1", "keyword2", "keyword3", "keyword4", "keyword5"]
}
"""
        return full_metrics_prompt

    def _get_mock_summary(self) -> IdentitySummary:
        """Mock response as a Pydantic model."""
        return IdentitySummary(
            short_summary="Fitness and wellness creator with consistent, motivational content.",
            persona_keywords=["fitness", "wellness", "motivation", "authentic", "consistent"]
        )

    def _create_fallback_summary(self) -> IdentitySummary:
        """Fallback identity summary as a Pydantic model."""
        return IdentitySummary(
            short_summary="Content creator with diverse topics and consistent posting schedule.",
            persona_keywords=["content creator", "consistent", "diverse"]
        )