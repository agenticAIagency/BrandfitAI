from pydantic import BaseModel, Field, validator
from typing import List, Optional, Union, Dict, Any

# --- Common Nested Models ---

class ContentOverview(BaseModel):
    primary_theme: str = Field(..., description="The main topic or subject matter.")
    secondary_theme: Optional[str] = Field(None, description="A secondary topic or subject matter.")
    mood: Optional[str] = Field(None, description="The overall feeling or atmosphere (e.g., Energetic, Calm, Humorous).")
    tone: str = Field(..., description="The style or manner of expression (e.g., Motivational, Educational, Sarcastic).")
    audience_emotion_targeted: Optional[List[str]] = Field(None, description="Emotions the content aims to evoke in the audience (e.g., ['motivation', 'curiosity']).")
    setting: Optional[str] = Field(None, description="The location or environment where the content takes place.")
    brand_presence: Optional[bool] = Field(None, description="Is a brand explicitly mentioned or prominently featured?")

class VisualAnalysisBase(BaseModel):
    dominant_colors: Optional[List[str]] = Field(None, description="List of dominant color hex codes (e.g., ['#FFFFFF', '#000000']).")
    composition_type: Optional[str] = Field(None, description="Description of the visual layout (e.g., Rule of thirds, Centered, Flat lay).")
    objects_detected: Optional[List[str]] = Field(None, description="Key objects visible in the media.")
    focus_subject: Optional[str] = Field(None, description="The main subject the camera focuses on.")
    aesthetic_score: float = Field(..., description="Estimated aesthetic appeal score (0.0-10.0).", ge=0.0, le=10.0)

class LinguisticAnalysis(BaseModel):
    caption_tone: Optional[str] = Field(None, description="The tone detected specifically in the caption text.")
    emotion_keywords: Optional[List[str]] = Field(None, description="Keywords related to emotions found in text (caption, overlays).")
    hashtag_intent: Optional[List[str]] = Field(None, description="Inferred intent or category of the hashtags used (e.g., ['motivation', 'product']).")
    text_overlay_presence: Optional[bool] = Field(None, description="Does the media contain significant text overlays?")
    # Add speech_summary here if Gemini provides overall speech summary for Reels
    speech_summary_overall: Optional[str] = Field(None, description="Overall summary of spoken content if applicable (especially for Reels).")


# --- Reel Specific Models ---

class TimelineSegment(BaseModel):
    timestamp_start: float = Field(..., description="Start time of the segment in seconds.")
    timestamp_end: float = Field(..., description="End time of the segment in seconds.")
    scene_summary: str = Field(..., description="Brief description of the visual scene in this segment.")
    speech_summary: Optional[str] = Field(None, description="Summary of spoken words in this segment, if any.")
    objects_detected: Optional[List[str]] = Field(None, description="Key objects visible in this segment.")
    actions: Optional[List[str]] = Field(None, description="Key actions happening in this segment.")
    facial_expression: Optional[str] = Field(None, description="Dominant facial expression detected, if applicable.")
    tone: Optional[str] = Field(None, description="Tone conveyed in this specific segment (visuals and audio).")
    audio_energy: Optional[float] = Field(None, description="Estimated energy level of the audio (0.0-1.0).", ge=0.0, le=1.0)
    visual_intensity: Optional[float] = Field(None, description="Estimated visual intensity or complexity (0.0-1.0).", ge=0.0, le=1.0)
    engagement_likelihood: Optional[str] = Field(None, description="Estimated likelihood (Low/Medium/High) of audience engagement during this segment.")


class ReelVisualAnalysis(VisualAnalysisBase):
    camera_style: Optional[str] = Field(None, description="Description of camera movement and framing (e.g., Handheld close-ups, Static tripod).")
    lighting: Optional[str] = Field(None, description="Description of the lighting conditions (e.g., Bright artificial, Natural daylight).")
    motion_density: Optional[float] = Field(None, description="Estimated amount of motion in the video (0.0-1.0).", ge=0.0, le=1.0)
    scene_changes: Optional[int] = Field(None, description="Approximate number of distinct scene changes.")

class ReelAudioAnalysis(BaseModel):
     music_type: Optional[str] = Field(None, description="Type or genre of background music, if present.")
     speech_style: Optional[str] = Field(None, description="Description of the speech style (e.g., Energetic VO, Calm narration).")
     sound_effects_present: Optional[bool] = Field(None, description="Are sound effects used prominently?")


# --- Main Analysis Schemas ---

class PostAnalysisResult(BaseModel):
    """Schema for Gemini's analysis of an Image Post."""
    content_overview: ContentOverview
    visual_analysis: VisualAnalysisBase # Use the base one for posts
    linguistic_analysis: LinguisticAnalysis
    # Optional: Add fields Gemini might provide specifically for images
    image_style: Optional[str] = Field(None, description="Specific style if identifiable (e.g., vintage, cartoonish).")
    comment_sentiment_estimated: Optional[str] = Field(None, description="Estimated sentiment (Positive/Neutral/Negative) based on content.")

    @validator('comment_sentiment_estimated')
    def sentiment_must_be_valid(cls, v):
        if v is not None and v not in ["Positive", "Neutral", "Negative"]:
            raise ValueError('comment_sentiment_estimated must be Positive, Neutral, or Negative')
        return v


class ReelAnalysisResult(BaseModel):
    """Schema for Gemini's analysis of a Video Reel."""
    content_overview: ContentOverview
    timeline_analysis: List[TimelineSegment] = Field(..., description="Segment-by-segment analysis of the video.")
    visual_analysis: ReelVisualAnalysis # Use the reel-specific one
    audio_analysis: ReelAudioAnalysis
    linguistic_analysis: LinguisticAnalysis # Includes overall speech summary if provided
    comment_sentiment_estimated: Optional[str] = Field(None, description="Estimated sentiment (Positive/Neutral/Negative) based on content.")

    @validator('comment_sentiment_estimated')
    def sentiment_must_be_valid(cls, v):
        if v is not None and v not in ["Positive", "Neutral", "Negative"]:
            raise ValueError('comment_sentiment_estimated must be Positive, Neutral, or Negative')
        return v

# --- Final Output Schema (for Agent 3) ---
# This isn't used for validation against Gemini, but defines the structure Agent 2 saves

class EngagementFeatures(BaseModel):
     engagement_rate: Optional[float] = Field(None, description="Calculated engagement rate.")
     # Add comment_sentiment_estimated here from the analysis result?
     comment_sentiment: Optional[str] = Field(None, description="Estimated sentiment from analysis.")

class FinalOutputMetadata(BaseModel):
     creator_id: str
     post_id: str
     upload_time: Optional[str] = None
     caption: Optional[str] = None
     hashtags: Optional[List[str]] = None
     likes: Optional[int] = None
     comments: Optional[int] = None
     views: Optional[int] = None # Important for engagement rate
     shares: Optional[int] = None # Included in demo data
     saves: Optional[int] = None # Included in demo data
     duration_seconds: Optional[float] = None
     media_type: str # 'post' or 'reel'
     is_promotional: Optional[bool] = None
     geo_tag: Optional[str] = None
     creator_follower_count: Optional[int] = None # Added field


class FinalOutputSchema(BaseModel):
    """Defines the final JSON structure saved for Agent 3."""
    # Combine Analysis and Original Metadata
    metadata: FinalOutputMetadata
    # Analysis results nested under keys matching Agent 3's expectations
    content_overview: Optional[ContentOverview] = None # From analysis
    visual_analysis: Optional[Union[VisualAnalysisBase, ReelVisualAnalysis]] = None # From analysis
    audio_analysis: Optional[ReelAudioAnalysis] = None # From analysis (Reels only)
    linguistic_analysis: Optional[LinguisticAnalysis] = None # From analysis
    timeline_analysis: Optional[List[TimelineSegment]] = None # From analysis (Reels only)
    # Calculated/Added Fields
    embedding: List[float] = Field(..., description="Normalized embedding vector of the analysis text.")
    engagement_features: EngagementFeatures = Field(..., description="Calculated engagement metrics.")

    # Include original post_id and creator_id at top level too for easier access?
    post_id: str
    creator_id: str

    class Config:
        # Pydantic v2 needs this if using Union types sometimes
        # extra = 'ignore' # If needed
        pass

