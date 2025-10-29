"""
DCPR Calculator Module
Calculates quantitative Digital Creator Persona Report metrics

Handles:
- Full persona calculation from all posts
- Incremental updates with new posts
- Statistical aggregation (no LLM inference here)
"""

import logging
from typing import Dict, Any, List, Tuple
from collections import defaultdict
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)


class DCPRCalculator:
    """
    Calculates Digital Creator Persona Report from post JSONs
    Supports both full calculation and incremental updates
    """
    
    def __init__(self):
        logger.info("DCPRCalculator initialized")
    
    def create_from_scratch(
        self,
        posts: List[Dict[str, Any]]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Create DCPR from scratch by processing all posts
        
        Args:
            posts: List of post/reel JSONs from Agent 2
            
        Returns:
            Tuple of (final_dcpr, dcpr_stats)
        """
        logger.info(f"Creating DCPR from scratch with {len(posts)} posts")
        
        # Initialize empty stats
        stats = self._initialize_stats()
        
        # Process each post
        for post in posts:
            self._process_post(post, stats)
        
        # Calculate final DCPR from stats
        final_dcpr = self._finalize_calculations(stats)
        
        logger.info("DCPR creation complete")
        return final_dcpr, stats
    
    def update_incrementally(
        self,
        old_stats: Dict[str, Any],
        new_posts: List[Dict[str, Any]]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Update DCPR incrementally with new posts
        
        Args:
            old_stats: Previous dcpr_stats.json
            new_posts: List of new post/reel JSONs
            
        Returns:
            Tuple of (updated_dcpr, updated_stats)
        """
        logger.info(f"Updating DCPR incrementally with {len(new_posts)} new posts")
        
        # Load existing stats
        stats = old_stats.copy()
        
        # Process each new post
        for post in new_posts:
            self._process_post(post, stats)
        
        # Recalculate final DCPR
        updated_dcpr = self._finalize_calculations(stats)
        
        logger.info("Incremental update complete")
        return updated_dcpr, stats
    
    def _initialize_stats(self) -> Dict[str, Any]:
        """Initialize empty statistics structure"""
        return {
            "total_posts_analyzed": 0,
            "first_post_date": None,
            "last_post_date": None,
            
            # Engagement aggregates
            "sum_engagement_rate": 0.0,
            "sum_likes": 0,
            "sum_comments": 0,
            "sum_saves": 0,
            
            # Aesthetic aggregates
            "sum_aesthetic_score": 0.0,
            "aesthetic_score_count": 0,
            
            # Text overlay tracking
            "sum_text_overlay_presence": 0,
            
            # Topic/Theme tracking
            "topic_counts": defaultdict(int),
            "tone_counts": defaultdict(int),
            "emotion_counts": defaultdict(int),
            "audience_intent_counts": defaultdict(int),
            
            # Visual style tracking
            "lighting_tags": defaultdict(int),
            "camera_style_tags": defaultdict(int),
            "composition_tags": defaultdict(int),
            "color_palette": defaultdict(int),
            
            # Audio style tracking
            "music_type_tags": defaultdict(int),
            "speech_style_tags": defaultdict(int),
            
            # Linguistic tracking
            "caption_tone_tags": defaultdict(int),
            "keyword_counts": defaultdict(int),
            
            # Embedding aggregation
            "embedding_accumulator": None,
            "embedding_count": 0,
            
            # Media type tracking
            "media_type_counts": defaultdict(int),
            
            # Posting dates (for frequency calculation)
            "posting_dates": []
        }
    
    def _process_post(
        self,
        post: Dict[str, Any],
        stats: Dict[str, Any]
    ) -> None:
        """
        Process a single post and update stats
        
        Args:
            post: Post or Reel JSON from Agent 2
            stats: Statistics dictionary to update (in-place)
        """
        # Increment post count
        stats["total_posts_analyzed"] += 1
        
        # Track media type
        media_type = post.get("media_type", "unknown")
        stats["media_type_counts"][media_type] += 1
        
        # Extract metadata
        metadata = post.get("metadata", {})
        upload_time = metadata.get("upload_time")
        
        # Update date range
        if upload_time:
            stats["posting_dates"].append(upload_time)
            if stats["first_post_date"] is None or upload_time < stats["first_post_date"]:
                stats["first_post_date"] = upload_time
            if stats["last_post_date"] is None or upload_time > stats["last_post_date"]:
                stats["last_post_date"] = upload_time
        
        # Aggregate engagement metrics
        engagement = post.get("engagement_features", {})
        if engagement_rate := engagement.get("engagement_rate"):
            stats["sum_engagement_rate"] += engagement_rate
        
        stats["sum_likes"] += metadata.get("likes", 0)
        stats["sum_comments"] += metadata.get("comments", 0)
        stats["sum_saves"] += metadata.get("saves", 0)
        
        # Extract content overview
        content = post.get("content_overview", {})
        
        # Topic/Theme
        if primary_theme := content.get("primary_theme"):
            stats["topic_counts"][primary_theme] += 1
        
        # Tone
        if tone := content.get("tone"):
            stats["tone_counts"][tone] += 1
        elif mood := content.get("mood"):
            stats["tone_counts"][mood] += 1
        
        # Emotions
        if emotions := content.get("audience_emotion_targeted"):
            for emotion in emotions:
                stats["emotion_counts"][emotion] += 1
        
        # Visual style
        if lighting := content.get("lighting"):
            stats["lighting_tags"][lighting] += 1
        
        if camera_style := content.get("camera_style"):
            stats["camera_style_tags"][camera_style] += 1
        
        # Music type
        if music_type := content.get("music_type"):
            stats["music_type_tags"][music_type] += 1
        
        # Extract visual analysis
        visual = post.get("visual_analysis", {})
        
        if aesthetic_score := visual.get("aesthetic_score"):
            stats["sum_aesthetic_score"] += aesthetic_score
            stats["aesthetic_score_count"] += 1
        
        if composition := visual.get("composition_type"):
            stats["composition_tags"][composition] += 1
        
        if colors := visual.get("dominant_colors"):
            for color in colors:
                stats["color_palette"][color] += 1
        
        # Extract linguistic analysis
        linguistic = post.get("linguistic_analysis", {})
        
        if caption_tone := linguistic.get("caption_tone"):
            stats["caption_tone_tags"][caption_tone] += 1
        
        if text_overlay := linguistic.get("text_overlay_presence"):
            stats["sum_text_overlay_presence"] += (1 if text_overlay else 0)
        
        # Process hashtags as keywords
        if hashtags := metadata.get("hashtags"):
            for tag in hashtags:
                stats["keyword_counts"][tag] += 1
        
        # Process timeline_analysis for REELs
        if timeline := post.get("timeline_analysis"):
            for segment in timeline:
                # Extract tone from each segment
                if seg_tone := segment.get("tone"):
                    stats["tone_counts"][seg_tone] += 1
        
        # Aggregate embeddings
        if embedding := post.get("embedding"):
            embedding_array = np.array(embedding)
            if stats["embedding_accumulator"] is None:
                stats["embedding_accumulator"] = embedding_array
            else:
                stats["embedding_accumulator"] += embedding_array
            stats["embedding_count"] += 1
    
    def _finalize_calculations(
        self,
        stats: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate final DCPR from aggregated stats
        
        Args:
            stats: Aggregated statistics
            
        Returns:
            Complete DCPR JSON
        """
        total = stats["total_posts_analyzed"]
        
        if total == 0:
            logger.warning("No posts to calculate DCPR from")
            return self._empty_dcpr()
        
        # Calculate averages
        avg_engagement_rate = stats["sum_engagement_rate"] / total
        avg_aesthetic_score = (
            stats["sum_aesthetic_score"] / stats["aesthetic_score_count"]
            if stats["aesthetic_score_count"] > 0 else 0
        )
        text_overlay_rate = stats["sum_text_overlay_presence"] / total
        
        # Calculate topic distribution
        topic_distribution = self._normalize_counts(stats["topic_counts"], total)
        dominant_topics = self._get_top_n(stats["topic_counts"], n=3)
        
        # Calculate tone distribution
        tone_distribution = self._normalize_counts(stats["tone_counts"], total)
        
        # Calculate emotion profile
        emotion_counts = stats["emotion_counts"]
        dominant_emotions = self._get_top_n(emotion_counts, n=3)
        avg_emotional_intensity = self._calculate_emotion_intensity(emotion_counts)
        
        # Calculate audience intent
        audience_intent = self._get_top_n(stats["emotion_counts"], n=5)
        
        # Visual signatures
        visual_signatures = {
            "lighting": self._get_top_n(stats["lighting_tags"], n=5),
            "camera_styles": self._get_top_n(stats["camera_style_tags"], n=5),
            "dominant_color_palette": self._get_top_n(stats["color_palette"], n=8),
            "composition_styles": self._get_top_n(stats["composition_tags"], n=5)
        }
        
        # Audio signatures
        audio_signatures = {
            "music_types": self._get_top_n(stats["music_type_tags"], n=5),
            "speech_style": self._infer_speech_style(stats)
        }
        
        # Linguistic signatures
        linguistic_signatures = {
            "caption_tones": self._get_top_n(stats["caption_tone_tags"], n=5),
            "common_keywords": self._get_top_n(stats["keyword_counts"], n=10),
            "text_overlay_presence_rate": round(text_overlay_rate, 3)
        }
        
        # Calculate posting frequency
        posting_frequency = self._calculate_posting_frequency(stats["posting_dates"])
        
        # Calculate consistency score
        consistency_score = self._calculate_consistency(stats)
        
        # Calculate authenticity score
        authenticity_score = self._calculate_authenticity(stats)
        
        # Calculate persona embedding
        persona_embedding = self._calculate_persona_embedding(stats)
        
        # Assemble final DCPR
        dcpr = {
            "creator_id": None,  # Will be set by caller
            "persona_version": None,  # Will be set by caller
            "identity_summary": {
                "short_summary": "",  # Will be generated by LLM
                "persona_keywords": []  # Will be generated by LLM
            },
            "content_intelligence": {
                "dominant_topics": dominant_topics,
                "topic_distribution": topic_distribution,
                "tone_distribution": tone_distribution,
                "emotional_profile": {
                    "dominant_emotions": dominant_emotions,
                    "average_emotional_intensity": round(avg_emotional_intensity, 2)
                },
                "audience_intent_targeted": audience_intent
            },
            "style_intelligence": {
                "visual_signatures": visual_signatures,
                "audio_signatures": audio_signatures,
                "linguistic_signatures": linguistic_signatures
            },
            "performance_intelligence": {
                "posting_frequency_per_week": round(posting_frequency, 2),
                "avg_engagement_rate": round(avg_engagement_rate, 4),
                "authenticity_score": round(authenticity_score, 2),
                "consistency_score": round(consistency_score, 2),
                "avg_aesthetic_score": round(avg_aesthetic_score, 2),
                "audience_sentiment": self._infer_audience_sentiment(stats),
                "total_posts_analyzed": total
            },
            "persona_embedding": persona_embedding,
            "metadata": {
                "first_post_date": stats["first_post_date"],
                "last_post_date": stats["last_post_date"],
                "last_updated": datetime.utcnow().isoformat() + "Z",
                "data_source": "Instagram",
                "language": "English"
            }
        }
        
        return dcpr
    
    def _normalize_counts(
        self,
        counts: Dict[str, int],
        total: int
    ) -> Dict[str, float]:
        """Normalize counts to percentages"""
        return {
            key: round(count / total, 3)
            for key, count in counts.items()
        }
    
    def _get_top_n(
        self,
        counts: Dict[str, int],
        n: int
    ) -> List[str]:
        """Get top N items by count"""
        sorted_items = sorted(
            counts.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return [item[0] for item in sorted_items[:n]]
    
    def _calculate_emotion_intensity(
        self,
        emotion_counts: Dict[str, int]
    ) -> float:
        """
        Calculate average emotional intensity
        Based on variety and frequency of emotions
        """
        if not emotion_counts:
            return 0.5
        
        total_emotions = sum(emotion_counts.values())
        unique_emotions = len(emotion_counts)
        
        # More variety = higher intensity
        variety_score = min(unique_emotions / 10, 1.0)
        
        # Frequency score
        frequency_score = min(total_emotions / 100, 1.0)
        
        return (variety_score + frequency_score) / 2
    
    def _calculate_posting_frequency(
        self,
        posting_dates: List[str]
    ) -> float:
        """Calculate posts per week"""
        if len(posting_dates) < 2:
            return 0.0
        
        try:
            dates = sorted([
                datetime.fromisoformat(d.replace('Z', '+00:00'))
                for d in posting_dates
            ])
            
            total_days = (dates[-1] - dates[0]).days
            if total_days == 0:
                return 0.0
            
            posts_per_day = len(posting_dates) / total_days
            posts_per_week = posts_per_day * 7
            
            return posts_per_week
        except Exception as e:
            logger.error(f"Error calculating posting frequency: {e}")
            return 0.0
    
    def _calculate_consistency(
        self,
        stats: Dict[str, Any]
    ) -> float:
        """
        Calculate consistency score based on:
        - Topic consistency
        - Visual style consistency
        - Posting regularity
        """
        total = stats["total_posts_analyzed"]
        
        # Topic consistency (entropy-based)
        topic_entropy = self._calculate_entropy(stats["topic_counts"], total)
        topic_score = 1.0 - min(topic_entropy / 3.0, 1.0)
        
        # Visual consistency
        lighting_entropy = self._calculate_entropy(stats["lighting_tags"], total)
        visual_score = 1.0 - min(lighting_entropy / 2.0, 1.0)
        
        # Posting regularity
        regularity_score = self._calculate_posting_regularity(stats["posting_dates"])
        
        # Weighted average
        consistency = (
            0.4 * topic_score +
            0.3 * visual_score +
            0.3 * regularity_score
        )
        
        return consistency
    
    def _calculate_entropy(
        self,
        counts: Dict[str, int],
        total: int
    ) -> float:
        """Calculate Shannon entropy"""
        if total == 0 or not counts:
            return 0.0
        
        entropy = 0.0
        for count in counts.values():
            if count > 0:
                p = count / total
                entropy -= p * np.log2(p)
        
        return entropy
    
    def _calculate_posting_regularity(
        self,
        posting_dates: List[str]
    ) -> float:
        """Calculate how regular posting intervals are"""
        if len(posting_dates) < 3:
            return 0.5
        
        try:
            dates = sorted([
                datetime.fromisoformat(d.replace('Z', '+00:00'))
                for d in posting_dates
            ])
            
            # Calculate intervals
            intervals = [
                (dates[i+1] - dates[i]).days
                for i in range(len(dates) - 1)
            ]
            
            # Lower variance = higher regularity
            mean_interval = np.mean(intervals)
            std_interval = np.std(intervals)
            
            if mean_interval == 0:
                return 0.5
            
            coefficient_of_variation = std_interval / mean_interval
            regularity = 1.0 / (1.0 + coefficient_of_variation)
            
            return regularity
        except Exception as e:
            logger.error(f"Error calculating regularity: {e}")
            return 0.5
    
    def _calculate_authenticity(
        self,
        stats: Dict[str, Any]
    ) -> float:
        """
        Calculate authenticity score based on:
        - Natural engagement patterns
        - Content variety
        - Consistent voice
        """
        # Content variety (more topics = more authentic)
        topic_variety = min(len(stats["topic_counts"]) / 10, 1.0)
        
        # Tone consistency (not too rigid)
        tone_count = len(stats["tone_counts"])
        tone_balance = 0.7 if tone_count >= 2 and tone_count <= 5 else 0.5
        
        # Engagement naturalness (placeholder - would need variance data)
        engagement_naturalness = 0.75
        
        authenticity = (
            0.4 * topic_variety +
            0.3 * tone_balance +
            0.3 * engagement_naturalness
        )
        
        return authenticity
    
    def _calculate_persona_embedding(
        self,
        stats: Dict[str, Any]
    ) -> List[float]:
        """Calculate averaged persona embedding"""
        if stats["embedding_accumulator"] is None or stats["embedding_count"] == 0:
            return []
        
        avg_embedding = stats["embedding_accumulator"] / stats["embedding_count"]
        return avg_embedding.tolist()
    
    def _infer_speech_style(
        self,
        stats: Dict[str, Any]
    ) -> str:
        """Infer speech style from data"""
        # Check if voiceover is common
        music_types = stats["music_type_tags"]
        
        if "voiceover" in music_types or "narration" in music_types:
            return "Voiceover narration"
        elif len(music_types) > 0:
            return "Music-focused (minimal speech)"
        else:
            return "Varied speech patterns"
    
    def _infer_audience_sentiment(
        self,
        stats: Dict[str, Any]
    ) -> str:
        """Infer audience sentiment from engagement and emotions"""
        # High engagement + positive emotions = Positive
        avg_engagement = stats["sum_engagement_rate"] / stats["total_posts_analyzed"]
        
        if avg_engagement > 0.08:
            return "Positive"
        elif avg_engagement > 0.05:
            return "Neutral to Positive"
        else:
            return "Neutral"
    
    def _empty_dcpr(self) -> Dict[str, Any]:
        """Return empty DCPR structure"""
        return {
            "creator_id": None,
            "persona_version": None,
            "identity_summary": {
                "short_summary": "Insufficient data",
                "persona_keywords": []
            },
            "content_intelligence": {
                "dominant_topics": [],
                "topic_distribution": {},
                "tone_distribution": {},
                "emotional_profile": {
                    "dominant_emotions": [],
                    "average_emotional_intensity": 0
                },
                "audience_intent_targeted": []
            },
            "style_intelligence": {
                "visual_signatures": {
                    "lighting": [],
                    "camera_styles": [],
                    "dominant_color_palette": [],
                    "composition_styles": []
                },
                "audio_signatures": {
                    "music_types": [],
                    "speech_style": "Unknown"
                },
                "linguistic_signatures": {
                    "caption_tones": [],
                    "common_keywords": [],
                    "text_overlay_presence_rate": 0
                }
            },
            "performance_intelligence": {
                "posting_frequency_per_week": 0,
                "avg_engagement_rate": 0,
                "authenticity_score": 0,
                "consistency_score": 0,
                "avg_aesthetic_score": 0,
                "audience_sentiment": "Unknown",
                "total_posts_analyzed": 0
            },
            "persona_embedding": [],
            "metadata": {
                "first_post_date": None,
                "last_post_date": None,
                "last_updated": datetime.utcnow().isoformat() + "Z",
                "data_source": "Instagram",
                "language": "English"
            }
        }