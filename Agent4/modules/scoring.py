import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class CreatorScoring:
    """Calculates Quality and Performance scores based on DCPR metrics."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config.get("scoring", {})
        self.quality_weights = self.config.get("quality_weights", {})
        self.performance_weights = self.config.get("performance_weights", {})
        self.scale_min = self.config.get("score_scale_min", 0)
        self.scale_max = self.config.get("score_scale_max", 100)
        
        # Normalize weights to ensure they sum to 1
        self._normalize_weights()
        logger.info("Creator Scoring initialized.")
        logger.debug(f"Quality Weights: {self.quality_weights}")
        logger.debug(f"Performance Weights: {self.performance_weights}")

    def _normalize_weights(self):
        """Ensures weights for each score sum to 1."""
        q_total = sum(self.quality_weights.values())
        if q_total > 0:
            self.quality_weights = {k: v / q_total for k, v in self.quality_weights.items()}
        
        p_total = sum(self.performance_weights.values())
        if p_total > 0:
            self.performance_weights = {k: v / p_total for k, v in self.performance_weights.items()}

    def calculate_scores(self, dcprs: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Calculates scores for a list of DCPRs.

        Args:
            dcprs: A list of DCPR dictionaries.

        Returns:
            pd.DataFrame: DataFrame with creator_id, quality_score, performance_score,
                          and potentially the raw metrics used.
        """
        if not dcprs:
            return pd.DataFrame(columns=['creator_id', 'quality_score', 'performance_score'])

        records = []
        for dcpr in dcprs:
            creator_id = dcpr.get("creator_id", "unknown")
            perf_intel = dcpr.get("performance_intelligence", {})
            content_intel = dcpr.get("content_intelligence", {})
            style_intel = dcpr.get("style_intelligence", {})
            metadata = dcpr.get("metadata", {})
            
            # --- Calculate Quality Score ---
            quality_score = 0.0
            
            # Authenticity (0-1)
            auth_score = perf_intel.get("authenticity_score", 0.0)
            quality_score += self.quality_weights.get("authenticity_score", 0.0) * auth_score
            
            # Consistency (0-1)
            cons_score = perf_intel.get("consistency_score", 0.0)
            quality_score += self.quality_weights.get("consistency_score", 0.0) * cons_score

            # Aesthetics (0-10, normalize to 0-1)
            aes_score = perf_intel.get("avg_aesthetic_score", 0.0) / 10.0
            quality_score += self.quality_weights.get("avg_aesthetic_score", 0.0) * np.clip(aes_score, 0, 1)

            # Topic Diversity (Heuristic: number of dominant topics / 5, clipped 0-1)
            num_topics = len(content_intel.get("dominant_topics", []))
            topic_div_score = np.clip(num_topics / 5.0, 0, 1)
            quality_score += self.quality_weights.get("topic_diversity", 0.0) * topic_div_score
            
            # --- Calculate Performance Score ---
            performance_score = 0.0

            # Engagement Rate (0-1, consider clipping/scaling if needed)
            eng_rate = perf_intel.get("avg_engagement_rate", 0.0)
            # Simple boost for high engagement, capped at a reasonable level (e.g., 20% = 1.0)
            eng_score = np.clip(eng_rate / 0.20, 0, 1) 
            performance_score += self.performance_weights.get("avg_engagement_rate", 0.0) * eng_score

            # Follower Count (Needs careful handling - log transform is common)
            followers = perf_intel.get("follower_count", 0)
            # Log10 transform, scale relative to a max (e.g., 10M = score 1.0)
            # Add 1 to avoid log(0). Max chosen arbitrarily.
            follower_score = np.clip(np.log10(followers + 1) / np.log10(10_000_000 + 1), 0, 1)
            performance_score += self.performance_weights.get("follower_count", 0.0) * follower_score

            # Posting Frequency (Normalize, e.g., 7 posts/week = 1.0)
            freq = perf_intel.get("posting_frequency_per_week", 0.0)
            freq_score = np.clip(freq / 7.0, 0, 1)
            performance_score += self.performance_weights.get("posting_frequency_per_week", 0.0) * freq_score

            # Audience Sentiment (Map to numeric: Positive=1, Neutral=0.5, Negative=0)
            sentiment_map = {"Positive": 1.0, "Neutral to Positive": 0.75, "Neutral": 0.5, "Negative": 0.0, "Unknown": 0.3}
            sentiment = perf_intel.get("audience_sentiment", "Unknown")
            sentiment_score = sentiment_map.get(sentiment, 0.3)
            performance_score += self.performance_weights.get("audience_sentiment", 0.0) * sentiment_score
            
            # --- Scale Scores ---
            scaled_quality = self._scale_score(quality_score)
            scaled_performance = self._scale_score(performance_score)

            records.append({
                "creator_id": creator_id,
                "quality_score": round(scaled_quality, 2),
                "performance_score": round(scaled_performance, 2),
                # Include raw metrics for transparency if needed by Agent 5
                "raw_authenticity": auth_score,
                "raw_consistency": cons_score,
                "raw_aesthetic": perf_intel.get("avg_aesthetic_score", 0.0),
                "raw_engagement_rate": eng_rate,
                "raw_follower_count": followers,
                "raw_posting_frequency": freq,
                "persona_version": dcpr.get("persona_version", "unknown"), # Pass version along
                "total_posts_analyzed": perf_intel.get("total_posts_analyzed", 0) # Data volume context
            })

        logger.info(f"Calculated scores for {len(records)} creators.")
        return pd.DataFrame(records)

    def _scale_score(self, score_0_1: float) -> float:
        """Scales a score from 0-1 range to the configured min/max scale."""
        score_0_1 = np.clip(score_0_1, 0.0, 1.0) # Ensure score is within 0-1
        return self.scale_min + (score_0_1 * (self.scale_max - self.scale_min))
