"""
Confidence Calculator
Calculates confidence score for generated personas

Factors:
- Data volume (number of posts)
- Data quality score
- Temporal coverage
- Consistency metrics
"""

import numpy as np
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class ConfidenceCalculator:
    """
    Calculates confidence score for persona generation
    """
    
    def __init__(
        self,
        min_posts_target: int = 20,
        min_temporal_days: int = 60,
        quality_weight: float = 0.3,
        volume_weight: float = 0.25,
        temporal_weight: float = 0.25,
        consistency_weight: float = 0.2
    ):
        """
        Initialize confidence calculator with weights
        
        Args:
            min_posts_target: Target minimum posts for full confidence
            min_temporal_days: Target temporal coverage in days
            quality_weight: Weight for data quality factor
            volume_weight: Weight for data volume factor
            temporal_weight: Weight for temporal coverage factor
            consistency_weight: Weight for consistency metrics factor
        """
        self.min_posts_target = min_posts_target
        self.min_temporal_days = min_temporal_days
        
        # Normalize weights
        total_weight = quality_weight + volume_weight + temporal_weight + consistency_weight
        self.quality_weight = quality_weight / total_weight
        self.volume_weight = volume_weight / total_weight
        self.temporal_weight = temporal_weight / total_weight
        self.consistency_weight = consistency_weight / total_weight
        
        logger.info("ConfidenceCalculator initialized")
    
    def calculate(
        self,
        num_posts: int,
        data_quality: float,
        temporal_coverage: int,
        consistency_score: float
    ) -> float:
        """
        Calculate overall confidence score (0-1)
        
        Args:
            num_posts: Number of posts analyzed
            data_quality: Input data quality score (0-1)
            temporal_coverage: Days of temporal coverage
            consistency_score: Content consistency score (0-1)
            
        Returns:
            Confidence score between 0 and 1
        """
        # Calculate individual components
        volume_score = self._calculate_volume_score(num_posts)
        quality_score = data_quality
        temporal_score = self._calculate_temporal_score(temporal_coverage)
        consistency_contribution = consistency_score
        
        # Weighted combination
        confidence = (
            self.quality_weight * quality_score +
            self.volume_weight * volume_score +
            self.temporal_weight * temporal_score +
            self.consistency_weight * consistency_contribution
        )
        
        # Apply penalties for critical issues
        confidence = self._apply_penalties(
            confidence,
            num_posts,
            data_quality,
            temporal_coverage
        )
        
        # Ensure in valid range
        confidence = float(np.clip(confidence, 0.0, 1.0))
        
        logger.info(f"Calculated confidence: {confidence:.3f}")
        return confidence
    
    def _calculate_volume_score(self, num_posts: int) -> float:
        """
        Score based on number of posts analyzed
        
        Uses sigmoid function to gradually increase score
        """
        # Sigmoid function centered at min_posts_target
        score = 1 / (1 + np.exp(-0.2 * (num_posts - self.min_posts_target)))
        return float(score)
    
    def _calculate_temporal_score(self, temporal_coverage: int) -> float:
        """
        Score based on temporal coverage in days
        
        Longer coverage = better understanding of evolution
        """
        if temporal_coverage <= 0:
            return 0.0
        
        # Logarithmic scale - diminishing returns after target
        score = min(
            np.log1p(temporal_coverage) / np.log1p(self.min_temporal_days),
            1.0
        )
        return float(score)
    
    def _apply_penalties(
        self,
        base_confidence: float,
        num_posts: int,
        data_quality: float,
        temporal_coverage: int
    ) -> float:
        """
        Apply penalties for critical issues
        """
        confidence = base_confidence
        
        # Severe penalty for very low post count
        if num_posts < 5:
            confidence *= 0.5
        
        # Penalty for poor data quality
        if data_quality < 0.5:
            confidence *= 0.7
        
        # Penalty for minimal temporal coverage
        if temporal_coverage < 14:  # Less than 2 weeks
            confidence *= 0.8
        
        return confidence
    
    def interpret_confidence(self, confidence: float) -> Dict[str, Any]:
        """
        Provide human-readable interpretation of confidence score
        
        Args:
            confidence: Confidence score (0-1)
            
        Returns:
            Dictionary with interpretation and recommendations
        """
        if confidence >= 0.9:
            level = "very_high"
            description = "Excellent data quality and volume"
            recommendation = "ready_for_matching"
            color = "green"
        elif confidence >= 0.7:
            level = "high"
            description = "Good data quality, reliable persona"
            recommendation = "ready_for_matching"
            color = "green"
        elif confidence >= 0.5:
            level = "moderate"
            description = "Adequate data, but could benefit from more samples"
            recommendation = "usable_with_caution"
            color = "yellow"
        elif confidence >= 0.3:
            level = "low"
            description = "Limited data, persona may be incomplete"
            recommendation = "needs_more_data"
            color = "orange"
        else:
            level = "very_low"
            description = "Insufficient data for reliable persona"
            recommendation = "insufficient_data"
            color = "red"
        
        return {
            "level": level,
            "score": confidence,
            "description": description,
            "recommendation": recommendation,
            "color": color
        }
    
    def calculate_detailed(
        self,
        num_posts: int,
        data_quality: float,
        temporal_coverage: int,
        consistency_score: float
    ) -> Dict[str, Any]:
        """
        Calculate confidence with detailed breakdown
        
        Returns full breakdown of scoring components
        """
        volume_score = self._calculate_volume_score(num_posts)
        quality_score = data_quality
        temporal_score = self._calculate_temporal_score(temporal_coverage)
        
        confidence = (
            self.quality_weight * quality_score +
            self.volume_weight * volume_score +
            self.temporal_weight * temporal_score +
            self.consistency_weight * consistency_score
        )
        
        confidence = self._apply_penalties(
            confidence,
            num_posts,
            data_quality,
            temporal_coverage
        )
        
        confidence = float(np.clip(confidence, 0.0, 1.0))
        
        return {
            "overall_confidence": confidence,
            "components": {
                "volume_score": float(volume_score),
                "quality_score": float(quality_score),
                "temporal_score": float(temporal_score),
                "consistency_score": float(consistency_score)
            },
            "weights": {
                "volume_weight": self.volume_weight,
                "quality_weight": self.quality_weight,
                "temporal_weight": self.temporal_weight,
                "consistency_weight": self.consistency_weight
            },
            "interpretation": self.interpret_confidence(confidence)
        }