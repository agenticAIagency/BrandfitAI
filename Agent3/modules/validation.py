"""
Validation Module (REFACTORED)
Validates new DCPR schema (5-layer structure)

Validates:
- identity_summary
- content_intelligence
- style_intelligence
- performance_intelligence
- persona_embedding
- metadata
"""

from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class ValidationModule:
    """
    Schema validation for DCPR output
    """
    
    def __init__(self):
        """Initialize validation rules"""
        self.required_top_level_fields = [
            "creator_id",
            "persona_version",
            "identity_summary",
            "content_intelligence",
            "style_intelligence",
            "performance_intelligence",
            "persona_embedding",
            "metadata"
        ]
        logger.info("ValidationModule initialized with DCPR schema")
    
    def validate_dcpr_output(
        self,
        dcpr: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate complete DCPR structure
        
        Args:
            dcpr: Generated DCPR dictionary
            
        Returns:
            Validation result with is_valid flag and errors
        """
        logger.info("Validating DCPR output")
        
        errors = []
        warnings = []
        
        # 1. Check top-level fields
        for field in self.required_top_level_fields:
            if field not in dcpr:
                errors.append(f"Missing required top-level field: {field}")
        
        # 2. Validate identity_summary
        if "identity_summary" in dcpr:
            identity_errors = self._validate_identity_summary(dcpr["identity_summary"])
            errors.extend(identity_errors)
        
        # 3. Validate content_intelligence
        if "content_intelligence" in dcpr:
            content_errors = self._validate_content_intelligence(dcpr["content_intelligence"])
            errors.extend(content_errors)
        
        # 4. Validate style_intelligence
        if "style_intelligence" in dcpr:
            style_errors = self._validate_style_intelligence(dcpr["style_intelligence"])
            errors.extend(style_errors)
        
        # 5. Validate performance_intelligence
        if "performance_intelligence" in dcpr:
            perf_errors = self._validate_performance_intelligence(dcpr["performance_intelligence"])
            errors.extend(perf_errors)
        
        # 6. Validate persona_embedding
        if "persona_embedding" in dcpr:
            embedding_warnings = self._validate_persona_embedding(dcpr["persona_embedding"])
            warnings.extend(embedding_warnings)
        
        # 7. Validate metadata
        if "metadata" in dcpr:
            metadata_errors = self._validate_metadata(dcpr["metadata"])
            errors.extend(metadata_errors)
        
        is_valid = len(errors) == 0
        
        result = {
            "is_valid": is_valid,
            "errors": errors,
            "warnings": warnings,
            "validation_timestamp": self._get_timestamp()
        }
        
        if is_valid:
            logger.info("DCPR validation passed")
        else:
            logger.warning(f"DCPR validation failed with {len(errors)} errors")
        
        return result
    
    def _validate_identity_summary(
        self,
        identity: Dict[str, Any]
    ) -> List[str]:
        """Validate identity_summary structure"""
        errors = []
        
        required = ["short_summary", "persona_keywords"]
        for field in required:
            if field not in identity:
                errors.append(f"identity_summary missing field: {field}")
        
        # Validate short_summary
        if "short_summary" in identity:
            if not isinstance(identity["short_summary"], str):
                errors.append("short_summary must be a string")
            elif len(identity["short_summary"]) < 10:
                errors.append("short_summary too short (min 10 chars)")
        
        # Validate persona_keywords
        if "persona_keywords" in identity:
            if not isinstance(identity["persona_keywords"], list):
                errors.append("persona_keywords must be a list")
            elif len(identity["persona_keywords"]) < 3:
                errors.append("persona_keywords must have at least 3 keywords")
        
        return errors
    
    def _validate_content_intelligence(
        self,
        content: Dict[str, Any]
    ) -> List[str]:
        """Validate content_intelligence structure"""
        errors = []
        
        required = [
            "dominant_topics",
            "topic_distribution",
            "tone_distribution",
            "emotional_profile",
            "audience_intent_targeted"
        ]
        
        for field in required:
            if field not in content:
                errors.append(f"content_intelligence missing field: {field}")
        
        # Validate emotional_profile
        if "emotional_profile" in content:
            emotional = content["emotional_profile"]
            if "dominant_emotions" not in emotional:
                errors.append("emotional_profile missing dominant_emotions")
            if "average_emotional_intensity" not in emotional:
                errors.append("emotional_profile missing average_emotional_intensity")
            elif not isinstance(emotional["average_emotional_intensity"], (int, float)):
                errors.append("average_emotional_intensity must be numeric")
            elif not (0 <= emotional["average_emotional_intensity"] <= 1):
                errors.append("average_emotional_intensity must be between 0 and 1")
        
        return errors
    
    def _validate_style_intelligence(
        self,
        style: Dict[str, Any]
    ) -> List[str]:
        """Validate style_intelligence structure"""
        errors = []
        
        required = ["visual_signatures", "audio_signatures", "linguistic_signatures"]
        
        for field in required:
            if field not in style:
                errors.append(f"style_intelligence missing field: {field}")
        
        # Validate visual_signatures
        if "visual_signatures" in style:
            visual = style["visual_signatures"]
            required_visual = ["lighting", "camera_styles", "dominant_color_palette", "composition_styles"]
            for field in required_visual:
                if field not in visual:
                    errors.append(f"visual_signatures missing field: {field}")
                elif not isinstance(visual[field], list):
                    errors.append(f"visual_signatures.{field} must be a list")
        
        # Validate audio_signatures
        if "audio_signatures" in style:
            audio = style["audio_signatures"]
            if "music_types" not in audio:
                errors.append("audio_signatures missing music_types")
            if "speech_style" not in audio:
                errors.append("audio_signatures missing speech_style")
        
        # Validate linguistic_signatures
        if "linguistic_signatures" in style:
            linguistic = style["linguistic_signatures"]
            required_ling = ["caption_tones", "common_keywords", "text_overlay_presence_rate"]
            for field in required_ling:
                if field not in linguistic:
                    errors.append(f"linguistic_signatures missing field: {field}")
            
            # Validate text_overlay_presence_rate
            if "text_overlay_presence_rate" in linguistic:
                rate = linguistic["text_overlay_presence_rate"]
                if not isinstance(rate, (int, float)):
                    errors.append("text_overlay_presence_rate must be numeric")
                elif not (0 <= rate <= 1):
                    errors.append("text_overlay_presence_rate must be between 0 and 1")
        
        return errors
    
    def _validate_performance_intelligence(
        self,
        performance: Dict[str, Any]
    ) -> List[str]:
        """Validate performance_intelligence structure"""
        errors = []
        
        required = [
            "posting_frequency_per_week",
            "avg_engagement_rate",
            "authenticity_score",
            "consistency_score",
            "avg_aesthetic_score",
            "audience_sentiment",
            "total_posts_analyzed"
        ]
        
        for field in required:
            if field not in performance:
                errors.append(f"performance_intelligence missing field: {field}")
        
        # Validate numeric ranges
        numeric_fields = {
            "posting_frequency_per_week": (0, 50),
            "avg_engagement_rate": (0, 1),
            "authenticity_score": (0, 1),
            "consistency_score": (0, 1),
            "avg_aesthetic_score": (0, 10),
            "total_posts_analyzed": (0, float('inf'))
        }
        
        for field, (min_val, max_val) in numeric_fields.items():
            if field in performance:
                value = performance[field]
                if not isinstance(value, (int, float)):
                    errors.append(f"{field} must be numeric")
                elif not (min_val <= value <= max_val):
                    errors.append(f"{field} out of range: {value} not in [{min_val}, {max_val}]")
        
        # Validate audience_sentiment
        if "audience_sentiment" in performance:
            valid_sentiments = ["Positive", "Neutral to Positive", "Neutral", "Negative", "Unknown"]
            if performance["audience_sentiment"] not in valid_sentiments:
                errors.append(f"Invalid audience_sentiment: {performance['audience_sentiment']}")
        
        return errors
    
    def _validate_persona_embedding(
        self,
        embedding: List[float]
    ) -> List[str]:
        """Validate persona_embedding"""
        warnings = []
        
        if not isinstance(embedding, list):
            warnings.append("persona_embedding should be a list")
        elif len(embedding) == 0:
            warnings.append("persona_embedding is empty")
        elif len(embedding) < 5:
            warnings.append(f"persona_embedding seems short: {len(embedding)} dimensions")
        
        return warnings
    
    def _validate_metadata(
        self,
        metadata: Dict[str, Any]
    ) -> List[str]:
        """Validate metadata structure"""
        errors = []
        
        required = [
            "first_post_date",
            "last_post_date",
            "last_updated",
            "data_source",
            "language"
        ]
        
        for field in required:
            if field not in metadata:
                errors.append(f"metadata missing field: {field}")
        
        # Validate date formats (basic check)
        date_fields = ["first_post_date", "last_post_date", "last_updated"]
        for field in date_fields:
            if field in metadata and metadata[field]:
                if not isinstance(metadata[field], str):
                    errors.append(f"{field} must be a string (ISO format)")
        
        return errors
    
    def validate_input_quality(
        self,
        raw_posts: List[Dict[str, Any]],
        min_posts: int = 5
    ) -> Dict[str, Any]:
        """
        Validate input post quality before processing
        
        Args:
            raw_posts: List of post JSONs from Agent 2
            min_posts: Minimum number of posts required
            
        Returns:
            Validation result with is_valid flag and details
        """
        issues = []
        
        # Check post count
        if len(raw_posts) < min_posts:
            issues.append(f"Insufficient posts: {len(raw_posts)} < {min_posts} required")
        
        # Check data completeness
        complete_posts = 0
        for i, post in enumerate(raw_posts):
            if self._is_post_complete(post):
                complete_posts += 1
        
        completeness_rate = complete_posts / len(raw_posts) if raw_posts else 0
        
        if completeness_rate < 0.7:
            issues.append(f"Low data completeness: {completeness_rate:.1%} of posts are complete")
        
        # Check for required fields across dataset
        required_fields = ["post_id", "creator_id", "media_type", "metadata"]
        for field in required_fields:
            missing_count = sum(1 for post in raw_posts if not post.get(field))
            if missing_count > len(raw_posts) * 0.2:
                issues.append(f"Field '{field}' missing in {missing_count} posts")
        
        # Calculate overall quality score
        quality_score = self._calculate_input_quality_score(
            raw_posts,
            completeness_rate,
            len(issues)
        )
        
        is_valid = len(issues) == 0 or (quality_score >= 0.5 and len(raw_posts) >= min_posts)
        
        return {
            "is_valid": is_valid,
            "score": quality_score,
            "issues": issues,
            "post_count": len(raw_posts),
            "completeness_rate": completeness_rate
        }
    
    def _is_post_complete(self, post: Dict[str, Any]) -> bool:
        """Check if a single post has all critical fields"""
        required = ["post_id", "creator_id", "media_type", "metadata"]
        has_required = all(post.get(field) for field in required)
        
        # Should have content_overview or visual_analysis
        has_analysis = any([
            post.get("content_overview"),
            post.get("visual_analysis"),
            post.get("linguistic_analysis")
        ])
        
        return has_required and has_analysis
    
    def _calculate_input_quality_score(
        self,
        raw_posts: List[Dict[str, Any]],
        completeness_rate: float,
        num_issues: int
    ) -> float:
        """
        Calculate overall input quality score (0-1)
        """
        # Base score from completeness
        base_score = completeness_rate
        
        # Penalty for insufficient data
        volume_score = min(len(raw_posts) / 20, 1.0)
        
        # Penalty for issues
        issue_penalty = min(num_issues * 0.1, 0.5)
        
        quality_score = (0.5 * base_score + 0.3 * volume_score + 0.2) - issue_penalty
        
        return max(0.0, min(1.0, quality_score))
    
    def _get_timestamp(self) -> str:
        """Get current ISO timestamp"""
        from datetime import datetime
        return datetime.utcnow().isoformat() + "Z"