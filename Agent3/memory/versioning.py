"""
Version Manager
Handles persona versioning and history tracking

Provides:
- Version ID generation
- Version comparison
- Rollback capability
- Change tracking
"""

from datetime import datetime
from typing import Dict, Any, List, Optional
import hashlib
import json
import logging

logger = logging.getLogger(__name__)


class VersionManager:
    """
    Manages persona versions and change history
    """
    
    def __init__(self):
        """Initialize version manager"""
        self.version_format = "v{timestamp}_{hash}"
        logger.info("VersionManager initialized")
    
    def create_version(
        self,
        creator_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate unique version identifier
        
        Args:
            creator_id: Creator identifier
            metadata: Optional metadata for this version
            
        Returns:
            Version identifier string
        """
        # Timestamp-based version
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        
        # Create hash from creator_id and timestamp
        hash_input = f"{creator_id}_{timestamp}"
        version_hash = hashlib.md5(hash_input.encode()).hexdigest()[:8]
        
        version_id = f"v{timestamp}_{version_hash}"
        
        logger.info(f"Created version {version_id} for creator {creator_id}")
        return version_id
    
    def parse_version(self, version_id: str) -> Dict[str, Any]:
        """
        Parse version ID to extract timestamp and hash
        
        Args:
            version_id: Version identifier
            
        Returns:
            Dictionary with parsed components
        """
        try:
            # Remove 'v' prefix
            version_parts = version_id[1:].split('_')
            
            if len(version_parts) >= 3:
                date_part = version_parts[0]
                time_part = version_parts[1]
                hash_part = version_parts[2]
                
                # Parse timestamp
                timestamp_str = f"{date_part}_{time_part}"
                timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                
                return {
                    "version_id": version_id,
                    "timestamp": timestamp,
                    "date": date_part,
                    "time": time_part,
                    "hash": hash_part,
                    "iso_timestamp": timestamp.isoformat()
                }
            else:
                return {"version_id": version_id, "valid": False}
                
        except Exception as e:
            logger.warning(f"Error parsing version {version_id}: {e}")
            return {"version_id": version_id, "valid": False}
    
    def compare_versions(
        self,
        persona_v1: Dict[str, Any],
        persona_v2: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compare two persona versions to identify changes
        
        Args:
            persona_v1: First persona version
            persona_v2: Second persona version
            
        Returns:
            Dictionary describing differences
        """
        changes = {
            "quantitative_changes": {},
            "qualitative_changes": {},
            "significant_changes": []
        }
        
        # Compare quantitative metrics
        if "quantitative_metrics" in persona_v1 and "quantitative_metrics" in persona_v2:
            metrics_v1 = persona_v1["quantitative_metrics"]
            metrics_v2 = persona_v2["quantitative_metrics"]
            
            for metric_name in metrics_v1.keys():
                if metric_name in metrics_v2:
                    value_v1 = metrics_v1[metric_name]
                    value_v2 = metrics_v2[metric_name]
                    
                    if isinstance(value_v1, (int, float)) and isinstance(value_v2, (int, float)):
                        change = value_v2 - value_v1
                        change_pct = (change / value_v1 * 100) if value_v1 != 0 else 0
                        
                        changes["quantitative_changes"][metric_name] = {
                            "old_value": value_v1,
                            "new_value": value_v2,
                            "absolute_change": change,
                            "percent_change": change_pct
                        }
                        
                        # Flag significant changes (>10%)
                        if abs(change_pct) > 10:
                            changes["significant_changes"].append({
                                "metric": metric_name,
                                "change_pct": change_pct,
                                "type": "increase" if change > 0 else "decrease"
                            })
        
        # Compare qualitative aspects
        qualitative_fields = [
            ("creator_summary", "headline"),
            ("content_profile", "content_quality"),
            ("growth_trajectory", "current_stage"),
            ("growth_trajectory", "momentum")
        ]
        
        for section, field in qualitative_fields:
            val_v1 = persona_v1.get(section, {}).get(field)
            val_v2 = persona_v2.get(section, {}).get(field)
            
            if val_v1 != val_v2:
                changes["qualitative_changes"][f"{section}.{field}"] = {
                    "old_value": val_v1,
                    "new_value": val_v2
                }
        
        # Summary
        changes["summary"] = {
            "num_quantitative_changes": len(changes["quantitative_changes"]),
            "num_qualitative_changes": len(changes["qualitative_changes"]),
            "num_significant_changes": len(changes["significant_changes"]),
            "has_major_changes": len(changes["significant_changes"]) > 0
        }
        
        return changes
    
    def get_version_history(
        self,
        all_versions: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Get chronologically sorted version history
        
        Args:
            all_versions: List of version IDs
            
        Returns:
            Sorted list of version info dictionaries
        """
        version_info = []
        
        for version_id in all_versions:
            parsed = self.parse_version(version_id)
            if parsed.get("valid", True):
                version_info.append(parsed)
        
        # Sort by timestamp (newest first)
        version_info.sort(
            key=lambda x: x.get("timestamp", datetime.min),
            reverse=True
        )
        
        return version_info
    
    def should_create_new_version(
        self,
        old_persona: Dict[str, Any],
        new_data_posts: int,
        time_since_last_version: int
    ) -> bool:
        """
        Determine if a new version should be created
        
        Args:
            old_persona: Existing persona
            new_data_posts: Number of new posts since last version
            time_since_last_version: Days since last version
            
        Returns:
            True if new version should be created
        """
        # Create new version if:
        # 1. Significant new data (10+ new posts)
        if new_data_posts >= 10:
            return True
        
        # 2. Long time since last update (30+ days)
        if time_since_last_version >= 30:
            return True
        
        # 3. Low confidence in existing persona
        if old_persona.get("confidence_score", 1.0) < 0.6:
            return True
        
        return False
    
    def create_changelog(
        self,
        changes: Dict[str, Any]
    ) -> str:
        """
        Generate human-readable changelog
        
        Args:
            changes: Output from compare_versions()
            
        Returns:
            Formatted changelog string
        """
        lines = ["# Persona Update Changelog\n"]
        
        # Significant changes
        if changes["significant_changes"]:
            lines.append("## Significant Changes")
            for change in changes["significant_changes"]:
                direction = "↑" if change["type"] == "increase" else "↓"
                lines.append(
                    f"- {direction} **{change['metric']}**: "
                    f"{change['change_pct']:+.1f}%"
                )
            lines.append("")
        
        # Quantitative changes
        if changes["quantitative_changes"]:
            lines.append("## Metric Updates")
            for metric, data in changes["quantitative_changes"].items():
                lines.append(
                    f"- **{metric}**: {data['old_value']:.3f} → "
                    f"{data['new_value']:.3f} ({data['percent_change']:+.1f}%)"
                )
            lines.append("")
        
        # Qualitative changes
        if changes["qualitative_changes"]:
            lines.append("## Profile Updates")
            for field, data in changes["qualitative_changes"].items():
                lines.append(
                    f"- **{field}**: \"{data['old_value']}\" → \"{data['new_value']}\""
                )
            lines.append("")
        
        return "\n".join(lines)