"""
Persona Memory Store (UPDATED)
Stores BOTH dcpr.json AND dcpr_stats.json for incremental updates

Handles:
- DCPR storage and retrieval
- Stats storage for incremental updates
- Version management
"""

import json
import os
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class PersonaMemoryStore:
    """
    Persistent storage for creator personas AND statistics
    """
    
    def __init__(
        self,
        storage_dir: str = "./data/personas"
    ):
        """
        Initialize memory store
        
        Args:
            storage_dir: Directory for JSON storage
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"PersonaMemoryStore initialized at {storage_dir}")
    
    def save_persona_and_stats(
        self,
        creator_id: str,
        dcpr: Dict[str, Any],
        stats: Dict[str, Any],
        version: str
    ) -> bool:
        """
        Save BOTH dcpr.json AND dcpr_stats.json
        
        Args:
            creator_id: Unique creator identifier
            dcpr: Complete DCPR dictionary
            stats: Statistics dictionary for incremental updates
            version: Version identifier
            
        Returns:
            Success status
        """
        try:
            # Create creator directory
            creator_dir = self.storage_dir / creator_id
            creator_dir.mkdir(exist_ok=True)
            
            # Save DCPR JSON
            dcpr_file = creator_dir / f"dcpr_{version}.json"
            with open(dcpr_file, 'w') as f:
                json.dump(dcpr, f, indent=2)
            
            # Save DCPR as latest
            latest_dcpr_file = creator_dir / "dcpr_latest.json"
            with open(latest_dcpr_file, 'w') as f:
                json.dump(dcpr, f, indent=2)
            
            # Save Stats JSON
            stats_file = creator_dir / f"dcpr_stats_{version}.json"
            with open(stats_file, 'w') as f:
                json.dump(self._serialize_stats(stats), f, indent=2)
            
            # Save Stats as latest
            latest_stats_file = creator_dir / "dcpr_stats_latest.json"
            with open(latest_stats_file, 'w') as f:
                json.dump(self._serialize_stats(stats), f, indent=2)
            
            logger.info(f"Saved DCPR and stats for creator {creator_id}, version {version}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving persona and stats: {e}", exc_info=True)
            return False
    
    def load_latest_dcpr(
        self,
        creator_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Load latest DCPR
        
        Args:
            creator_id: Creator identifier
            
        Returns:
            DCPR dictionary or None if not found
        """
        try:
            creator_dir = self.storage_dir / creator_id
            dcpr_file = creator_dir / "dcpr_latest.json"
            
            if not dcpr_file.exists():
                logger.warning(f"DCPR not found: {creator_id}")
                return None
            
            with open(dcpr_file, 'r') as f:
                dcpr = json.load(f)
            
            logger.info(f"Loaded DCPR for creator {creator_id}")
            return dcpr
            
        except Exception as e:
            logger.error(f"Error loading DCPR: {e}", exc_info=True)
            return None
    
    def load_latest_stats(
        self,
        creator_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Load latest statistics for incremental updates
        
        Args:
            creator_id: Creator identifier
            
        Returns:
            Stats dictionary or None if not found
        """
        try:
            creator_dir = self.storage_dir / creator_id
            stats_file = creator_dir / "dcpr_stats_latest.json"
            
            if not stats_file.exists():
                logger.warning(f"Stats not found: {creator_id}")
                return None
            
            with open(stats_file, 'r') as f:
                stats = json.load(f)
            
            # Deserialize back to proper format
            stats = self._deserialize_stats(stats)
            
            logger.info(f"Loaded stats for creator {creator_id}")
            return stats
            
        except Exception as e:
            logger.error(f"Error loading stats: {e}", exc_info=True)
            return None
    
    def load_dcpr_by_version(
        self,
        creator_id: str,
        version: str
    ) -> Optional[Dict[str, Any]]:
        """Load specific version of DCPR"""
        try:
            creator_dir = self.storage_dir / creator_id
            dcpr_file = creator_dir / f"dcpr_{version}.json"
            
            if not dcpr_file.exists():
                return None
            
            with open(dcpr_file, 'r') as f:
                return json.load(f)
                
        except Exception as e:
            logger.error(f"Error loading DCPR version: {e}")
            return None
    
    def get_all_versions(
        self,
        creator_id: str
    ) -> list[str]:
        """Get all available versions for a creator"""
        try:
            creator_dir = self.storage_dir / creator_id
            
            if not creator_dir.exists():
                return []
            
            # Find all DCPR files
            dcpr_files = list(creator_dir.glob("dcpr_v*.json"))
            
            # Extract version numbers
            versions = []
            for file in dcpr_files:
                # Extract version from filename like "dcpr_v6.json"
                version = file.stem.replace("dcpr_", "")
                versions.append(version)
            
            versions.sort(reverse=True)  # Most recent first
            return versions
            
        except Exception as e:
            logger.error(f"Error getting versions: {e}")
            return []
    
    def persona_exists(
        self,
        creator_id: str
    ) -> bool:
        """Check if persona exists for creator"""
        creator_dir = self.storage_dir / creator_id
        latest_file = creator_dir / "dcpr_latest.json"
        return latest_file.exists()
    
    def stats_exist(
        self,
        creator_id: str
    ) -> bool:
        """Check if stats exist for creator"""
        creator_dir = self.storage_dir / creator_id
        stats_file = creator_dir / "dcpr_stats_latest.json"
        return stats_file.exists()
    
    def delete_persona(
        self,
        creator_id: str,
        version: Optional[str] = None
    ) -> bool:
        """Delete persona(s)"""
        try:
            creator_dir = self.storage_dir / creator_id
            
            if not creator_dir.exists():
                return False
            
            if version:
                # Delete specific version
                dcpr_file = creator_dir / f"dcpr_{version}.json"
                stats_file = creator_dir / f"dcpr_stats_{version}.json"
                
                if dcpr_file.exists():
                    dcpr_file.unlink()
                if stats_file.exists():
                    stats_file.unlink()
                    
                logger.info(f"Deleted version {version} for creator {creator_id}")
            else:
                # Delete all versions
                import shutil
                shutil.rmtree(creator_dir)
                logger.info(f"Deleted all data for creator {creator_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error deleting persona: {e}")
            return False
    
    def _serialize_stats(
        self,
        stats: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Serialize stats for JSON storage
        Converts defaultdict to dict, numpy arrays to lists
        """
        from collections import defaultdict
        import numpy as np
        
        serialized = {}
        
        for key, value in stats.items():
            if isinstance(value, defaultdict):
                # Convert defaultdict to regular dict
                serialized[key] = dict(value)
            elif isinstance(value, np.ndarray):
                # Convert numpy array to list
                serialized[key] = value.tolist()
            elif isinstance(value, list):
                # Keep lists as-is
                serialized[key] = value
            else:
                # Keep other types as-is
                serialized[key] = value
        
        return serialized
    
    def _deserialize_stats(
        self,
        stats: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Deserialize stats from JSON storage
        Converts dict back to defaultdict where needed
        """
        from collections import defaultdict
        import numpy as np
        
        # Keys that should be defaultdict(int)
        count_keys = [
            "topic_counts", "tone_counts", "emotion_counts",
            "audience_intent_counts", "lighting_tags", "camera_style_tags",
            "composition_tags", "color_palette", "music_type_tags",
            "speech_style_tags", "caption_tone_tags", "keyword_counts",
            "media_type_counts"
        ]
        
        deserialized = {}
        
        for key, value in stats.items():
            if key in count_keys and isinstance(value, dict):
                # Convert to defaultdict(int)
                deserialized[key] = defaultdict(int, value)
            elif key == "embedding_accumulator" and value is not None:
                # Convert to numpy array
                deserialized[key] = np.array(value)
            else:
                deserialized[key] = value
        
        return deserialized
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get statistics about stored personas"""
        try:
            total_creators = len(list(self.storage_dir.iterdir()))
            
            total_versions = 0
            for creator_dir in self.storage_dir.iterdir():
                if creator_dir.is_dir():
                    versions = self.get_all_versions(creator_dir.name)
                    total_versions += len(versions)
            
            return {
                "total_creators": total_creators,
                "total_versions": total_versions,
                "storage_path": str(self.storage_dir)
            }
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {}