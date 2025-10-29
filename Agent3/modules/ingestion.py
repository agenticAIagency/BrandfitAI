"""
Data Ingestion Module
Loads Rich JSON data from Agent 2 outputs

Handles:
- File system loading
- Data validation
- Sorting and filtering
- Error handling
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class DataIngestionModule:
    """
    Loads and preprocesses creator data for persona building
    """
    
    def __init__(
        self,
        data_root: str = "./data/creators"
    ):
        """
        Initialize ingestion module
        
        Args:
            data_root: Root directory for creator data
        """
        self.data_root = Path(data_root)
        logger.info(f"DataIngestionModule initialized with root: {data_root}")
    
    async def load_creator_data(
        self,
        creator_id: str,
        file_pattern: str = "rich_post_*.json"
    ) -> List[Dict[str, Any]]:
        """
        Load all Rich JSON files for a creator
        
        Args:
            creator_id: Unique creator identifier
            file_pattern: Glob pattern for post files
            
        Returns:
            List of Rich JSON dictionaries
        """
        creator_dir = self.data_root / creator_id
        
        if not creator_dir.exists():
            logger.error(f"Creator directory not found: {creator_dir}")
            raise FileNotFoundError(f"No data found for creator: {creator_id}")
        
        # Find all matching files
        post_files = list(creator_dir.glob(file_pattern))
        
        if not post_files:
            logger.warning(f"No post files found for creator: {creator_id}")
            return []
        
        logger.info(f"Found {len(post_files)} posts for creator {creator_id}")
        
        # Load all JSON files
        posts = []
        for file_path in post_files:
            try:
                post_data = self._load_json_file(file_path)
                if post_data:
                    posts.append(post_data)
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
                continue
        
        # Sort by timestamp
        posts = self._sort_by_timestamp(posts)
        
        logger.info(f"Successfully loaded {len(posts)} posts for creator {creator_id}")
        return posts
    
    def _load_json_file(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """
        Load and validate a single JSON file
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            Parsed JSON dictionary or None if invalid
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Basic validation
            if not isinstance(data, dict):
                logger.warning(f"Invalid JSON structure in {file_path}")
                return None
            
            return data
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error in {file_path}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
            return None
    
    def _sort_by_timestamp(
        self,
        posts: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Sort posts chronologically by timestamp
        
        Args:
            posts: List of post dictionaries
            
        Returns:
            Sorted list
        """
        try:
            from datetime import datetime
            
            def get_timestamp(post):
                ts = post.get("timestamp", "2020-01-01T00:00:00Z")
                try:
                    return datetime.fromisoformat(ts.replace('Z', '+00:00'))
                except:
                    return datetime.min
            
            return sorted(posts, key=get_timestamp)
            
        except Exception as e:
            logger.warning(f"Error sorting posts: {e}")
            return posts
    
    async def load_new_posts_only(
        self,
        creator_id: str,
        last_processed_timestamp: str
    ) -> List[Dict[str, Any]]:
        """
        Load only posts newer than last processed timestamp
        Used for incremental updates
        
        Args:
            creator_id: Creator identifier
            last_processed_timestamp: ISO timestamp of last processed post
            
        Returns:
            List of new posts
        """
        from datetime import datetime
        
        all_posts = await self.load_creator_data(creator_id)
        
        last_processed = datetime.fromisoformat(
            last_processed_timestamp.replace('Z', '+00:00')
        )
        
        new_posts = [
            post for post in all_posts
            if datetime.fromisoformat(
                post.get("timestamp", "2020-01-01").replace('Z', '+00:00')
            ) > last_processed
        ]
        
        logger.info(
            f"Found {len(new_posts)} new posts since {last_processed_timestamp}"
        )
        
        return new_posts
    
    def filter_posts_by_quality(
        self,
        posts: List[Dict[str, Any]],
        min_quality_score: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Filter out low-quality posts
        
        Args:
            posts: List of post dictionaries
            min_quality_score: Minimum quality threshold
            
        Returns:
            Filtered list
        """
        filtered = []
        
        for post in posts:
            quality = self._assess_post_quality(post)
            if quality >= min_quality_score:
                filtered.append(post)
        
        logger.info(
            f"Filtered {len(posts) - len(filtered)} low-quality posts. "
            f"{len(filtered)} remain."
        )
        
        return filtered
    
    def _assess_post_quality(self, post: Dict[str, Any]) -> float:
        """
        Assess individual post quality (0-1)
        
        Quality factors:
        - Has timestamp
        - Has content (caption, scenes, or audio)
        - Has engagement metrics
        - Completeness of analysis
        """
        quality = 0.0
        
        # Has timestamp (0.2)
        if post.get("timestamp"):
            quality += 0.2
        
        # Has content analysis (0.4)
        has_content = any([
            post.get("caption"),
            post.get("scenes"),
            post.get("audio_analysis")
        ])
        if has_content:
            quality += 0.4
        
        # Has engagement metrics (0.2)
        if post.get("engagement_metrics"):
            quality += 0.2
        
        # Analysis completeness (0.2)
        if post.get("scenes") and len(post["scenes"]) > 0:
            quality += 0.1
        if post.get("audio_analysis"):
            quality += 0.1
        
        return quality
    
    def get_data_statistics(
        self,
        posts: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Get statistics about loaded data
        
        Args:
            posts: List of post dictionaries
            
        Returns:
            Statistics dictionary
        """
        if not posts:
            return {"total_posts": 0}
        
        # Content type distribution
        content_types = {}
        for post in posts:
            ctype = post.get("content_type", "unknown")
            content_types[ctype] = content_types.get(ctype, 0) + 1
        
        # Temporal coverage
        timestamps = [
            post.get("timestamp")
            for post in posts
            if post.get("timestamp")
        ]
        
        if len(timestamps) >= 2:
            from datetime import datetime
            sorted_ts = sorted([
                datetime.fromisoformat(ts.replace('Z', '+00:00'))
                for ts in timestamps
            ])
            temporal_span = (sorted_ts[-1] - sorted_ts[0]).days
        else:
            temporal_span = 0
        
        # Average engagement
        total_likes = sum(
            post.get("engagement_metrics", {}).get("likes", 0)
            for post in posts
        )
        avg_likes = total_likes / len(posts)
        
        return {
            "total_posts": len(posts),
            "content_type_distribution": content_types,
            "temporal_span_days": temporal_span,
            "has_audio_count": sum(
                1 for post in posts
                if post.get("audio_analysis")
            ),
            "has_scenes_count": sum(
                1 for post in posts
                if post.get("scenes")
            ),
            "avg_likes": avg_likes
        }
    
    async def load_new_posts_since(
        self,
        creator_id: str,
        last_post_date: str
    ) -> List[Dict[str, Any]]:
        """
        Load only posts newer than last_post_date (for incremental updates)
        
        Args:
            creator_id: Creator identifier
            last_post_date: ISO timestamp of last processed post
            
        Returns:
            List of new posts only
        """
        from datetime import datetime
        
        # Load all posts
        all_posts = await self.load_creator_data(creator_id)
        
        if not last_post_date:
            return all_posts
        
        # Parse last_post_date
        try:
            last_date = datetime.fromisoformat(last_post_date.replace('Z', '+00:00'))
        except:
            logger.warning(f"Could not parse last_post_date: {last_post_date}")
            return all_posts
        
        # Filter posts newer than last_date
        new_posts = []
        for post in all_posts:
            post_date_str = post.get("metadata", {}).get("upload_time")
            if not post_date_str:
                continue
            
            try:
                post_date = datetime.fromisoformat(post_date_str.replace('Z', '+00:00'))
                if post_date > last_date:
                    new_posts.append(post)
            except:
                continue
        
        logger.info(
            f"Filtered {len(all_posts)} posts → {len(new_posts)} new posts "
            f"since {last_post_date}"
        )
        
        return new_posts
    
    async def load_sample_data(
        self,
        creator_id: str = "sample_creator",
        num_posts: int = 15
    ) -> List[Dict[str, Any]]:
        """
        Generate sample data in NEW Agent 2 format (with timeline_analysis for reels)
        
        Args:
            creator_id: Creator identifier
            num_posts: Number of sample posts to generate
            
        Returns:
            List of sample Post/Reel JSON dictionaries
        """
        from datetime import datetime, timedelta
        import random
        
        logger.warning("Generating sample data for testing")
        
        sample_posts = []
        base_date = datetime(2024, 8, 1)
        
        themes = ["Fitness - Back Workout", "Healthy lifestyle", "Nutrition tips", "Motivation", "Wellness"]
        tones = ["Motivational", "Educational", "Encouraging", "Energetic"]
        emotions = ["motivation", "discipline", "consistency", "confidence"]
        
        for i in range(num_posts):
            post_date = base_date + timedelta(days=i * 5)
            
            # Alternate between post and reel
            is_reel = i % 3 == 0
            
            if is_reel:
                # REEL JSON with timeline_analysis
                sample_post = {
                    "post_id": f"p{9000 + i}",
                    "creator_id": creator_id,
                    "media_type": "reel",
                    "metadata": {
                        "upload_time": post_date.isoformat() + "Z",
                        "caption": f"Killer workout today! 💪 Post {i}",
                        "hashtags": [f"#fitness", f"#motivation", f"#workout"],
                        "likes": random.randint(5000, 15000),
                        "comments": random.randint(100, 500),
                        "duration_seconds": random.uniform(15, 45)
                    },
                    "content_overview": {
                        "primary_theme": random.choice(themes),
                        "mood": random.choice(tones),
                        "audience_emotion_targeted": random.sample(emotions, 2),
                        "camera_style": random.choice(["Handheld close-ups", "Static tripod", "Dynamic angles"]),
                        "lighting": random.choice(["Bright artificial light", "Natural daylight", "Mixed lighting"]),
                        "music_type": random.choice(["Fast-tempo EDM", "Motivational background", "Voiceover"])
                    },
                    "timeline_analysis": [
                        {
                            "timestamp_start": 0,
                            "timestamp_end": 3.2,
                            "scene_summary": "Intro: creator looking at camera",
                            "tone": "Motivational",
                            "visual_intensity": 0.6
                        },
                        {
                            "timestamp_start": 3.2,
                            "timestamp_end": 18.7,
                            "scene_summary": "Demonstrates workout technique",
                            "tone": "Instructional",
                            "visual_intensity": 0.85
                        }
                    ],
                    "linguistic_analysis": {
                        "caption_tone": random.choice(tones),
                        "text_overlay_presence": random.choice([True, False])
                    },
                    "engagement_features": {
                        "engagement_rate": random.uniform(0.06, 0.12)
                    },
                    "embedding": [random.uniform(0.05, 0.25) for _ in range(5)]
                }
            else:
                # POST JSON (static image)
                sample_post = {
                    "post_id": f"p{7000 + i}",
                    "creator_id": creator_id,
                    "media_type": "post",
                    "metadata": {
                        "upload_time": post_date.isoformat() + "Z",
                        "caption": f"Fuel your morning with healthy choices 🥗 Post {i}",
                        "hashtags": [f"#healthylifestyle", f"#wellness", f"#nutrition"],
                        "likes": random.randint(3000, 10000),
                        "comments": random.randint(50, 300),
                        "saves": random.randint(100, 500)
                    },
                    "content_overview": {
                        "primary_theme": random.choice(themes),
                        "tone": random.choice(tones),
                        "audience_emotion_targeted": random.sample(emotions, 2),
                        "camera_style": random.choice(["Static overhead shot", "Flat lay", "Close-up"]),
                        "lighting": random.choice(["Natural daylight", "Soft artificial", "Bright natural"])
                    },
                    "visual_analysis": {
                        "dominant_colors": [f"#{random.randint(0, 255):02X}{random.randint(0, 255):02X}{random.randint(0, 255):02X}" for _ in range(3)],
                        "composition_type": random.choice(["Flat lay", "Rule of thirds", "Centered"]),
                        "aesthetic_score": random.uniform(7.0, 9.5)
                    },
                    "linguistic_analysis": {
                        "caption_tone": random.choice(tones),
                        "text_overlay_presence": random.choice([True, False])
                    },
                    "engagement_features": {
                        "engagement_rate": random.uniform(0.05, 0.10)
                    },
                    "embedding": [random.uniform(0.05, 0.25) for _ in range(5)]
                }
            
            sample_posts.append(sample_post)
        
        logger.info(f"Generated {num_posts} sample posts (mix of posts and reels)")
        return sample_posts