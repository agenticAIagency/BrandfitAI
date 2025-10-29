import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class DCPR_Ingestion:
    """Loads DCPR JSON files output by Agent 3."""

    def __init__(self, dcpr_dir: str):
        self.dcpr_root_dir = Path(dcpr_dir)
        if not self.dcpr_root_dir.exists():
            logger.warning(f"DCPR input directory not found: {self.dcpr_root_dir}")
        logger.info(f"DCPR Ingestion initialized, reading from: {self.dcpr_root_dir}")

    async def load_latest_dcprs(self) -> List[Dict[str, Any]]:
        """
        Scans the input directory, finds creator subdirectories,
        and loads the 'dcpr_latest.json' from each.
        """
        dcprs = []
        creator_dirs = 0
        loaded_count = 0

        if not self.dcpr_root_dir.exists():
             logger.error(f"Cannot load DCPRs, directory does not exist: {self.dcpr_root_dir}")
             return []

        for creator_dir in self.dcpr_root_dir.iterdir():
            if creator_dir.is_dir():
                creator_dirs += 1
                latest_dcpr_file = creator_dir / "dcpr_latest.json"
                if latest_dcpr_file.exists():
                    try:
                        with open(latest_dcpr_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            # Basic validation - ensure essential fields exist
                            if "creator_id" in data and "persona_embedding" in data and "performance_intelligence" in data:
                                dcprs.append(data)
                                loaded_count += 1
                            else:
                                logger.warning(f"Skipping invalid DCPR for {creator_dir.name}: Missing essential fields.")
                    except json.JSONDecodeError:
                        logger.error(f"Failed to decode JSON for {latest_dcpr_file}")
                    except Exception as e:
                        logger.error(f"Error loading {latest_dcpr_file}: {e}")
                else:
                    logger.warning(f"No dcpr_latest.json found for creator: {creator_dir.name}")

        logger.info(f"Scan complete. Found {creator_dirs} creator directories, loaded {loaded_count} valid DCPRs.")
        return dcprs
