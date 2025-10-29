import logging
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional 
import os

logger = logging.getLogger(__name__)

class EvaluationStore:
    """Saves the evaluation results (clustering and scores)."""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"EvaluationStore initialized, saving to: {self.output_dir}")

    def save_results(self, results_df: pd.DataFrame, clustering_meta: dict):
        """
        Saves the combined clustering and scoring results to a CSV file.
        Includes clustering metadata.

        Args:
            results_df (pd.DataFrame): DataFrame containing creator_id, cluster_id, scores, etc.
            clustering_meta (dict): Metadata from the clustering process.
        """
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"creator_evaluation_{timestamp}.csv"
        filepath = self.output_dir / filename
        latest_filepath = self.output_dir / "creator_evaluation_latest.csv"

        try:
            # Add clustering metadata as columns (optional, could be separate file)
            # Prefixing keys to avoid potential column name clashes
            for key, value in clustering_meta.items():
                results_df[f"meta_{key}"] = value

            results_df.to_csv(filepath, index=False)
            results_df.to_csv(latest_filepath, index=False) # Overwrite latest

            logger.info(f"Successfully saved evaluation results for {len(results_df)} creators to {filepath}")
            logger.info(f"Updated {latest_filepath}")

        except Exception as e:
            logger.error(f"Failed to save evaluation results: {e}", exc_info=True)

    def load_latest_results(self) -> Optional[pd.DataFrame]:
        """Loads the most recent evaluation results."""
        latest_filepath = self.output_dir / "creator_evaluation_latest.csv"
        if latest_filepath.exists():
            try:
                df = pd.read_csv(latest_filepath)
                logger.info(f"Loaded latest evaluation results from {latest_filepath}")
                return df
            except Exception as e:
                logger.error(f"Failed to load latest evaluation results: {e}", exc_info=True)
                return None
        else:
            logger.warning("No latest evaluation results file found.")
            return None
