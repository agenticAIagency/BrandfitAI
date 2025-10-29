import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import ast
import operator
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

class MatchingEngine:
    """
    Performs semantic search, filtering, and weighted ranking on creator data.
    Uses CSV output from Agent 4 as the queryable data source.
    """

    def __init__(self, config: Dict[str, Any], data_dir: str):
        # Configuration for the matching logic
        self.config = config.get("matching", {})
        self.ranking_weights = self.config.get("ranking_weights", {"similarity": 0.4, "quality": 0.4, "performance": 0.2})
        self.TOP_N = self.config.get("top_n_candidates", 5)
        
        # Initialize embedding model (must be the same model used by Agent 4)
        self.EMBEDDING_MODEL_NAME = self.config.get("embedding_model", "all-MiniLM-L6-v2")
        self.model = SentenceTransformer(self.EMBEDDING_MODEL_NAME)
        logger.info(f"Loaded matching embedding model: {self.EMBEDDING_MODEL_NAME}")


    def _get_query_vector(self, query: str) -> np.ndarray:
        """Generates the embedding vector for the user's text query."""
        # Wrap sync model call in a thread to prevent blocking if run in async context
        query_vector = self.model.encode(query, convert_to_numpy=True)
        # Reshape to a 2D array [1, D] for similarity calculation against the matrix
        return query_vector.reshape(1, -1)
    
    
    def _get_vectors_and_matrix(self, results_df: pd.DataFrame, vector_col: str) -> Optional[np.ndarray]:
        """
        Extracts and verifies the vector data from the DataFrame column and converts it to a NumPy matrix.
        
        Args:
            results_df: DataFrame loaded from Agent 4's CSV.
            vector_col: The name of the column containing the vector lists (e.g., 'brand_fit_vector').
            
        Returns:
            A NumPy array (matrix) of the vectors, or None if invalid.
        """
        # Ensure the column exists and is not entirely null
        if vector_col not in results_df.columns or results_df[vector_col].isnull().all():
            return None

        # Filter out invalid entries and use the already-parsed lists
        valid_vectors = results_df[vector_col].dropna().tolist()
        
        if not valid_vectors:
            return None
        
        # Convert list of lists (Python) into a single NumPy matrix
        # This assumes the input data is a list of lists of floats, which the converter handled.
        return np.array(valid_vectors)


    def _apply_semantic_search(self, query_vector: np.ndarray, results_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates cosine similarity between the query and all creator brand_fit_vectors.
        
        Args:
            query_vector: The embedded user query vector.
            results_df: DataFrame containing the brand_fit_vector column.
            
        Returns:
            DataFrame with a new 'similarity_score' column.
        """
        # 1. Extract and format the brand_fit_vectors from the DataFrame
        # IMPORTANT: We only calculate similarity for rows that have a vector.
        
        # Get the subset of the DataFrame that has valid vectors
        df_valid_vectors = results_df[results_df['brand_fit_vector'].notna()].copy()

        creator_vectors = self._get_vectors_and_matrix(df_valid_vectors, 'brand_fit_vector')
        
        if creator_vectors is None:
            logger.warning("No valid brand_fit_vectors found in data for semantic search.")
            results_df['similarity_score'] = 0.0 # Assign zero similarity if vectors are missing
            return results_df
            
        # 2. Calculate Cosine Similarity
        # cosine_similarity returns a matrix [N_creators, N_queries], we take the first (only) column
        similarity_scores = cosine_similarity(creator_vectors, query_vector).flatten()
        
        # 3. Add scores back to the original DataFrame using the correct index
        # Create a Series of scores using the index of the valid rows
        scores_series = pd.Series(similarity_scores, index=df_valid_vectors.index)
        
        # Merge this Series back into the original results_df
        # Fill missing values (for rows that had no vector) with 0.0
        results_df['similarity_score'] = scores_series.reindex(results_df.index, fill_value=0.0)

        return results_df


    def _apply_hard_filters(self, df: pd.DataFrame, constraints: Dict[str, Any]) -> pd.DataFrame:
        """Applies non-negotiable constraints based on the parsed query."""
        
        filtered_df = df.copy()
        
        # 1. Follower Max Constraint
        follower_max = constraints.get('follower_max')
        if follower_max is not None:
            logger.debug(f"Applying filter: follower_count <= {follower_max}")
            filtered_df = filtered_df[filtered_df['raw_follower_count'] <= follower_max]
            
        # 2. Quality Score Minimum
        min_quality = constraints.get('min_quality_score')
        if min_quality is not None:
            # Note: quality_score is 0-100 scale from Agent 4's config
            logger.debug(f"Applying filter: quality_score >= {min_quality}")
            filtered_df = filtered_df[filtered_df['quality_score'] >= min_quality]
            
        # 3. Authenticity Minimum
        min_auth = constraints.get('min_authenticity_score')
        if min_auth is not None:
            # Note: raw_authenticity is 0-1 scale
            logger.debug(f"Applying filter: raw_authenticity >= {min_auth}")
            filtered_df = filtered_df[filtered_df['raw_authenticity'] >= min_auth]
            
        # 4. Thematic Match (MVP: Placeholder check against a generic column, if one existed)
        # We assume the embedding match handles theme, but a robust system would filter here using DCPR keywords.
        
        logger.info(f"Hard Filtering reduced candidates from {len(df)} to {len(filtered_df)}")
        return filtered_df

    def _rank_and_select(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Calculates the final weighted score and selects the top N candidates."""
        
        # Normalize scores (Quality and Performance are already 0-100, normalize them to 0-1)
        # Note: Avoid division by zero if score_scale_max is 0, but assuming 100 here.
        df['quality_norm'] = df['quality_score'] / 100.0
        df['performance_norm'] = df['performance_score'] / 100.0
        
        # Ensure similarity score is present (it might be 0.0 if vectors failed)
        if 'similarity_score' not in df.columns:
            df['similarity_score'] = 0.0

        # Calculate Final Match Score (Weighted Rank Formula)
        final_score = (
            self.ranking_weights['similarity'] * df['similarity_score'] +
            self.ranking_weights['quality'] * df['quality_norm'] +
            self.ranking_weights['performance'] * df['performance_norm']
        )
        
        df['final_match_score'] = final_score
        
        # Select top N candidates
        top_candidates = df.sort_values(by='final_match_score', ascending=False).head(self.TOP_N)
        
        logger.info(f"Ranked and selected {len(top_candidates)} candidates.")
        
        # Convert to list of dictionaries for use by the email module
        # Select key columns only
        output_cols = ['creator_id', 'final_match_score', 'quality_score', 'performance_score', 'similarity_score', 'raw_follower_count', 'raw_authenticity']
        return top_candidates[output_cols].to_dict(orient='records')


    def run_matching_workflow(self, query: str, results_df: pd.DataFrame, constraints: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Orchestrates the entire matching process.
        """
        logger.info(f"Starting matching for query: '{query[:50]}...'")

        # 1. Generate Query Vector
        query_vector = self._get_query_vector(query)

        # 2. Semantic Search (Adds 'similarity_score' column)
        df_with_sim = self._apply_semantic_search(query_vector, results_df.copy())
        
        # 3. Hard Filtering (Filter-First Strategy)
        df_filtered = self._apply_hard_filters(df_with_sim, constraints)
        
        if df_filtered.empty:
            logger.warning("No candidates remain after hard filtering.")
            return []
            
        # 4. Weighted Ranking and Selection
        final_candidates = self._rank_and_select(df_filtered)
        
        return final_candidates
