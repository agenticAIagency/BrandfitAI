import logging
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
import asyncio
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sentence_transformers import SentenceTransformer
from typing import Optional

logger = logging.getLogger(__name__)

class CreatorClustering:
    """Performs clustering on creators based on generated embeddings."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config.get("clustering", {})
        self.embedding_config = config.get("embeddings", {})
        self.algorithm = self.config.get("algorithm", "kmeans")
        self.min_creators = self.config.get("min_creators_for_clustering", 10)
        self.embedding_model_name = self.embedding_config.get("model", "all-MiniLM-L6-v2")
        self.embedding_dim = 384 # Default dimension for fallback model

        # Initialize Embedding Model
        self.embedding_model = self._initialize_embedding_model()
        if self.embedding_model:
            self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()

        logger.info(f"Creator Clustering initialized with algorithm: {self.algorithm}")

    def _initialize_embedding_model(self) -> Optional[SentenceTransformer]:
        """Loads the Sentence Transformer model, with fallback if the primary fails."""
        try:
            # Note: Using all-MiniLM-L6-v2 as a fallback/default as the specific Gemma-300m might be hard to load in all environments.
            # In a real setup, this would point to the specific Gemma model path/name.
            model = SentenceTransformer(self.embedding_model_name)
            logger.info(f"Loaded embedding model: {self.embedding_model_name}")
            return model
        except Exception as e:
            logger.error(f"Failed to load primary embedding model '{self.embedding_model_name}': {e}")
            # Fallback to a highly reliable, small model
            try:
                fallback_model_name = "all-MiniLM-L6-v2"
                model = SentenceTransformer(fallback_model_name)
                logger.warning(f"Using fallback embedding model: {fallback_model_name}")
                return model
            except Exception as fe:
                logger.error(f"FATAL: Failed to load fallback embedding model: {fe}")
                return None

    def _generate_embeddings(self, dcprs: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Generates clustering_embedding and brand_fit_vector from DCPR text fields.
        """
        if not self.embedding_model:
            logger.error("Embedding model failed to initialize. Cannot generate vectors.")
            return pd.DataFrame()

        records = []
        for dcpr in dcprs:
            creator_id = dcpr.get("creator_id", "unknown")
            identity = dcpr.get("identity_summary", {})
            content = dcpr.get("content_intelligence", {})
            style = dcpr.get("style_intelligence", {})
            perf = dcpr.get("performance_intelligence", {})

            # --- Text 1: For General Clustering (Content & Style Focus) ---
            clustering_text = (
                f"{identity.get('short_summary', '')} "
                f"Keywords: {', '.join(identity.get('persona_keywords', []))}. "
                f"Topics: {', '.join(content.get('dominant_topics', []))}. "
                f"Tones: {', '.join(content.get('tone_distribution', {}).keys())}. "
                f"Visual Style: {', '.join(style.get('visual_signatures', {}).get('lighting', []))} / {', '.join(style.get('visual_signatures', {}).get('camera_styles', []))}. "
                f"Performance: Auth {perf.get('authenticity_score', 0):.2f}, Cons {perf.get('consistency_score', 0):.2f}."
            )

            # --- Text 2: For Brand Fit (Relevance & Performance Focus) ---
            brand_fit_text = (
                f"Brand Alignment Profile: {', '.join(content.get('audience_intent_targeted', []))}. "
                f"Keywords: {', '.join(identity.get('persona_keywords', []))}. "
                f"Authenticity Score: {perf.get('authenticity_score', 0):.2f}. "
                f"Engagement Rate: {perf.get('avg_engagement_rate', 0):.3f}. "
                f"Audience Sentiment: {perf.get('audience_sentiment', 'Neutral')}."
            )
            
            # Encode vectors (sync operation, suitable for to_thread wrapping)
            clustering_emb = self.embedding_model.encode(clustering_text, normalize_embeddings=True)
            brand_fit_emb = self.embedding_model.encode(brand_fit_text, normalize_embeddings=True)
            
            records.append({
                "creator_id": creator_id,
                "clustering_embedding": clustering_emb.tolist(), # Convert to list for DataFrame/CSV
                "brand_fit_vector": brand_fit_emb.tolist(),
                "embedding_text_1": clustering_text,
                "embedding_text_2": brand_fit_text,
            })

        logger.info(f"Generated embeddings for {len(records)} creators.")
        return pd.DataFrame(records)


    def cluster_creators(self, dcprs: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Clusters creators using the configured algorithm after generating embeddings.
        """
        if len(dcprs) < self.min_creators:
            logger.warning(f"Not enough creators ({len(dcprs)}) to perform clustering (minimum {self.min_creators}). Skipping.")
            df = pd.DataFrame({'creator_id': [d['creator_id'] for d in dcprs], 'cluster_id': -1}) 
            return df, {"status": "skipped_low_count", "k_used": 0, "algorithm": self.algorithm}

        # 1. Generate Embeddings
        # NOTE: This sync task is meant to be run via asyncio.to_thread in agent4_graph.py
        embeddings_df = self._generate_embeddings(dcprs)
        
        if embeddings_df.empty:
             df = pd.DataFrame({'creator_id': [d['creator_id'] for d in dcprs], 'cluster_id': -1})
             return df, {"status": "error_no_embeddings", "k_used": 0, "algorithm": self.algorithm}

        # Extract vectors for clustering
        embedding_matrix = np.array(embeddings_df['clustering_embedding'].tolist())
        
        logger.info(f"Performing clustering on {len(embedding_matrix)} creators with valid embeddings.")
        
        # 2. Perform Clustering (KMeans MVP)
        clustering_meta = {"algorithm": self.algorithm}
        k = self._determine_kmeans_k(embedding_matrix)

        if k <= 1:
             logger.warning("Clustering determined K=1 or failed. Assigning all to cluster 0.")
             cluster_labels = np.zeros(len(embedding_matrix), dtype=int)
             clustering_meta["status"] = "warning_k_determination_failed"
             clustering_meta["k_used"] = 1
        else:
             kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
             cluster_labels = kmeans.fit_predict(embedding_matrix)
             clustering_meta["status"] = "success"
             clustering_meta["k_used"] = k
             clustering_meta["inertia"] = kmeans.inertia_

        # 3. Combine results
        embeddings_df['cluster_id'] = cluster_labels
        
        # Ensure all original creators are present, even if embedding failed (they will have cluster_id -1)
        all_creator_ids_df = pd.DataFrame({'creator_id': [d['creator_id'] for d in dcprs]})
        final_df = pd.merge(all_creator_ids_df, embeddings_df, on='creator_id', how='left')
        
        # Fill missing values: Cluster ID -1, Embeddings as empty lists
        final_df['cluster_id'] = final_df['cluster_id'].fillna(-1).astype(int)
        
        # Final cleanup for missing vectors (if embedding failed)
        if final_df['clustering_embedding'].isnull().any():
             logger.warning("One or more creators resulted in a null embedding, filling with zeros for safety.")
             # This is important for Agent 5 if it tries to load all vectors
             zero_vector = np.zeros(self.embedding_dim).tolist()
             final_df['clustering_embedding'] = final_df['clustering_embedding'].apply(lambda x: x if isinstance(x, list) else zero_vector)
             final_df['brand_fit_vector'] = final_df['brand_fit_vector'].apply(lambda x: x if isinstance(x, list) else zero_vector)


        logger.info(f"Clustering complete. Assigned {len(set(final_df['cluster_id'])) - (1 if -1 in final_df['cluster_id'].unique() else 0)} clusters.")
        return final_df, clustering_meta

    def _determine_kmeans_k(self, data: np.ndarray) -> int:
        """Determines the optimal K for KMeans using configured method."""
        method = self.config.get("kmeans_k_method", "elbow")
        max_k = self.config.get("kmeans_max_k", 15)
        fixed_k = self.config.get("kmeans_fixed_k", 5)
        
        max_k = min(max_k, len(data) - 1) 
        if max_k < 2:
             return 1

        if isinstance(method, int): return method
        if method == "fixed": return fixed_k

        k_range = range(2, max_k + 1)
        
        if method == "elbow":
            inertias = []
            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(data)
                inertias.append(kmeans.inertia_)
            
            # Simple elbow heuristic (requires optimization, safe defaults used here)
            try:
                diff1 = np.diff(inertias)
                diff2 = np.diff(diff1)
                elbow_index = np.argmax(diff2) + 2
                optimal_k = k_range[elbow_index - 2]
                return optimal_k
            except Exception:
                 return max(2, min(fixed_k, max_k))

        elif method == "silhouette":
            silhouette_scores = []
            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(data)
                if len(set(labels)) < 2: continue
                score = silhouette_score(data, labels)
                silhouette_scores.append(score)
            
            if not silhouette_scores:
                 return max(2, min(fixed_k, max_k))
                 
            optimal_k_index = np.argmax(silhouette_scores)
            optimal_k = k_range[optimal_k_index]
            return optimal_k

        return fixed_k
