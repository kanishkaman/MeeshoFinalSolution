"""
Semantic Candidate Generation with Sentence-BERT and FAISS
Meaning-based query matching using dense vector embeddings
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from typing import List, Tuple
from tqdm import tqdm

# ============================================================================
# SEMANTIC CANDIDATE GENERATOR
# ============================================================================

class SemanticCandidateGenerator:
    """
    Semantic candidate generation using Sentence-BERT embeddings and FAISS.
    Provides meaning-based search for auto-complete systems.
    """
    
    def __init__(self, queries: pd.Series, model_name: str = 'all-MiniLM-L6-v2', 
                 batch_size: int = 64):
        """
        Initialize semantic search index with Sentence-BERT and FAISS.
        
        Args:
            queries: pandas Series containing query strings
            model_name: Name of the Sentence-BERT model (default: 'all-MiniLM-L6-v2')
            batch_size: Batch size for encoding (default: 64)
        """
        self.queries = queries.values  # Store as numpy array for faster access
        self.model_name = model_name
        self.batch_size = batch_size
        
        print(f"Loading Sentence-BERT model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        print(f"✓ Model loaded successfully!")
        print(f"  Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
        
        # Generate embeddings for all queries
        print(f"\nGenerating embeddings for {len(queries):,} queries...")
        self.embeddings = self._generate_embeddings(self.queries)
        print(f"✓ Embeddings generated!")
        print(f"  Shape: {self.embeddings.shape}")
        
        # Build FAISS index
        print(f"\nBuilding FAISS index (IndexFlatL2)...")
        self.dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(self.embeddings.astype('float32'))
        print(f"✓ FAISS index built successfully!")
        print(f"  Total vectors indexed: {self.index.ntotal:,}")
    
    def _generate_embeddings(self, texts: np.ndarray) -> np.ndarray:
        """
        Generate embeddings for a list of texts using Sentence-BERT.
        
        Args:
            texts: Array of text strings
        
        Returns:
            numpy array of embeddings (shape: [num_texts, embedding_dim])
        """
        # Convert texts to list and handle any None/NaN values
        text_list = [str(text) if pd.notna(text) else "" for text in texts]
        
        # Generate embeddings with progress bar
        embeddings = self.model.encode(
            text_list,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=False  # We use L2 distance, not cosine
        )
        
        return embeddings
    
    def get_candidates(self, prefix: str, n: int = 300) -> List[Tuple[str, float]]:
        """
        Get top n semantically similar candidate queries for a given prefix.
        
        Args:
            prefix: Input prefix string
            n: Number of top candidates to return (default: 300)
        
        Returns:
            List of tuples (query, distance) sorted by similarity (ascending distance)
        """
        # Encode the prefix
        prefix_embedding = self.model.encode(
            [prefix],
            convert_to_numpy=True,
            normalize_embeddings=False
        )
        
        # Search FAISS index for nearest neighbors
        top_n = min(n, self.index.ntotal)
        distances, indices = self.index.search(
            prefix_embedding.astype('float32'), 
            top_n
        )
        
        # Convert to list of (query, distance) tuples
        candidates = [
            (self.queries[idx], float(dist))
            for idx, dist in zip(indices[0], distances[0])
        ]
        
        return candidates
    
    def get_candidate_queries(self, prefix: str, n: int = 300) -> List[str]:
        """
        Get top n semantically similar candidate queries (without distances).
        
        Args:
            prefix: Input prefix string
            n: Number of top candidates to return (default: 300)
        
        Returns:
            List of query strings sorted by semantic similarity
        """
        candidates = self.get_candidates(prefix, n=n)
        return [query for query, distance in candidates]
    
    def batch_get_candidates(self, prefixes: List[str], n: int = 300) -> List[List[str]]:
        """
        Get candidates for multiple prefixes in batch (more efficient).
        
        Args:
            prefixes: List of input prefix strings
            n: Number of top candidates to return per prefix (default: 300)
        
        Returns:
            List of candidate lists, one for each prefix
        """
        # Encode all prefixes at once
        prefix_embeddings = self.model.encode(
            prefixes,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=False
        )
        
        # Search for all prefixes
        top_n = min(n, self.index.ntotal)
        distances, indices = self.index.search(
            prefix_embeddings.astype('float32'),
            top_n
        )
        
        # Convert to list of candidate lists
        all_candidates = []
        for batch_indices in indices:
            candidates = [self.queries[idx] for idx in batch_indices]
            all_candidates.append(candidates)
        
        return all_candidates


# ============================================================================
# CONVENIENCE FUNCTION FOR DIRECT USE
# ============================================================================

# Global variable to store the semantic generator instance
_semantic_generator = None

def initialize_semantic_index(query_pool_df: pd.DataFrame, 
                             model_name: str = 'all-MiniLM-L6-v2',
                             batch_size: int = 64):
    """
    Initialize the global semantic search index from a query pool DataFrame.
    
    Args:
        query_pool_df: DataFrame containing a 'query' column
        model_name: Name of the Sentence-BERT model (default: 'all-MiniLM-L6-v2')
        batch_size: Batch size for encoding (default: 64)
    
    Returns:
        SemanticCandidateGenerator instance
    """
    global _semantic_generator
    
    if 'query' not in query_pool_df.columns:
        raise ValueError("DataFrame must contain a 'query' column")
    
    _semantic_generator = SemanticCandidateGenerator(
        query_pool_df['query'],
        model_name=model_name,
        batch_size=batch_size
    )
    
    return _semantic_generator


def get_semantic_candidates(prefix: str, n: int = 300) -> List[str]:
    """
    Get top n semantically similar candidates for a given prefix.
    
    This is a convenience function that uses the globally initialized semantic index.
    Call initialize_semantic_index() first to set up the index.
    
    Args:
        prefix: Input prefix string
        n: Number of top candidates to return (default: 300)
    
    Returns:
        List of candidate query strings sorted by semantic similarity
    
    Example:
        >>> initialize_semantic_index(query_pool_df)
        >>> candidates = get_semantic_candidates("clothing for ladies", n=10)
        >>> print(candidates[:5])
        ['dress for women', 'women clothing', 'ladies wear', 'women fashion', 'female apparel']
    """
    global _semantic_generator
    
    if _semantic_generator is None:
        raise RuntimeError(
            "Semantic index not initialized. Call initialize_semantic_index() first."
        )
    
    return _semantic_generator.get_candidate_queries(prefix, n=n)


# ============================================================================
# EXAMPLE USAGE AND TESTING
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("SEMANTIC CANDIDATE GENERATION - DEMO")
    print("=" * 80)
    
    # Load query pool
    print("\nLoading query pool...")
    query_pool_df = pd.read_parquet('query_pool.parquet')
    print(f"✓ Loaded {len(query_pool_df):,} queries")
    
    # Initialize semantic index with Sentence-BERT and FAISS
    print("\n" + "-" * 80)
    generator = initialize_semantic_index(
        query_pool_df, 
        model_name='all-MiniLM-L6-v2',
        batch_size=64
    )
    
    # Test with sample prefixes (semantic variations)
    print("\n" + "=" * 80)
    print("TESTING SEMANTIC CANDIDATE GENERATION")
    print("=" * 80)
    
    test_prefixes = [
        "dress for women",           # Standard query
        "clothing for ladies",       # Semantic variation
        "footwear",                  # General term
        "shoes for men",             # Specific query
        "traditional indian wear",   # Descriptive phrase
        "wedding outfit",            # Context-based
        "summer clothes",            # Seasonal
        "kurta",                     # Specific item
    ]
    
    for prefix in test_prefixes:
        print(f"\n{'-' * 80}")
        print(f"Prefix: '{prefix}'")
        print(f"{'-' * 80}")
        
        # Get top 10 candidates using the convenience function
        candidates = get_semantic_candidates(prefix, n=10)
        
        print(f"Top 10 semantically similar candidates:")
        for i, query in enumerate(candidates, 1):
            print(f"  {i:2d}. {query}")
        
        # Get candidates with distances for analysis
        candidates_with_distances = generator.get_candidates(prefix, n=10)
        print(f"\nTop 3 with L2 distances:")
        for i, (query, distance) in enumerate(candidates_with_distances[:3], 1):
            print(f"  {i}. {query:40s} (distance: {distance:.4f})")
    
    print("\n" + "=" * 80)
    print("SEMANTIC SIMILARITY COMPARISON")
    print("=" * 80)
    
    # Compare semantic vs lexical matching
    test_pairs = [
        ("dress", "gown"),                    # Synonyms
        ("shoes", "footwear"),                # Hypernym
        ("kurti", "kurta"),                   # Spelling variations
        ("saree", "sari"),                    # Alternative spellings
        ("women dress", "dress for women"),   # Word order
    ]
    
    print("\nComparing semantic similarity between query pairs:")
    print(f"{'Query 1':<25} | {'Query 2':<25} | L2 Distance")
    print("-" * 80)
    
    for q1, q2 in test_pairs:
        # Encode both queries
        embeddings = generator.model.encode([q1, q2], convert_to_numpy=True)
        
        # Calculate L2 distance
        distance = np.linalg.norm(embeddings[0] - embeddings[1])
        
        print(f"{q1:<25} | {q2:<25} | {distance:.4f}")
    
    print("\n" + "=" * 80)
    print("PERFORMANCE STATISTICS")
    print("=" * 80)
    
    print(f"\nModel: {generator.model_name}")
    print(f"Embedding dimension: {generator.dimension}")
    print(f"Total queries indexed: {generator.index.ntotal:,}")
    print(f"Index type: IndexFlatL2 (exact search)")
    print(f"Memory footprint: ~{generator.embeddings.nbytes / (1024**2):.2f} MB")
    
    # Test batch processing
    print("\n" + "-" * 80)
    print("Testing batch candidate generation...")
    batch_prefixes = ["dress", "shoes", "kurti"]
    batch_candidates = generator.batch_get_candidates(batch_prefixes, n=5)
    
    print(f"\nBatch results for {len(batch_prefixes)} prefixes:")
    for prefix, candidates in zip(batch_prefixes, batch_candidates):
        print(f"\n'{prefix}': {candidates[:3]}...")
    
    print("\n" + "=" * 80)
    print("✓ SEMANTIC CANDIDATE GENERATION READY!")
    print("=" * 80)
    print("\nUsage:")
    print("  1. Load query pool: query_pool_df = pd.read_parquet('query_pool.parquet')")
    print("  2. Initialize index: initialize_semantic_index(query_pool_df)")
    print("  3. Get candidates: candidates = get_semantic_candidates('prefix', n=300)")
    print("\nNote: First run will download the Sentence-BERT model (~80MB)")
