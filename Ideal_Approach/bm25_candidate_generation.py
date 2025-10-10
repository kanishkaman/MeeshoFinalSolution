"""
BM25 Candidate Generation with Character N-grams
Typo-resistant lexical search using character-level trigrams
"""

import pandas as pd
import numpy as np
from rank_bm25 import BM25Okapi
from typing import List, Tuple

# ============================================================================
# CHARACTER N-GRAM TOKENIZER
# ============================================================================

def char_ngram_tokenizer(text: str, n: int = 3) -> List[str]:
    """
    Convert text into character n-grams for typo-resistant matching.
    
    Args:
        text: Input string to tokenize
        n: Size of character n-grams (default: 3 for trigrams)
    
    Returns:
        List of character n-grams
    
    Example:
        >>> char_ngram_tokenizer("dress", n=3)
        ['dre', 'res', 'ess']
    """
    if not text or not isinstance(text, str):
        return []
    
    # Convert to lowercase for case-insensitive matching
    text = text.lower().strip()
    
    # Handle short strings
    if len(text) < n:
        return [text]
    
    # Generate character n-grams
    ngrams = []
    for i in range(len(text) - n + 1):
        ngrams.append(text[i:i+n])
    
    return ngrams


# ============================================================================
# BM25 INDEX BUILDER
# ============================================================================

class BM25CandidateGenerator:
    """
    BM25-based candidate generation with character n-gram tokenization.
    Provides typo-resistant lexical search for auto-complete systems.
    """
    
    def __init__(self, queries: pd.Series, n: int = 3):
        """
        Initialize BM25 index with character n-gram tokenization.
        
        Args:
            queries: pandas Series containing query strings
            n: Size of character n-grams (default: 3 for trigrams)
        """
        self.queries = queries.values  # Store as numpy array for faster access
        self.n = n
        
        print(f"Building BM25 index with {n}-grams for {len(queries):,} queries...")
        
        # Tokenize all queries into character n-grams
        self.tokenized_queries = [
            char_ngram_tokenizer(query, n=n) 
            for query in self.queries
        ]
        
        # Build BM25 index
        self.bm25 = BM25Okapi(self.tokenized_queries)
        
        print(f"✓ BM25 index built successfully!")
    
    def get_candidates(self, prefix: str, n: int = 300) -> List[Tuple[str, float]]:
        """
        Get top n candidate queries for a given prefix using BM25 scoring.
        
        Args:
            prefix: Input prefix string (potentially with typos)
            n: Number of top candidates to return (default: 300)
        
        Returns:
            List of tuples (query, score) sorted by BM25 score (descending)
        """
        # Tokenize the prefix using the same n-gram approach
        tokenized_prefix = char_ngram_tokenizer(prefix, n=self.n)
        
        if not tokenized_prefix:
            return []
        
        # Get BM25 scores for all queries
        scores = self.bm25.get_scores(tokenized_prefix)
        
        # Get top n indices sorted by score (descending)
        top_n = min(n, len(scores))
        top_indices = np.argsort(scores)[::-1][:top_n]
        
        # Return query-score pairs
        candidates = [
            (self.queries[idx], scores[idx]) 
            for idx in top_indices
        ]
        
        return candidates
    
    def get_candidate_queries(self, prefix: str, n: int = 300) -> List[str]:
        """
        Get top n candidate queries (without scores) for a given prefix.
        
        Args:
            prefix: Input prefix string (potentially with typos)
            n: Number of top candidates to return (default: 300)
        
        Returns:
            List of query strings sorted by relevance
        """
        candidates = self.get_candidates(prefix, n=n)
        return [query for query, score in candidates]


# ============================================================================
# CONVENIENCE FUNCTION FOR DIRECT USE
# ============================================================================

# Global variable to store the BM25 generator instance
_bm25_generator = None

def initialize_bm25_index(query_pool_df: pd.DataFrame, ngram_size: int = 3):
    """
    Initialize the global BM25 index from a query pool DataFrame.
    
    Args:
        query_pool_df: DataFrame containing a 'query' column
        ngram_size: Size of character n-grams (default: 3 for trigrams)
    """
    global _bm25_generator
    
    if 'query' not in query_pool_df.columns:
        raise ValueError("DataFrame must contain a 'query' column")
    
    _bm25_generator = BM25CandidateGenerator(
        query_pool_df['query'], 
        n=ngram_size
    )
    
    return _bm25_generator


def get_bm25_candidates(prefix: str, n: int = 300) -> List[str]:
    """
    Get top n BM25 candidates for a given prefix.
    
    This is a convenience function that uses the globally initialized BM25 index.
    Call initialize_bm25_index() first to set up the index.
    
    Args:
        prefix: Input prefix string (potentially with typos)
        n: Number of top candidates to return (default: 300)
    
    Returns:
        List of candidate query strings sorted by BM25 relevance
    
    Example:
        >>> initialize_bm25_index(query_pool_df)
        >>> candidates = get_bm25_candidates("dres", n=10)
        >>> print(candidates[:5])
        ['dress', 'dresses', 'dress for women', 'dress material', 'dresser']
    """
    global _bm25_generator
    
    if _bm25_generator is None:
        raise RuntimeError(
            "BM25 index not initialized. Call initialize_bm25_index() first."
        )
    
    return _bm25_generator.get_candidate_queries(prefix, n=n)


# ============================================================================
# EXAMPLE USAGE AND TESTING
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("BM25 CANDIDATE GENERATION - DEMO")
    print("=" * 80)
    
    # Load query pool
    print("\nLoading query pool...")
    query_pool_df = pd.read_parquet('query_pool.parquet')
    print(f"✓ Loaded {len(query_pool_df):,} queries")
    
    # Initialize BM25 index with trigrams (n=3)
    print("\n" + "-" * 80)
    generator = initialize_bm25_index(query_pool_df, ngram_size=3)
    
    # Test with sample prefixes (including typos)
    print("\n" + "=" * 80)
    print("TESTING BM25 CANDIDATE GENERATION")
    print("=" * 80)
    
    test_prefixes = [
        "dress",           # Clean prefix
        "dres",            # Missing letter
        "sheos",           # Typo for "shoes"
        "womn",            # Typo for "women"
        "tshrt",           # Missing vowels
        "kurti",           # Clean prefix
        "saree",           # Clean prefix
    ]
    
    for prefix in test_prefixes:
        print(f"\n{'-' * 80}")
        print(f"Prefix: '{prefix}'")
        print(f"{'-' * 80}")
        
        # Get top 10 candidates using the convenience function
        candidates = get_bm25_candidates(prefix, n=10)
        
        print(f"Top 10 candidates:")
        for i, query in enumerate(candidates, 1):
            print(f"  {i:2d}. {query}")
        
        # Get candidates with scores for analysis
        candidates_with_scores = generator.get_candidates(prefix, n=10)
        print(f"\nTop 3 with scores:")
        for i, (query, score) in enumerate(candidates_with_scores[:3], 1):
            print(f"  {i}. {query:30s} (score: {score:.4f})")
    
    print("\n" + "=" * 80)
    print("PERFORMANCE STATISTICS")
    print("=" * 80)
    
    # Analyze n-gram coverage
    sample_query = query_pool_df['query'].iloc[0] if len(query_pool_df) > 0 else "example"
    sample_ngrams = char_ngram_tokenizer(sample_query, n=3)
    print(f"\nSample query: '{sample_query}'")
    print(f"Trigrams generated: {sample_ngrams}")
    print(f"Number of trigrams: {len(sample_ngrams)}")
    
    # Test character n-gram tokenizer with different n values
    print(f"\n{'Query':<20} | n=2 (bigrams) | n=3 (trigrams) | n=4 (4-grams)")
    print("-" * 80)
    test_words = ["dress", "kurti", "shoe", "saree"]
    for word in test_words:
        bigrams = char_ngram_tokenizer(word, n=2)
        trigrams = char_ngram_tokenizer(word, n=3)
        fourgrams = char_ngram_tokenizer(word, n=4)
        print(f"{word:<20} | {len(bigrams):13d} | {len(trigrams):14d} | {len(fourgrams):11d}")
    
    print("\n" + "=" * 80)
    print("✓ BM25 CANDIDATE GENERATION READY!")
    print("=" * 80)
    print("\nUsage:")
    print("  1. Load your query pool: query_pool_df = pd.read_parquet('query_pool.parquet')")
    print("  2. Initialize index: initialize_bm25_index(query_pool_df)")
    print("  3. Get candidates: candidates = get_bm25_candidates('prefix', n=300)")
