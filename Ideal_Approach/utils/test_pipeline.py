"""
Quick Testing and Debugging Utility
Test the complete pipeline with sample prefixes
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import json
from Ideal_Approach.bm25_candidate_generation import BM25CandidateGenerator
from Ideal_Approach.semantic_candidate_generation import SemanticCandidateGenerator

# ============================================================================
# INTERACTIVE TESTING
# ============================================================================

def test_prefix_interactive(prefix: str,
                           bm25_gen,
                           semantic_gen,
                           lgb_model,
                           sentence_model,
                           query_features_dict,
                           feature_names,
                           n_candidates=100,
                           top_k=20):
    """
    Test a single prefix and show detailed results.
    
    Args:
        prefix: Input prefix to test
        bm25_gen: BM25 generator
        semantic_gen: Semantic generator
        lgb_model: Trained LightGBM model
        sentence_model: Sentence-BERT model
        query_features_dict: Query features dictionary
        feature_names: List of feature names
        n_candidates: Number of candidates to generate
        top_k: Number of top results to show
    """
    print("=" * 80)
    print(f"TESTING PREFIX: '{prefix}'")
    print("=" * 80)
    
    # 1. BM25 Candidates
    print("\n[1] BM25 Candidates (Top 10):")
    print("-" * 80)
    bm25_candidates = bm25_gen.get_candidates(prefix, n=n_candidates//2)
    for i, (query, score) in enumerate(bm25_candidates[:10], 1):
        print(f"  {i:2d}. {query:<40} (score: {score:.4f})")
    
    # 2. Semantic Candidates
    print("\n[2] Semantic Candidates (Top 10):")
    print("-" * 80)
    semantic_candidates = semantic_gen.get_candidates(prefix, n=n_candidates//2)
    for i, (query, distance) in enumerate(semantic_candidates[:10], 1):
        print(f"  {i:2d}. {query:<40} (distance: {distance:.4f})")
    
    # 3. Combine and extract features
    print("\n[3] Combining candidates and extracting features...")
    print("-" * 80)
    
    # Merge candidates
    bm25_df = pd.DataFrame(bm25_candidates, columns=['candidate', 'bm25_score'])
    semantic_df = pd.DataFrame(semantic_candidates, columns=['candidate', 'semantic_distance'])
    combined_df = bm25_df.merge(semantic_df, on='candidate', how='outer')
    combined_df['bm25_score'] = combined_df['bm25_score'].fillna(0.0)
    combined_df['semantic_distance'] = combined_df['semantic_distance'].fillna(1000.0)
    
    candidates = combined_df['candidate'].unique().tolist()
    print(f"  • Total unique candidates: {len(candidates)}")
    
    # Encode
    prefix_embedding = sentence_model.encode([prefix], convert_to_numpy=True)[0]
    candidate_embeddings = sentence_model.encode(candidates, convert_to_numpy=True, show_progress_bar=False)
    
    # Extract features
    feature_rows = []
    for candidate, candidate_embedding in zip(candidates, candidate_embeddings):
        # Get BM25 score
        candidate_row = combined_df[combined_df['candidate'] == candidate]
        bm25_score = candidate_row['bm25_score'].values[0] if len(candidate_row) > 0 else 0.0
        
        # Cosine similarity
        cosine_sim = cosine_similarity(
            prefix_embedding.reshape(1, -1),
            candidate_embedding.reshape(1, -1)
        )[0][0]
        
        # Get popularity features
        popularity_features = query_features_dict.get(candidate, {})
        
        # Basic features
        features = {
            'candidate': candidate,
            'cosine_similarity': float(cosine_sim),
            'bm25_score': float(bm25_score),
            'prefix_length': len(prefix),
            'candidate_length': len(candidate),
            'length_ratio': len(prefix) / max(len(candidate), 1),
            'prefix_in_candidate': int(prefix.lower() in candidate.lower()),
            'candidate_starts_with_prefix': int(candidate.lower().startswith(prefix.lower())),
        }
        
        # Character overlap
        prefix_chars = set(prefix.lower())
        candidate_chars = set(candidate.lower())
        features['char_overlap'] = len(prefix_chars & candidate_chars)
        features['char_jaccard'] = len(prefix_chars & candidate_chars) / max(len(prefix_chars | candidate_chars), 1)
        
        # Add popularity features
        for feat_name, feat_value in popularity_features.items():
            if feat_name != 'query':
                features[f'popularity_{feat_name}'] = feat_value
        
        feature_rows.append(features)
    
    features_df = pd.DataFrame(feature_rows)
    
    # Fill missing features with 0
    for feat_name in feature_names:
        if feat_name not in features_df.columns:
            features_df[feat_name] = 0.0
    
    # 4. Predict with LightGBM
    print("\n[4] LightGBM Re-ranking:")
    print("-" * 80)
    
    X = features_df[feature_names]
    scores = lgb_model.predict_proba(X)[:, 1]
    features_df['lgb_score'] = scores
    
    # Sort by LightGBM score
    ranked_df = features_df.sort_values('lgb_score', ascending=False)
    
    print(f"\nTop {top_k} Results (LightGBM Re-ranked):")
    print(f"{'Rank':<6} | {'Query':<40} | {'LGB Score':<10} | {'Cosine':<8} | {'BM25':<8} | {'Starts'}")
    print("-" * 80)
    
    for i, row in enumerate(ranked_df.head(top_k).iterrows(), 1):
        idx, data = row
        print(f"{i:<6} | {data['candidate']:<40} | {data['lgb_score']:<10.4f} | "
              f"{data['cosine_similarity']:<8.4f} | {data['bm25_score']:<8.2f} | "
              f"{'✓' if data['candidate_starts_with_prefix'] else '✗'}")
    
    # 5. Feature importance for this prediction
    print("\n[5] Top Features Contributing to Ranking:")
    print("-" * 80)
    
    # Get feature importance from model
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': lgb_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"{'Feature':<40} | {'Importance':<12} | {'Avg Value'}")
    print("-" * 80)
    for idx, row in feature_importance.head(10).iterrows():
        feat_name = row['feature']
        feat_importance = row['importance']
        avg_value = X[feat_name].mean()
        print(f"{feat_name:<40} | {feat_importance:<12.2f} | {avg_value:.4f}")
    
    return ranked_df


# ============================================================================
# COMPARISON MODE
# ============================================================================

def compare_methods(prefix: str, bm25_gen, semantic_gen, n=10):
    """
    Compare BM25 vs Semantic search side-by-side.
    
    Args:
        prefix: Input prefix
        bm25_gen: BM25 generator
        semantic_gen: Semantic generator
        n: Number of results to show
    """
    print("=" * 80)
    print(f"COMPARISON: BM25 vs SEMANTIC for prefix '{prefix}'")
    print("=" * 80)
    
    # Get candidates
    bm25_results = bm25_gen.get_candidates(prefix, n=n)
    semantic_results = semantic_gen.get_candidates(prefix, n=n)
    
    bm25_queries = [q for q, s in bm25_results]
    semantic_queries = [q for q, s in semantic_results]
    
    # Find overlap
    overlap = set(bm25_queries) & set(semantic_queries)
    
    print(f"\n{'Rank':<6} | {'BM25 (Lexical)':<40} | {'Semantic (Meaning)':<40} | {'Overlap'}")
    print("-" * 120)
    
    for i in range(n):
        bm25_q = bm25_queries[i] if i < len(bm25_queries) else ""
        semantic_q = semantic_queries[i] if i < len(semantic_queries) else ""
        overlap_mark = "✓" if (bm25_q in overlap or semantic_q in overlap) else ""
        
        print(f"{i+1:<6} | {bm25_q:<40} | {semantic_q:<40} | {overlap_mark}")
    
    print(f"\nOverlap: {len(overlap)}/{n} queries appear in both top-{n}")
    print(f"Unique to BM25: {len(set(bm25_queries) - set(semantic_queries))}")
    print(f"Unique to Semantic: {len(set(semantic_queries) - set(bm25_queries))}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("QUICK TESTING UTILITY")
    print("=" * 80)
    
    # Load everything
    print("\nLoading data and models from Hugging Face...")
    
    from utils.data_loader import load_query_pool, load_query_features
    
    query_pool_df = load_query_pool()
    query_features_df = load_query_features()
    query_features_dict = query_features_df.set_index('query').to_dict('index')
    
    print("✓ Data loaded")
    
    print("\nInitializing BM25...")
    bm25_gen = BM25CandidateGenerator(query_pool_df['query'], n=3)
    
    print("\nInitializing Semantic search...")
    semantic_gen = SemanticCandidateGenerator(
        query_pool_df['query'],
        model_name='all-MiniLM-L6-v2',
        batch_size=64
    )
    
    print("\nLoading Sentence-BERT model...")
    sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    print("\nLoading LightGBM model...")
    with open('reranking_model.pkl', 'rb') as f:
        lgb_model = pickle.load(f)
    
    with open('reranking_model_metadata.json', 'r') as f:
        metadata = json.load(f)
    feature_names = metadata['feature_names']
    
    print(f"✓ All models loaded (Validation AUC: {metadata['metrics']['val_auc']:.4f})")
    
    # Test prefixes
    test_prefixes = [
        "dress",
        "women clot",  # Incomplete
        "sheos",       # Typo
        "kurta",
        "footwear",
    ]
    
    print("\n" + "=" * 80)
    print("COMPARISON MODE: BM25 vs SEMANTIC")
    print("=" * 80)
    
    for prefix in test_prefixes:
        print("\n")
        compare_methods(prefix, bm25_gen, semantic_gen, n=10)
        print("\n")
    
    print("\n" + "=" * 80)
    print("DETAILED TESTING MODE")
    print("=" * 80)
    
    # Test one prefix in detail
    test_prefix = "dress for women"
    test_prefix_interactive(
        test_prefix,
        bm25_gen,
        semantic_gen,
        lgb_model,
        sentence_model,
        query_features_dict,
        feature_names,
        n_candidates=200,
        top_k=20
    )
    
    print("\n" + "=" * 80)
    print("INTERACTIVE MODE")
    print("=" * 80)
    print("\nEnter 'quit' to exit")
    
    while True:
        try:
            user_prefix = input("\nEnter prefix to test: ").strip()
            
            if user_prefix.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not user_prefix:
                continue
            
            test_prefix_interactive(
                user_prefix,
                bm25_gen,
                semantic_gen,
                lgb_model,
                sentence_model,
                query_features_dict,
                feature_names,
                n_candidates=200,
                top_k=15
            )
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\n⚠ Error: {str(e)}")
            continue
