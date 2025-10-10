"""
Final Submission Generation
Complete inference pipeline for auto-complete system
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import pickle
import json
import warnings
warnings.filterwarnings('ignore')

# Import our candidate generation modules
from Ideal_Approach.bm25_candidate_generation import BM25CandidateGenerator
from Ideal_Approach.semantic_candidate_generation import SemanticCandidateGenerator

# ============================================================================
# FEATURE EXTRACTION (SAME AS TRAINING)
# ============================================================================

def extract_features_for_inference(prefix: str, candidate: str, 
                                   prefix_embedding: np.ndarray,
                                   candidate_embedding: np.ndarray,
                                   bm25_score: float,
                                   query_features_dict: dict,
                                   feature_names: list) -> dict:
    """
    Extract all features for a prefix-candidate pair during inference.
    Must match exactly the features used during training.
    
    Args:
        prefix: Input prefix string
        candidate: Candidate query string
        prefix_embedding: Embedding vector for prefix
        candidate_embedding: Embedding vector for candidate
        bm25_score: BM25 relevance score
        query_features_dict: Dictionary of popularity features for the candidate
        feature_names: List of feature names from training (for validation)
    
    Returns:
        Dictionary of features in the correct order
    """
    features = {}
    
    # 1. Semantic features - Cosine similarity
    cosine_sim = cosine_similarity(
        prefix_embedding.reshape(1, -1),
        candidate_embedding.reshape(1, -1)
    )[0][0]
    features['cosine_similarity'] = float(cosine_sim)
    
    # 2. BM25 lexical score
    features['bm25_score'] = float(bm25_score)
    
    # 3. Lexical features
    features['prefix_length'] = len(prefix)
    features['candidate_length'] = len(candidate)
    features['length_ratio'] = len(prefix) / max(len(candidate), 1)
    
    # Check if prefix is contained in candidate
    features['prefix_in_candidate'] = int(prefix.lower() in candidate.lower())
    features['candidate_starts_with_prefix'] = int(candidate.lower().startswith(prefix.lower()))
    
    # Edit distance features (character overlap)
    prefix_chars = set(prefix.lower())
    candidate_chars = set(candidate.lower())
    features['char_overlap'] = len(prefix_chars & candidate_chars)
    features['char_jaccard'] = len(prefix_chars & candidate_chars) / max(len(prefix_chars | candidate_chars), 1)
    
    # 4. Popularity features from query_features.parquet
    for feature_name, feature_value in query_features_dict.items():
        if feature_name != 'query':
            features[f'popularity_{feature_name}'] = feature_value
    
    return features


# ============================================================================
# CANDIDATE GENERATION WITH FEATURES
# ============================================================================

def generate_candidates_with_features(prefix: str,
                                     bm25_generator: BM25CandidateGenerator,
                                     semantic_generator: SemanticCandidateGenerator,
                                     model: SentenceTransformer,
                                     query_features_dict: dict,
                                     feature_names: list,
                                     n_bm25: int = 250,
                                     n_semantic: int = 250) -> pd.DataFrame:
    """
    Generate candidates and extract all features for re-ranking.
    
    Args:
        prefix: Input prefix string
        bm25_generator: BM25 candidate generator
        semantic_generator: Semantic candidate generator
        model: Sentence-BERT model for embeddings
        query_features_dict: Dictionary of popularity features
        feature_names: List of feature names from training
        n_bm25: Number of BM25 candidates
        n_semantic: Number of semantic candidates
    
    Returns:
        DataFrame with candidates and all features
    """
    # Get BM25 candidates with scores
    bm25_candidates = bm25_generator.get_candidates(prefix, n=n_bm25)
    bm25_df = pd.DataFrame(bm25_candidates, columns=['candidate', 'bm25_score'])
    
    # Get semantic candidates with distances
    semantic_candidates = semantic_generator.get_candidates(prefix, n=n_semantic)
    semantic_df = pd.DataFrame(semantic_candidates, columns=['candidate', 'semantic_distance'])
    
    # Merge both candidate sets
    combined_df = bm25_df.merge(semantic_df, on='candidate', how='outer')
    
    # Fill missing scores
    combined_df['bm25_score'] = combined_df['bm25_score'].fillna(0.0)
    max_distance = combined_df['semantic_distance'].max() if not combined_df['semantic_distance'].isna().all() else 1000.0
    combined_df['semantic_distance'] = combined_df['semantic_distance'].fillna(max_distance)
    
    # Get unique candidates
    candidates = combined_df['candidate'].unique().tolist()
    
    # Encode prefix once
    prefix_embedding = model.encode([prefix], convert_to_numpy=True)[0]
    
    # Encode all candidates
    candidate_embeddings = model.encode(candidates, convert_to_numpy=True, show_progress_bar=False)
    
    # Extract features for each candidate
    feature_rows = []
    for candidate, candidate_embedding in zip(candidates, candidate_embeddings):
        # Get BM25 score
        candidate_row = combined_df[combined_df['candidate'] == candidate]
        bm25_score = candidate_row['bm25_score'].values[0] if len(candidate_row) > 0 else 0.0
        
        # Get popularity features (with default zeros if not found)
        popularity_features = query_features_dict.get(candidate, {})
        if not popularity_features:
            # Get feature columns from first entry
            sample_features = next(iter(query_features_dict.values()), {})
            popularity_features = {k: 0.0 for k in sample_features.keys() if k != 'query'}
        
        # Extract features
        features = extract_features_for_inference(
            prefix=prefix,
            candidate=candidate,
            prefix_embedding=prefix_embedding,
            candidate_embedding=candidate_embedding,
            bm25_score=bm25_score,
            query_features_dict=popularity_features,
            feature_names=feature_names
        )
        
        features['candidate'] = candidate
        feature_rows.append(features)
    
    # Create DataFrame
    features_df = pd.DataFrame(feature_rows)
    
    return features_df


# ============================================================================
# INFERENCE PIPELINE
# ============================================================================

def generate_submission(test_prefixes_df: pd.DataFrame,
                       model,
                       bm25_generator: BM25CandidateGenerator,
                       semantic_generator: SemanticCandidateGenerator,
                       sentence_model: SentenceTransformer,
                       query_features_dict: dict,
                       feature_names: list,
                       n_candidates: int = 500,
                       top_k: int = 150) -> pd.DataFrame:
    """
    Generate final submission with top-K ranked queries for each test prefix.
    
    Args:
        test_prefixes_df: DataFrame with test prefixes
        model: Trained LightGBM model
        bm25_generator: BM25 candidate generator
        semantic_generator: Semantic candidate generator
        sentence_model: Sentence-BERT model
        query_features_dict: Dictionary of popularity features
        feature_names: List of feature names from training
        n_candidates: Number of candidates to generate (~500)
        top_k: Number of top candidates to return (150)
    
    Returns:
        DataFrame with 'prefix' and 'retrieved_queries' columns
    """
    print("=" * 80)
    print("GENERATING SUBMISSION")
    print("=" * 80)
    print(f"\nParameters:")
    print(f"  • Test prefixes: {len(test_prefixes_df):,}")
    print(f"  • Candidates per prefix: {n_candidates}")
    print(f"  • Top-K to retrieve: {top_k}")
    print()
    
    results = []
    
    # Process each test prefix
    for idx, row in tqdm(test_prefixes_df.iterrows(), total=len(test_prefixes_df), 
                         desc="Processing prefixes"):
        prefix = row['prefix'] if 'prefix' in row else row.iloc[0]
        
        # Skip if prefix is missing
        if pd.isna(prefix) or prefix == '':
            print(f"⚠ Warning: Skipping empty prefix at index {idx}")
            results.append({
                'prefix': '',
                'retrieved_queries': []
            })
            continue
        
        try:
            # 1. Generate candidates with features
            candidates_df = generate_candidates_with_features(
                prefix=prefix,
                bm25_generator=bm25_generator,
                semantic_generator=semantic_generator,
                model=sentence_model,
                query_features_dict=query_features_dict,
                feature_names=feature_names,
                n_bm25=n_candidates // 2,
                n_semantic=n_candidates // 2
            )
            
            # 2. Extract feature matrix (in correct order)
            X = candidates_df[feature_names]
            
            # 3. Predict relevance scores using LightGBM
            scores = model.predict_proba(X)[:, 1]  # Probability of positive class
            
            # 4. Add scores to DataFrame
            candidates_df['relevance_score'] = scores
            
            # 5. Sort by score (descending) and select top K
            top_candidates = candidates_df.nlargest(top_k, 'relevance_score')
            
            # 6. Extract query list
            retrieved_queries = top_candidates['candidate'].tolist()
            
            # 7. Store result
            results.append({
                'prefix': prefix,
                'retrieved_queries': retrieved_queries
            })
            
        except Exception as e:
            print(f"\n⚠ Error processing prefix '{prefix}': {str(e)}")
            print(f"   Returning empty results for this prefix")
            results.append({
                'prefix': prefix,
                'retrieved_queries': []
            })
    
    # Create submission DataFrame
    submission_df = pd.DataFrame(results)
    
    print(f"\n✓ Submission generated!")
    print(f"  • Total prefixes processed: {len(submission_df):,}")
    print(f"  • Average candidates per prefix: {submission_df['retrieved_queries'].apply(len).mean():.1f}")
    
    return submission_df


# ============================================================================
# SUBMISSION FORMATTING
# ============================================================================

def format_submission(submission_df: pd.DataFrame, format_type: str = 'list') -> pd.DataFrame:
    """
    Format submission according to required format.
    
    Args:
        submission_df: DataFrame with 'prefix' and 'retrieved_queries' columns
        format_type: 'list' (keeps as list) or 'string' (joins with delimiter)
    
    Returns:
        Formatted submission DataFrame
    """
    print("\n" + "=" * 80)
    print("FORMATTING SUBMISSION")
    print("=" * 80)
    
    if format_type == 'string':
        # Convert list to comma-separated string
        submission_df['retrieved_queries'] = submission_df['retrieved_queries'].apply(
            lambda x: ','.join(x) if isinstance(x, list) else ''
        )
        print("  • Format: Comma-separated string")
    else:
        print("  • Format: List of queries")
    
    print(f"  • Shape: {submission_df.shape}")
    print("\nSample rows:")
    print(submission_df.head())
    
    return submission_df


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("FINAL SUBMISSION GENERATION")
    print("=" * 80)
    print(f"Auto-Complete System - Inference Pipeline")
    print("=" * 80)
    
    # ========================================================================
    # STEP 1: LOAD DATA
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("STEP 1: LOADING DATA")
    print("=" * 80)
    
    from utils.data_loader import load_test_prefixes, load_query_pool, load_query_features
    
    print("\n[1/4] Loading test prefixes from Hugging Face...")
    test_prefixes_df = load_test_prefixes()
    print(f"✓ Loaded {len(test_prefixes_df):,} test prefixes")
    print(f"  Columns: {test_prefixes_df.columns.tolist()}")
    print(f"  Sample prefixes: {test_prefixes_df.head(3).values.flatten().tolist()}")
    
    print("\n[2/4] Loading query pool from Hugging Face...")
    query_pool_df = load_query_pool()
    print(f"✓ Loaded {len(query_pool_df):,} queries")
    
    print("\n[3/4] Loading query features from Hugging Face...")
    query_features_df = load_query_features()
    print(f"✓ Loaded features for {len(query_features_df):,} queries")
    
    # Prepare features lookup
    query_features_dict = query_features_df.set_index('query').to_dict('index')
    
    print("\n[4/4] Loading trained model...")
    with open('reranking_model.pkl', 'rb') as f:
        lgb_model = pickle.load(f)
    print(f"✓ Model loaded successfully")
    
    # Load metadata to get feature names
    with open('reranking_model_metadata.json', 'r') as f:
        metadata = json.load(f)
    feature_names = metadata['feature_names']
    print(f"  • Number of features: {len(feature_names)}")
    print(f"  • Best iteration: {metadata['best_iteration']}")
    print(f"  • Validation AUC: {metadata['metrics']['val_auc']:.4f}")
    
    # ========================================================================
    # STEP 2: INITIALIZE CANDIDATE GENERATORS
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("STEP 2: INITIALIZING CANDIDATE GENERATORS")
    print("=" * 80)
    
    print("\n[1/3] Initializing BM25 generator...")
    bm25_generator = BM25CandidateGenerator(query_pool_df['query'], n=3)
    
    print("\n[2/3] Initializing Semantic generator...")
    semantic_generator = SemanticCandidateGenerator(
        query_pool_df['query'],
        model_name='all-MiniLM-L6-v2',
        batch_size=64
    )
    
    print("\n[3/3] Loading Sentence-BERT model...")
    sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
    print("✓ Model loaded!")
    
    # ========================================================================
    # STEP 3: GENERATE SUBMISSION
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("STEP 3: GENERATING PREDICTIONS")
    print("=" * 80)
    
    submission_df = generate_submission(
        test_prefixes_df=test_prefixes_df,
        model=lgb_model,
        bm25_generator=bm25_generator,
        semantic_generator=semantic_generator,
        sentence_model=sentence_model,
        query_features_dict=query_features_dict,
        feature_names=feature_names,
        n_candidates=500,
        top_k=150
    )
    
    # ========================================================================
    # STEP 4: FORMAT AND SAVE SUBMISSION
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("STEP 4: SAVING SUBMISSION FILES")
    print("=" * 80)
    
    # Save as Parquet (preserves list structure)
    parquet_file = 'submission.parquet'
    submission_df.to_parquet(parquet_file, index=False)
    print(f"\n✓ Saved to: {parquet_file}")
    print(f"  • Format: Parquet with list column")
    
    # Save as CSV with comma-separated queries
    csv_submission_df = submission_df.copy()
    csv_submission_df['retrieved_queries'] = csv_submission_df['retrieved_queries'].apply(
        lambda x: ','.join(x) if isinstance(x, list) else ''
    )
    csv_file = 'submission.csv'
    csv_submission_df.to_csv(csv_file, index=False)
    print(f"\n✓ Saved to: {csv_file}")
    print(f"  • Format: CSV with comma-separated queries")
    
    # Save detailed version with scores for analysis
    print("\n[Bonus] Saving detailed submission with top 10 for inspection...")
    
    # Create a sample with fewer queries for inspection
    inspection_df = submission_df.copy()
    inspection_df['top_10_queries'] = inspection_df['retrieved_queries'].apply(lambda x: x[:10])
    inspection_df['num_queries'] = inspection_df['retrieved_queries'].apply(len)
    inspection_df_export = inspection_df[['prefix', 'top_10_queries', 'num_queries']]
    
    inspection_file = 'submission_inspection.csv'
    inspection_df_export.to_csv(inspection_file, index=False)
    print(f"\n✓ Saved to: {inspection_file}")
    print(f"  • Format: CSV with top 10 queries for easy inspection")
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("✓ SUBMISSION GENERATION COMPLETE!")
    print("=" * 80)
    
    print("\nSubmission Statistics:")
    print(f"  • Total test prefixes: {len(submission_df):,}")
    print(f"  • Queries per prefix: {submission_df['retrieved_queries'].apply(len).mean():.1f} (target: 150)")
    print(f"  • Min queries: {submission_df['retrieved_queries'].apply(len).min()}")
    print(f"  • Max queries: {submission_df['retrieved_queries'].apply(len).max()}")
    
    print("\nOutput Files:")
    print(f"  1. {parquet_file:<30} - Main submission (Parquet)")
    print(f"  2. {csv_file:<30} - CSV format")
    print(f"  3. {inspection_file:<30} - For manual inspection")
    
    print("\nSample Submissions:")
    print("-" * 80)
    for idx in range(min(3, len(submission_df))):
        prefix = submission_df.iloc[idx]['prefix']
        queries = submission_df.iloc[idx]['retrieved_queries'][:5]
        print(f"\nPrefix: '{prefix}'")
        print(f"Top 5 retrieved queries:")
        for i, q in enumerate(queries, 1):
            print(f"  {i}. {q}")
    
    print("\n" + "=" * 80)
    print("Ready for submission! 🚀")
    print("=" * 80)
