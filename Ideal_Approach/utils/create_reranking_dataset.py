"""
LightGBM Re-Ranking Dataset Generation
Creates labeled training data with BM25, semantic, and popularity features
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import our candidate generation modules
from Ideal_Approach.bm25_candidate_generation import BM25CandidateGenerator
from Ideal_Approach.semantic_candidate_generation import SemanticCandidateGenerator

# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

def extract_features(prefix: str, candidate: str, 
                    prefix_embedding: np.ndarray,
                    candidate_embedding: np.ndarray,
                    bm25_score: float,
                    query_features_dict: dict) -> dict:
    """
    Extract all features for a prefix-candidate pair.
    
    Args:
        prefix: Input prefix string
        candidate: Candidate query string
        prefix_embedding: Embedding vector for prefix
        candidate_embedding: Embedding vector for candidate
        bm25_score: BM25 relevance score
        query_features_dict: Dictionary of popularity features for the candidate
    
    Returns:
        Dictionary of features
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
    
    # Check if prefix is contained in candidate (common for auto-complete)
    features['prefix_in_candidate'] = int(prefix.lower() in candidate.lower())
    features['candidate_starts_with_prefix'] = int(candidate.lower().startswith(prefix.lower()))
    
    # Edit distance features (simple character overlap)
    prefix_chars = set(prefix.lower())
    candidate_chars = set(candidate.lower())
    features['char_overlap'] = len(prefix_chars & candidate_chars)
    features['char_jaccard'] = len(prefix_chars & candidate_chars) / max(len(prefix_chars | candidate_chars), 1)
    
    # 4. Popularity features from query_features.parquet
    # Add all available popularity metrics
    for feature_name, feature_value in query_features_dict.items():
        if feature_name != 'query':  # Skip the query column itself
            features[f'popularity_{feature_name}'] = feature_value
    
    return features


# ============================================================================
# CANDIDATE GENERATION WITH SCORING
# ============================================================================

def generate_candidates_with_scores(prefix: str,
                                   bm25_generator: BM25CandidateGenerator,
                                   semantic_generator: SemanticCandidateGenerator,
                                   n_bm25: int = 250,
                                   n_semantic: int = 250) -> pd.DataFrame:
    """
    Generate candidates from both BM25 and semantic search with their scores.
    
    Args:
        prefix: Input prefix string
        bm25_generator: BM25 candidate generator instance
        semantic_generator: Semantic candidate generator instance
        n_bm25: Number of BM25 candidates to retrieve
        n_semantic: Number of semantic candidates to retrieve
    
    Returns:
        DataFrame with columns: ['candidate', 'bm25_score', 'semantic_distance']
    """
    # Get BM25 candidates with scores
    bm25_candidates = bm25_generator.get_candidates(prefix, n=n_bm25)
    bm25_df = pd.DataFrame(bm25_candidates, columns=['candidate', 'bm25_score'])
    
    # Get semantic candidates with distances
    semantic_candidates = semantic_generator.get_candidates(prefix, n=n_semantic)
    semantic_df = pd.DataFrame(semantic_candidates, columns=['candidate', 'semantic_distance'])
    
    # Merge both candidate sets
    # Use outer join to get all unique candidates from both methods
    combined_df = bm25_df.merge(
        semantic_df, 
        on='candidate', 
        how='outer'
    )
    
    # Fill missing scores with default values
    # For candidates not in BM25 results, use 0 score
    combined_df['bm25_score'] = combined_df['bm25_score'].fillna(0.0)
    
    # For candidates not in semantic results, use max distance (least similar)
    max_distance = combined_df['semantic_distance'].max() if not combined_df['semantic_distance'].isna().all() else 1000.0
    combined_df['semantic_distance'] = combined_df['semantic_distance'].fillna(max_distance)
    
    return combined_df


# ============================================================================
# TRAINING DATA GENERATION
# ============================================================================

def create_reranking_dataset(train_df: pd.DataFrame,
                            query_pool_df: pd.DataFrame,
                            query_features_df: pd.DataFrame,
                            bm25_generator: BM25CandidateGenerator,
                            semantic_generator: SemanticCandidateGenerator,
                            model: SentenceTransformer,
                            n_candidates: int = 500,
                            n_negatives_min: int = 5,
                            n_negatives_max: int = 10,
                            max_samples: int = None) -> pd.DataFrame:
    """
    Create labeled training dataset for LightGBM re-ranking model.
    
    Args:
        train_df: Training data with (prefix, query) pairs
        query_pool_df: Pool of all candidate queries
        query_features_df: Query popularity features
        bm25_generator: BM25 candidate generator
        semantic_generator: Semantic candidate generator
        model: Sentence-BERT model for encoding
        n_candidates: Total candidates to generate per prefix (~500)
        n_negatives_min: Minimum number of negative samples
        n_negatives_max: Maximum number of negative samples
        max_samples: Maximum number of training samples to process (for testing)
    
    Returns:
        DataFrame with features and labels for training
    """
    print("=" * 80)
    print("CREATING RE-RANKING TRAINING DATASET")
    print("=" * 80)
    
    # Prepare query features lookup dictionary
    print("\nPreparing query features lookup...")
    query_features_dict = query_features_df.set_index('query').to_dict('index')
    print(f"✓ Loaded features for {len(query_features_dict):,} queries")
    
    # Feature columns from query_features
    feature_columns = [col for col in query_features_df.columns if col != 'query']
    print(f"  Feature columns: {feature_columns}")
    
    # Initialize results list
    all_rows = []
    
    # Limit samples if specified
    if max_samples:
        train_df = train_df.head(max_samples)
        print(f"\n⚠ Processing only first {max_samples:,} samples for testing")
    
    print(f"\nProcessing {len(train_df):,} training examples...")
    print("-" * 80)
    
    # Process each training example
    for idx, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Generating dataset"):
        prefix = row['prefix']
        true_query = row['query']
        
        # Skip if prefix or query is missing
        if pd.isna(prefix) or pd.isna(true_query):
            continue
        
        # 1. Generate candidates using both BM25 and semantic search
        candidates_df = generate_candidates_with_scores(
            prefix,
            bm25_generator,
            semantic_generator,
            n_bm25=n_candidates // 2,
            n_semantic=n_candidates // 2
        )
        
        # Ensure true_query is in candidates (important!)
        if true_query not in candidates_df['candidate'].values:
            # Add true query with default scores
            true_query_row = pd.DataFrame({
                'candidate': [true_query],
                'bm25_score': [0.0],
                'semantic_distance': [0.0]
            })
            candidates_df = pd.concat([candidates_df, true_query_row], ignore_index=True)
        
        # 2. Sample negatives (exclude true_query)
        negative_candidates = candidates_df[
            candidates_df['candidate'] != true_query
        ]['candidate'].tolist()
        
        if len(negative_candidates) == 0:
            continue
        
        # Randomly sample 5-10 negatives
        n_negatives = np.random.randint(n_negatives_min, n_negatives_max + 1)
        n_negatives = min(n_negatives, len(negative_candidates))
        sampled_negatives = np.random.choice(
            negative_candidates, 
            size=n_negatives, 
            replace=False
        )
        
        # 3. Combine positive and negative examples
        examples = [true_query] + list(sampled_negatives)
        labels = [1] + [0] * n_negatives  # 1 for positive, 0 for negatives
        
        # 4. Encode prefix once for all examples
        prefix_embedding = model.encode([prefix], convert_to_numpy=True)[0]
        
        # 5. Extract features for each example
        candidate_embeddings = model.encode(examples, convert_to_numpy=True)
        
        for candidate, label, candidate_embedding in zip(examples, labels, candidate_embeddings):
            # Get BM25 score for this candidate
            candidate_row = candidates_df[candidates_df['candidate'] == candidate]
            bm25_score = candidate_row['bm25_score'].values[0] if len(candidate_row) > 0 else 0.0
            
            # Get popularity features
            popularity_features = query_features_dict.get(candidate, {})
            
            # If no features found, use zeros
            if not popularity_features:
                popularity_features = {col: 0.0 for col in feature_columns}
            
            # Extract all features
            features = extract_features(
                prefix=prefix,
                candidate=candidate,
                prefix_embedding=prefix_embedding,
                candidate_embedding=candidate_embedding,
                bm25_score=bm25_score,
                query_features_dict=popularity_features
            )
            
            # Add metadata
            features['prefix'] = prefix
            features['candidate'] = candidate
            features['target'] = label
            
            # Add to results
            all_rows.append(features)
    
    # Convert to DataFrame
    print("\n\nConverting to DataFrame...")
    training_df = pd.DataFrame(all_rows)
    
    print(f"✓ Dataset created successfully!")
    print(f"\n{'Metric':<30} | Value")
    print("-" * 80)
    print(f"{'Total examples':<30} | {len(training_df):,}")
    print(f"{'Positive examples (target=1)':<30} | {(training_df['target'] == 1).sum():,}")
    print(f"{'Negative examples (target=0)':<30} | {(training_df['target'] == 0).sum():,}")
    print(f"{'Unique prefixes':<30} | {training_df['prefix'].nunique():,}")
    print(f"{'Unique candidates':<30} | {training_df['candidate'].nunique():,}")
    print(f"{'Number of features':<30} | {len([col for col in training_df.columns if col not in ['prefix', 'candidate', 'target']])}")
    
    return training_df


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("LIGHTGBM RE-RANKING DATASET GENERATION")
    print("=" * 80)
    
    # ========================================================================
    # STEP 1: LOAD DATA
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("STEP 1: LOADING DATA")
    print("=" * 80)
    
    print("\nLoading datasets from Hugging Face...")
    from utils.data_loader import load_train_data, load_query_pool, load_query_features
    
    train_df = load_train_data()
    query_pool_df = load_query_pool()
    query_features_df = load_query_features()
    
    print(f"✓ Train data: {len(train_df):,} rows")
    print(f"✓ Query pool: {len(query_pool_df):,} queries")
    print(f"✓ Query features: {len(query_features_df):,} queries")
    
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
    
    print("\n[3/3] Loading Sentence-BERT model for feature extraction...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("✓ Model loaded!")
    
    # ========================================================================
    # STEP 3: CREATE TRAINING DATASET
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("STEP 3: GENERATING TRAINING DATASET")
    print("=" * 80)
    
    # For initial testing, process a subset (remove max_samples for full dataset)
    training_df = create_reranking_dataset(
        train_df=train_df,
        query_pool_df=query_pool_df,
        query_features_df=query_features_df,
        bm25_generator=bm25_generator,
        semantic_generator=semantic_generator,
        model=model,
        n_candidates=500,
        n_negatives_min=5,
        n_negatives_max=10,
        max_samples=100  # Remove this line to process full dataset
    )
    
    # ========================================================================
    # STEP 4: ANALYZE AND SAVE
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("STEP 4: DATASET ANALYSIS")
    print("=" * 80)
    
    print("\nDataset shape:", training_df.shape)
    print("\nFeature columns:")
    feature_cols = [col for col in training_df.columns if col not in ['prefix', 'candidate', 'target']]
    for i, col in enumerate(feature_cols, 1):
        print(f"  {i:2d}. {col}")
    
    print("\nDataset info:")
    print(training_df.info())
    
    print("\nFeature statistics for positive examples:")
    print(training_df[training_df['target'] == 1][feature_cols].describe())
    
    print("\nFeature statistics for negative examples:")
    print(training_df[training_df['target'] == 0][feature_cols].describe())
    
    print("\nSample rows:")
    print(training_df[['prefix', 'candidate', 'target', 'cosine_similarity', 
                       'bm25_score', 'prefix_in_candidate']].head(10))
    
    # ========================================================================
    # STEP 5: SAVE DATASET
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("STEP 5: SAVING DATASET")
    print("=" * 80)
    
    # Save to parquet
    output_file = 'reranking_training_data.parquet'
    training_df.to_parquet(output_file, index=False)
    print(f"✓ Training dataset saved to: {output_file}")
    
    # Also save to CSV for easy inspection
    csv_file = 'reranking_training_data.csv'
    training_df.to_csv(csv_file, index=False)
    print(f"✓ Training dataset saved to: {csv_file}")
    
    # Save feature names for later use
    feature_names_file = 'feature_names.txt'
    with open(feature_names_file, 'w') as f:
        for col in feature_cols:
            f.write(f"{col}\n")
    print(f"✓ Feature names saved to: {feature_names_file}")
    
    print("\n" + "=" * 80)
    print("✓ RE-RANKING DATASET GENERATION COMPLETE!")
    print("=" * 80)
    print("\nNext steps:")
    print("  1. Review the generated dataset")
    print("  2. Train LightGBM model using the features")
    print("  3. Evaluate model performance on validation set")
