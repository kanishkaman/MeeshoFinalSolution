import numpy as np
import pandas as pd
import spacy
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from typing import List, Callable, Optional, Dict
from sklearn.feature_extraction.text import HashingVectorizer
from scipy.sparse import csr_matrix
from tqdm import tqdm

# Load spacy for tokenization
nlp = spacy.load("en_core_web_sm", disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer", "ner"])

SparseVector = Dict[str, List]
Array = List[float]

class BM25:
    """Implementation of OKapi BM25 with HashingVectorizer"""

    def __init__(self, tokenizer: Callable[[str], List[str]], n_features=2 ** 16, b=0.75, k1=1.6):
        self.ndocs: int = 0
        self.n_features: int = n_features
        self.doc_freq: Array = []
        self.avgdl: Optional[float] = None
        self._tokenizer: Callable[[str], List[str]] = tokenizer
        self._vectorizer = HashingVectorizer(
            n_features=n_features,
            token_pattern=None,
            tokenizer=tokenizer,
            norm=None,
            alternate_sign=False,
            binary=True
        )
        self.b: float = b
        self.k1: float = k1

    def fit(self, corpus: List[str]) -> "BM25":
        """Fit IDF to documents"""
        X = self._vectorizer.transform(corpus)
        self.avgdl = X.sum(1).mean()
        self.ndocs = X.shape[0]
        self.doc_freq = (
            self._vectorizer
            .transform(corpus)
            .sum(axis=0)
            .A1
        )
        return self

    def transform_doc(self, doc: str) -> SparseVector:
        """Normalize document for BM25 scoring"""
        doc_tf = self._vectorizer.transform([doc])
        norm_doc_tf = self._norm_doc_tf(doc_tf)
        return {'indices': [int(x) for x in doc_tf.indices], 'values': norm_doc_tf}

    def transform_query(self, query: str) -> SparseVector:
        """Normalize query for BM25 scoring"""
        query_tf = self._vectorizer.transform([query])
        indices, values = self._norm_query_tf(query_tf)
        return {'indices': [int(x) for x in indices], 'values': values}
    
    def transform_docs_batch(self, docs: List[str]) -> csr_matrix:
        """Batch transform documents to sparse matrix"""
        doc_tfs = self._vectorizer.transform(docs)
        b, k1, avgdl = self.b, self.k1, self.avgdl
        
        # Normalize each document
        normalized_data = []
        for i in range(doc_tfs.shape[0]):
            row = doc_tfs.getrow(i)
            tf = row.data
            dl = tf.sum()
            norm_tf = tf / (k1 * (1.0 - b + b * (dl / avgdl)) + tf)
            normalized_data.append(norm_tf)
        
        # Reconstruct sparse matrix
        from scipy.sparse import csr_matrix
        normalized_matrix = csr_matrix((
            np.concatenate(normalized_data),
            doc_tfs.indices,
            doc_tfs.indptr
        ), shape=doc_tfs.shape)
        
        return normalized_matrix

    def _norm_doc_tf(self, doc_tf) -> np.ndarray:
        """Calculate BM25 normalized document term-frequencies"""
        b, k1, avgdl = self.b, self.k1, self.avgdl
        tf = doc_tf.data
        dl = tf.sum()
        norm_tf = tf / (k1 * (1.0 - b + b * (dl / avgdl)) + tf)
        return norm_tf

    def _norm_query_tf(self, query_tf):
        """Calculate BM25 normalized query term-frequencies"""
        df = self.doc_freq[query_tf.indices]
        idf = np.log((self.ndocs + 1) / (df + 0.5))
        norm_query_tf = idf / idf.sum()
        return query_tf.indices, norm_query_tf


def tokenizer(text):
    """Tokenize text using spacy"""
    return [token.text.lower() for token in nlp(text)]


def hybrid_score_norm(dense, sparse, alpha=0.5):
    """Normalize dense and sparse vectors for hybrid search"""
    # Normalize dense vector
    dense_norm = np.array(dense) / (np.linalg.norm(dense) + 1e-10)
    
    # Normalize sparse vector
    sparse_values = np.array(sparse['values'])
    sparse_norm = sparse_values / (np.linalg.norm(sparse_values) + 1e-10)
    
    # Apply alpha weighting
    hdense = (dense_norm * alpha).tolist()
    hsparse = {'indices': sparse['indices'], 'values': (sparse_norm * (1 - alpha)).tolist()}
    
    return hdense, hsparse


# Loading the Data from Hugging Face Hub
print("Loading datasets...")
train_data = load_dataset("123tushar/Dice_Challenge_2025", data_files="train_data/*.parquet")
test_prefixes_data = load_dataset("123tushar/Dice_Challenge_2025", data_files="test_prefixes_data/*.parquet")
query_features = load_dataset("123tushar/Dice_Challenge_2025", data_files="query_features/*.parquet")
pool = load_dataset("123tushar/Dice_Challenge_2025", data_files="pool/*.parquet")

# Reading the Data to pandas DF
df_train = train_data['train'].to_pandas()
df_test = test_prefixes_data['train'].to_pandas()
df_query_features = query_features['train'].to_pandas()
df_pool = pool['train'].to_pandas()

print(f"Train data: {len(df_train)} prefix-query pairs")
print(f"Test prefixes: {len(df_test)} prefixes")
print(f"Query pool: {len(df_pool)} queries")
print(f"Query features: {len(df_query_features)} queries")

# Merge query features with pool for enhanced retrieval
df_pool_enhanced = df_pool.merge(df_query_features, on='query', how='left')
df_pool_enhanced.fillna(0, inplace=True)

# Initialize BM25 on the query pool
print("\nFitting BM25 on query pool...")
bm25 = BM25(tokenizer=tokenizer, b=0.75, k1=1.6)
bm25.fit(df_pool['query'].tolist())

# Initialize Sentence Transformer for dense embeddings
print("Loading Sentence Transformer model...")
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Pre-compute dense embeddings for the entire pool
print("Computing dense embeddings for query pool...")
pool_queries = df_pool['query'].tolist()
pool_dense_embeddings = model.encode(pool_queries, show_progress_bar=True, batch_size=32)

# Pre-compute sparse embeddings in batches and store as sparse matrix (MEMORY EFFICIENT!)
print("Computing sparse embeddings for query pool (in batches)...")
batch_size = 10000
sparse_matrices = []

for i in tqdm(range(0, len(pool_queries), batch_size)):
    batch = pool_queries[i:i+batch_size]
    sparse_batch = bm25.transform_docs_batch(batch)
    sparse_matrices.append(sparse_batch)

# Concatenate all sparse matrices
from scipy.sparse import vstack
pool_sparse_matrix = vstack(sparse_matrices)
print(f"Sparse matrix shape: {pool_sparse_matrix.shape}, Memory: {pool_sparse_matrix.data.nbytes / 1e9:.2f} GB")

# Clear temporary data
del sparse_matrices

# Normalize query features for boosting
feature_cols = ['catalog_clicks', 'orders', 'volume', 'catalog_views']
for col in feature_cols:
    if col in df_pool_enhanced.columns:
        max_val = df_pool_enhanced[col].max()
        if max_val > 0:
            df_pool_enhanced[f'{col}_norm'] = df_pool_enhanced[col] / max_val
        else:
            df_pool_enhanced[f'{col}_norm'] = 0

# Calculate popularity score
df_pool_enhanced['popularity_score'] = (
    df_pool_enhanced.get('orders_norm', 0) * 0.4 +
    df_pool_enhanced.get('catalog_clicks_norm', 0) * 0.3 +
    df_pool_enhanced.get('volume_norm', 0) * 0.2 +
    df_pool_enhanced.get('catalog_views_norm', 0) * 0.1
)


def retrieve_candidates(prefix: str, top_k: int = 150, alpha: float = 0.5, popularity_weight: float = 0.2):
    """Retrieve top-k candidate queries for a given prefix using hybrid search"""
    
    # Get dense and sparse representations of the prefix
    prefix_dense = model.encode([prefix])[0]
    prefix_sparse = bm25.transform_query(prefix)
    
    # Normalize for hybrid scoring
    hdense, hsparse = hybrid_score_norm(prefix_dense.tolist(), prefix_sparse, alpha=alpha)
    
    # VECTORIZED Dense similarity (cosine)
    dense_scores = np.dot(pool_dense_embeddings, hdense)
    
    # VECTORIZED Sparse similarity using sparse matrix multiplication
    prefix_sparse_vec = csr_matrix((hsparse['values'], (np.zeros(len(hsparse['indices']), dtype=int), hsparse['indices'])), 
                                    shape=(1, bm25.n_features))
    sparse_scores = pool_sparse_matrix.dot(prefix_sparse_vec.T).toarray().flatten()
    
    # Combine scores
    hybrid_scores = dense_scores + sparse_scores
    
    # Add popularity boost
    popularity_scores = df_pool_enhanced['popularity_score'].values * popularity_weight
    final_scores = hybrid_scores + popularity_scores
    
    # Get top-k indices efficiently
    top_indices = np.argpartition(final_scores, -top_k)[-top_k:]
    top_indices = top_indices[np.argsort(final_scores[top_indices])][::-1]
    
    return [pool_queries[idx] for idx in top_indices]


# Run inference on test prefixes
print("\nGenerating predictions for test prefixes...")
results = []

# Process first 10 as a test
for prefix in tqdm(df_test['prefix'].tolist()[:10]):
    candidates = retrieve_candidates(prefix, top_k=150, alpha=0.6, popularity_weight=0.15)
    results.append({
        'prefix': prefix,
        'candidates': candidates
    })

print(f"\nSample results:")
for i in range(min(3, len(results))):
    print(f"\nPrefix: '{results[i]['prefix']}'")
    print(f"Top 5 candidates: {results[i]['candidates'][:5]}")

# Save results
results_df = pd.DataFrame(results)
print(f"\nGenerated {len(results_df)} predictions")

# To process all and save for submission:
# all_results = []
# for prefix in tqdm(df_test['prefix'].tolist()):
#     candidates = retrieve_candidates(prefix, top_k=150, alpha=0.6, popularity_weight=0.15)
#     all_results.append({'prefix': prefix, 'candidates': candidates})
# results_df = pd.DataFrame(all_results)
# results_df.to_parquet('submission.parquet', index=False)