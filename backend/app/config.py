"""
Application configuration for Book Search IR system.
"""

import os
from pathlib import Path

# Base Paths
BACKEND_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BACKEND_DIR.parent
DATA_DIR = BACKEND_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = BACKEND_DIR / "artifacts"

# Ensure directories exist
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# Dataset file paths
RAW_BOOKS_PATH = RAW_DATA_DIR / "BX-Books.csv"
RAW_RATINGS_PATH = RAW_DATA_DIR / "BX-Book-Ratings.csv"
RAW_USERS_PATH = RAW_DATA_DIR / "BX-Users.csv"

PROCESSED_BOOKS_PARQUET = PROCESSED_DATA_DIR / "books_cleaned.parquet"
PROCESSED_BOOKS_CSV = PROCESSED_DATA_DIR / "books_cleaned.csv"
PROCESSED_STATS_JSON = PROCESSED_DATA_DIR / "dataset_stats.json"

# Inverted Index & Retrieval Artifacts
INVERTED_INDEX_PATH = ARTIFACTS_DIR / "inverted_index.pkl"
TFIDF_VECTORIZER_PATH = ARTIFACTS_DIR / "tfidf_vectorizer.pkl"
TFIDF_MATRIX_PATH = ARTIFACTS_DIR / "tfidf_matrix.npz"
BM25_INDEX_PATH = ARTIFACTS_DIR / "bm25_index.pkl"
DENSE_EMBEDDINGS_PATH = ARTIFACTS_DIR / "dense_embeddings.npy"
METADATA_CACHE_PATH = ARTIFACTS_DIR / "catalog_metadata.pkl"
EVALUATION_RESULTS_PATH = ARTIFACTS_DIR / "evaluation_results.json"

# BM25 Parameters
BM25_K1 = 1.5
BM25_B = 0.75

# RRF Parameters
RRF_K = 60  # Standard Reciprocal Rank Fusion constant

# Dense Semantic Search Parameters
DENSE_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DENSE_BATCH_SIZE = 64
# For fast startup and responsive semantic embedding demo on CPU, configurable index limit:
# By default, we index the top/most-rated/cleaned books for dense search or full dataset
DENSE_MAX_DOCS = int(os.environ.get("DENSE_MAX_DOCS", "25000")) 

# Server settings
HOST = "127.0.0.1"
PORT = 8000
