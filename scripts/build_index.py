"""
Index generation script for Project 15: Book Search using Book-Crossing.
Builds the Inverted Index, TF-IDF matrix, BM25 index, Dense embeddings,
and book metadata cache.
"""

import sys
import time
import argparse
import pickle
from pathlib import Path
import pandas as pd
from tqdm import tqdm

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.app.config import (
    PROCESSED_BOOKS_PARQUET,
    PROCESSED_BOOKS_CSV,
    INVERTED_INDEX_PATH,
    TFIDF_VECTORIZER_PATH,
    TFIDF_MATRIX_PATH,
    BM25_INDEX_PATH,
    METADATA_CACHE_PATH,
    DENSE_MAX_DOCS
)
from backend.app.preprocessing.preprocessor import (
    preprocess_lexical,
    preprocess_lexical_string,
    preprocess_dense,
    build_searchable_document
)
from backend.app.indexing.inverted_index import InvertedIndex
from backend.app.retrieval.tfidf_retriever import TFIDFRetriever
from backend.app.retrieval.bm25_retriever import BM25Retriever
from backend.app.retrieval.dense_retriever import DenseRetriever


def load_dataset() -> pd.DataFrame:
    if PROCESSED_BOOKS_PARQUET.exists():
        print(f"Loading cleaned dataset from {PROCESSED_BOOKS_PARQUET}...")
        return pd.read_parquet(PROCESSED_BOOKS_PARQUET)
    elif PROCESSED_BOOKS_CSV.exists():
        print(f"Loading cleaned dataset from {PROCESSED_BOOKS_CSV}...")
        return pd.read_csv(PROCESSED_BOOKS_CSV, dtype=str)
    else:
        print("[-] Cleaned dataset not found. Please run scripts/preprocess_dataset.py first.")
        sys.exit(1)


def build_catalog_cache(df: pd.DataFrame):
    """Caches fast ISBN -> book metadata lookup dictionary."""
    print("[*] Creating fast metadata cache...")
    cache = {}
    records = df[[
        "isbn", "title", "author", "year", "publisher",
        "image_url_m", "image_url_l", "rating_count", "avg_rating"
    ]].to_dict(orient="records")

    for rec in records:
        cache[rec["isbn"]] = rec

    METADATA_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METADATA_CACHE_PATH, "wb") as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[+] Saved metadata cache ({len(cache):,} books) to {METADATA_CACHE_PATH}")


def main():
    parser = argparse.ArgumentParser(description="Build Search Indexes for Book Search IR")
    parser.add_argument("--skip-dense", action="store_true", help="Skip dense embedding generation")
    parser.add_argument("--dense-docs", type=int, default=DENSE_MAX_DOCS, help="Number of books to embed for dense retrieval")
    parser.add_argument("--full-dense", action="store_true", help="Encode all 271,045 catalog books using resumable chunking")
    parser.add_argument("--chunk-size", type=int, default=10000, help="Chunk size for resumable checkpointing")
    parser.add_argument("--max-docs", type=int, default=None, help="Optional document limit for lexical indexes (default: all)")
    args = parser.parse_args()

    print("=" * 70)
    print("BUILDING SEARCH INDEXES AND ARTIFACTS")
    print("=" * 70)

    start_total = time.time()
    df = load_dataset()
    total_books = len(df)
    print(f"[+] Loaded {total_books:,} books.")

    if args.max_docs and args.max_docs < total_books:
        print(f"[*] Limiting indexing to top {args.max_docs:,} books...")
        df = df.iloc[:args.max_docs].copy()
        total_books = len(df)

    # 1. Build and Save Catalog Cache
    build_catalog_cache(df)

    # 2. Build Inverted Index
    print("\n[*] 1/4. Building Inverted Index...")
    t0 = time.time()
    inv_index = InvertedIndex().build(df)
    inv_index.save(INVERTED_INDEX_PATH)
    inv_stats = inv_index.get_statistics()
    print(f"[+] Inverted Index built in {time.time() - t0:.2f}s:")
    print(f"    - Vocabulary size:         {inv_stats['vocabulary_size']:,}")
    print(f"    - Total postings entries:  {inv_stats['total_postings_entries']:,}")
    print(f"    - Avg document length:     {inv_stats['avg_document_length']} tokens")

    # 3. Build TF-IDF Index
    print("\n[*] 2/4. Building TF-IDF Index...")
    t0 = time.time()
    # Prepare searchable corpus strings
    isbns = df["isbn"].tolist()
    doc_strings = [
        build_searchable_document(
            title=row.title,
            author=row.author,
            publisher=row.publisher,
            year=row.year
        )
        for row in df.itertuples(index=False)
    ]
    preprocessed_doc_strings = [
        preprocess_lexical_string(doc, remove_stops=True, apply_stem=True)
        for doc in tqdm(doc_strings, desc="Lexical preprocessing for TF-IDF")
    ]
    
    tfidf_retriever = TFIDFRetriever()
    tfidf_retriever.fit(preprocessed_doc_strings, isbns)
    tfidf_retriever.save(TFIDF_VECTORIZER_PATH, TFIDF_MATRIX_PATH)
    print(f"[+] TF-IDF index built and saved in {time.time() - t0:.2f}s.")

    # 4. Build BM25 Index
    print("\n[*] 3/4. Building BM25 Index...")
    t0 = time.time()
    tokenized_corpus = [s.split() for s in preprocessed_doc_strings]
    bm25_retriever = BM25Retriever()
    bm25_retriever.fit(tokenized_corpus, isbns)
    bm25_retriever.save(BM25_INDEX_PATH)
    print(f"[+] BM25 index built and saved in {time.time() - t0:.2f}s.")

    # 5. Build Dense Semantic Embeddings (Sentence Transformers)
    if args.skip_dense:
        print("\n[-] Skipping dense embeddings as requested (--skip-dense).")
    else:
        print(f"\n[*] 4/4. Building Dense Semantic Embeddings (Sentence-Transformers)...")
        t0 = time.time()
        dense_retriever = DenseRetriever()
        
        # Sort books by rating count and availability to embed most relevant & active books first
        df_dense = df.sort_values(by=["rating_count", "avg_rating"], ascending=False)
        if args.full_dense:
            dense_limit = len(df_dense)
            print(f"[*] Full catalog dense embedding mode enabled: {dense_limit:,} books.")
        else:
            dense_limit = min(args.dense_docs, len(df_dense))
            print(f"[*] Dense embedding subset mode: {dense_limit:,} books ({dense_limit/len(df)*100:.1f}% of catalog).")

        df_dense_subset = df_dense.iloc[:dense_limit]

        dense_texts = [
            preprocess_dense(
                title=row.title,
                author=row.author,
                publisher=row.publisher,
                year=row.year
            )
            for row in df_dense_subset.itertuples(index=False)
        ]
        dense_isbns = df_dense_subset["isbn"].tolist()

        success = dense_retriever.encode_corpus_resumable(
            texts=dense_texts,
            isbns=dense_isbns,
            chunk_size=args.chunk_size,
            batch_size=64
        )
        if success:
            print(f"[+] Dense embeddings for {dense_limit:,} books generated and saved in {time.time() - t0:.2f}s.")
        else:
            print(f"[-] Dense embedding generation could not be completed: {dense_retriever.load_error}")
            print("    Lexical retrieval models (TF-IDF, BM25) and Inverted Index remain fully functional.")

    print("\n" + "=" * 70)
    print(f"[+] ALL INDEXING COMPLETED in {time.time() - start_total:.2f} seconds.")
    print("=" * 70)


if __name__ == "__main__":
    main()
