"""
Search Orchestration Service.
Loads all retrieval models and metadata caches, handles filtering,
hydrates results with book metadata, and manages system readiness.
"""

import time
import pickle
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path

from backend.app.config import (
    METADATA_CACHE_PATH,
    INVERTED_INDEX_PATH,
    TFIDF_VECTORIZER_PATH,
    TFIDF_MATRIX_PATH,
    BM25_INDEX_PATH,
    DENSE_EMBEDDINGS_PATH
)
from backend.app.indexing.inverted_index import InvertedIndex
from backend.app.retrieval.tfidf_retriever import TFIDFRetriever
from backend.app.retrieval.bm25_retriever import BM25Retriever
from backend.app.retrieval.dense_retriever import DenseRetriever
from backend.app.retrieval.hybrid_rrf import HybridRRFRetriever
from backend.app.schemas.search import SearchResponse, SearchResultItem

logger = logging.getLogger(__name__)


class SearchService:
    _instance = None

    def __init__(self):
        self.metadata_cache: Dict[str, Dict[str, Any]] = {}
        self.inverted_index: Optional[InvertedIndex] = None
        self.tfidf_retriever = TFIDFRetriever()
        self.bm25_retriever = BM25Retriever()
        self.dense_retriever = DenseRetriever()
        self.hybrid_retriever: Optional[HybridRRFRetriever] = None
        self.is_initialized = False

    @classmethod
    def get_instance(cls) -> "SearchService":
        if cls._instance is None:
            cls._instance = SearchService()
        return cls._instance

    def initialize(self):
        """Loads all artifacts and models into memory."""
        logger.info("Initializing SearchService and loading artifacts...")
        
        # 1. Metadata cache
        if METADATA_CACHE_PATH.exists():
            try:
                with open(METADATA_CACHE_PATH, "rb") as f:
                    self.metadata_cache = pickle.load(f)
                logger.info(f"Loaded {len(self.metadata_cache):,} books into metadata cache.")
            except Exception as e:
                logger.error(f"Failed to load metadata cache: {e}")

        # 2. Inverted Index
        if INVERTED_INDEX_PATH.exists():
            try:
                self.inverted_index = InvertedIndex.load(INVERTED_INDEX_PATH)
                logger.info("Inverted Index loaded.")
            except Exception as e:
                logger.error(f"Failed to load Inverted Index: {e}")

        # 3. TF-IDF
        self.tfidf_retriever.load()

        # 4. BM25
        self.bm25_retriever.load()

        # 5. Dense Semantic Retriever
        self.dense_retriever.load()

        # 6. Hybrid Retriever
        self.hybrid_retriever = HybridRRFRetriever(
            retrievers={
                "tfidf": self.tfidf_retriever,
                "bm25": self.bm25_retriever,
                "dense": self.dense_retriever
            }
        )

        # Pre-warm models so first user query has consistent low latency
        if self.dense_retriever.is_available:
            try:
                self.dense_retriever.load_model()
                self.dense_retriever.retrieve("warmup", limit=1)
            except Exception as e:
                logger.warning(f"Dense model warm-up: {e}")

        self.is_initialized = True
        logger.info("SearchService initialization complete.")

    def get_readiness_status(self) -> Dict[str, Any]:
        """Provides fine-grained readiness inspection for health endpoint."""
        return {
            "metadata_cache_loaded": len(self.metadata_cache) > 0,
            "metadata_catalog_size": len(self.metadata_cache),
            "inverted_index_ready": self.inverted_index is not None,
            "tfidf_ready": self.tfidf_retriever.is_available,
            "bm25_ready": self.bm25_retriever.is_available,
            "dense_semantic_ready": self.dense_retriever.is_available,
            "dense_load_error": self.dense_retriever.load_error,
            "hybrid_ready": self.hybrid_retriever is not None and self.hybrid_retriever.is_available,
            "overall_engine_ready": (
                len(self.metadata_cache) > 0 and
                (self.tfidf_retriever.is_available or self.bm25_retriever.is_available)
            )
        }

    def search(
        self,
        query: str,
        model_name: str = "hybrid",
        limit: int = 10,
        author_filter: Optional[str] = None,
        year_filter: Optional[str] = None
    ) -> SearchResponse:
        """Executes search with selected retrieval model and optional metadata filters."""
        if not self.is_initialized:
            self.initialize()

        t0 = time.perf_counter()
        query_clean = str(query).strip()
        model_key = model_name.lower().strip()

        # Check model selection
        retriever_map = {
            "tfidf": self.tfidf_retriever,
            "tf-idf": self.tfidf_retriever,
            "bm25": self.bm25_retriever,
            "dense": self.dense_retriever,
            "semantic": self.dense_retriever,
            "hybrid": self.hybrid_retriever,
            "rrf": self.hybrid_retriever
        }

        retriever = retriever_map.get(model_key, self.hybrid_retriever)
        model_display = retriever.model_name if retriever else "Unknown"

        # Check availability
        if retriever is None or not retriever.is_available:
            latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            return SearchResponse(
                query=query_clean,
                model=model_display,
                results=[],
                total_results=0,
                latency_ms=latency_ms,
                model_availability={
                    "tfidf": self.tfidf_retriever.is_available,
                    "bm25": self.bm25_retriever.is_available,
                    "dense": self.dense_retriever.is_available,
                    "hybrid": self.hybrid_retriever.is_available if self.hybrid_retriever else False
                },
                message=f"Model '{model_display}' is currently not available. Please rebuild index or select an available model."
            )

        # Retrieve candidates (fetch slightly more to accommodate filters)
        fetch_limit = limit * 4 if (author_filter or year_filter) else limit

        if model_key in ["hybrid", "rrf"] and isinstance(retriever, HybridRRFRetriever):
            raw_results = retriever.retrieve_detailed(query_clean, limit=fetch_limit)
        else:
            simple_results = retriever.retrieve(query_clean, limit=fetch_limit)
            raw_results = [
                {"isbn": isbn, "score": score, "rank": idx + 1, "component_ranks": None}
                for idx, (isbn, score) in enumerate(simple_results)
            ]

        # Apply metadata filters and hydrate
        hydrated_results: List[SearchResultItem] = []
        author_filter_lower = author_filter.lower().strip() if author_filter else None
        year_filter_str = str(year_filter).strip() if year_filter else None

        for item in raw_results:
            isbn = item["isbn"]
            meta = self.metadata_cache.get(isbn, {
                "isbn": isbn,
                "title": f"Book {isbn}",
                "author": "Unknown",
                "year": "Unknown",
                "publisher": "Unknown",
                "image_url_m": "",
                "image_url_l": "",
                "rating_count": 0,
                "avg_rating": 0.0
            })

            # Check author filter
            if author_filter_lower:
                book_author = str(meta.get("author", "")).lower()
                if author_filter_lower not in book_author:
                    continue

            # Check year filter
            if year_filter_str:
                book_year = str(meta.get("year", "")).strip()
                if year_filter_str != book_year:
                    continue

            hydrated_results.append(SearchResultItem(
                isbn=isbn,
                title=meta.get("title", ""),
                author=meta.get("author", "Unknown"),
                year=str(meta.get("year", "Unknown")),
                publisher=meta.get("publisher", "Unknown"),
                image_url_m=meta.get("image_url_m") or None,
                image_url_l=meta.get("image_url_l") or None,
                rating_count=int(meta.get("rating_count", 0)),
                avg_rating=float(meta.get("avg_rating", 0.0)),
                score=round(float(item["score"]), 4),
                rank=len(hydrated_results) + 1,
                retrieval_model=model_display,
                component_ranks=item.get("component_ranks")
            ))

            if len(hydrated_results) >= limit:
                break

        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        
        status_msg = None
        if model_key in ["hybrid", "rrf"] and not self.dense_retriever.is_available:
            status_msg = "Hybrid RRF is running in degraded 2-model mode (TF-IDF + BM25) because Sentence Transformer dense index is not loaded."

        return SearchResponse(
            query=query_clean,
            model=model_display,
            results=hydrated_results,
            total_results=len(hydrated_results),
            latency_ms=latency_ms,
            model_availability={
                "tfidf": self.tfidf_retriever.is_available,
                "bm25": self.bm25_retriever.is_available,
                "dense": self.dense_retriever.is_available,
                "hybrid": self.hybrid_retriever.is_available if self.hybrid_retriever else False
            },
            message=status_msg
        )
