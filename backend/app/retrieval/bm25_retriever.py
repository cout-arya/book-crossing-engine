"""
BM25 Retrieval Engine.
Implements Okapi BM25 ranking algorithm with configurable k1 and b parameters.
"""

import pickle
import logging
from typing import List, Tuple, Optional
import numpy as np
from rank_bm25 import BM25Okapi
from pathlib import Path

from backend.app.retrieval.base import BaseRetriever
from backend.app.preprocessing.preprocessor import preprocess_lexical
from backend.app.config import BM25_INDEX_PATH, BM25_K1, BM25_B

logger = logging.getLogger(__name__)


class BM25Retriever(BaseRetriever):
    def __init__(self, k1: float = BM25_K1, b: float = BM25_B):
        self.k1 = k1
        self.b = b
        self.bm25: Optional[BM25Okapi] = None
        self.doc_isbns: List[str] = []
        self._is_available = False

    @property
    def model_name(self) -> str:
        return "BM25"

    @property
    def is_available(self) -> bool:
        return self._is_available and self.bm25 is not None

    def fit(self, tokenized_corpus: List[List[str]], isbns: List[str]):
        """Build BM25 index over tokenized document lists."""
        logger.info(f"Indexing {len(tokenized_corpus):,} documents for BM25...")
        self.bm25 = BM25Okapi(tokenized_corpus, k1=self.k1, b=self.b)
        self.doc_isbns = list(isbns)
        self._is_available = True
        logger.info("BM25 indexing complete.")

    def retrieve(
        self,
        query: str,
        limit: int = 10,
        filter_doc_ids: Optional[set] = None
    ) -> List[Tuple[str, float]]:
        if not self.is_available:
            return []

        tokens = preprocess_lexical(query, remove_stops=True, apply_stem=True)
        if not tokens:
            tokens = preprocess_lexical(query, remove_stops=False, apply_stem=True)
            if not tokens:
                return []

        try:
            scores = self.bm25.get_scores(tokens)
            non_zero_indices = np.flatnonzero(scores > 0.0)
            if len(non_zero_indices) == 0:
                return []

            if filter_doc_ids is not None:
                non_zero_indices = np.array([idx for idx in non_zero_indices if idx in filter_doc_ids], dtype=int)
                if len(non_zero_indices) == 0:
                    return []

            actual_limit = min(limit, len(non_zero_indices))
            if len(non_zero_indices) > actual_limit:
                sub_scores = scores[non_zero_indices]
                partition_idx = np.argpartition(sub_scores, -actual_limit)[-actual_limit:]
                top_indices = non_zero_indices[partition_idx]
                top_indices = top_indices[np.argsort(-scores[top_indices])]
            else:
                top_indices = non_zero_indices[np.argsort(-scores[non_zero_indices])]

            results = [(self.doc_isbns[idx], float(scores[idx])) for idx in top_indices]
            return results

        except Exception as e:
            logger.error(f"Error during BM25 retrieval: {e}")
            return []

    def save(self, file_path: Path = BM25_INDEX_PATH):
        """Persist BM25 model and ISBN list."""
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "wb") as f:
            pickle.dump({
                "bm25": self.bm25,
                "doc_isbns": self.doc_isbns,
                "k1": self.k1,
                "b": self.b
            }, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"Saved BM25 artifact to {file_path}")

    def load(self, file_path: Path = BM25_INDEX_PATH) -> bool:
        """Load BM25 artifact from disk."""
        if not file_path.exists():
            logger.warning(f"BM25 artifact missing at {file_path}")
            return False
        try:
            with open(file_path, "rb") as f:
                data = pickle.load(f)
                self.bm25 = data["bm25"]
                self.doc_isbns = data["doc_isbns"]
                self.k1 = data.get("k1", BM25_K1)
                self.b = data.get("b", BM25_B)
            self._is_available = True
            logger.info("BM25 retriever loaded successfully.")
            return True
        except Exception as e:
            logger.error(f"Failed to load BM25 artifact: {e}")
            return False
