"""
TF-IDF Retrieval Engine.
Uses scikit-learn's sparse TfidfVectorizer for high performance metadata search.
"""

import pickle
import logging
from typing import List, Tuple, Optional
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path

from backend.app.retrieval.base import BaseRetriever
from backend.app.preprocessing.preprocessor import preprocess_lexical_string
from backend.app.config import TFIDF_VECTORIZER_PATH, TFIDF_MATRIX_PATH

logger = logging.getLogger(__name__)


class TFIDFRetriever(BaseRetriever):
    def __init__(self):
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.doc_matrix: Optional[sp.csr_matrix] = None
        self.doc_isbns: List[str] = []
        self._is_available = False

    @property
    def model_name(self) -> str:
        return "TF-IDF"

    @property
    def is_available(self) -> bool:
        return self._is_available and self.vectorizer is not None and self.doc_matrix is not None

    def fit(self, documents: List[str], isbns: List[str], max_features: int = 100000):
        """Fit TF-IDF on preprocessed document metadata."""
        logger.info(f"Fitting TF-IDF vectorizer on {len(documents):,} documents...")
        self.vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            max_features=max_features,
            sublinear_tf=True
        )
        self.doc_matrix = self.vectorizer.fit_transform(documents)
        self.doc_isbns = list(isbns)
        self._is_available = True
        logger.info(f"TF-IDF fit complete. Vocabulary size: {len(self.vectorizer.vocabulary_):,}")

    def retrieve(
        self,
        query: str,
        limit: int = 10,
        filter_doc_ids: Optional[set] = None
    ) -> List[Tuple[str, float]]:
        if not self.is_available:
            return []

        # Preprocess query with the same pipeline
        processed_query = preprocess_lexical_string(query, remove_stops=True, apply_stem=True)
        if not processed_query.strip():
            # Fallback to without stopword removal if query was all stopwords
            processed_query = preprocess_lexical_string(query, remove_stops=False, apply_stem=True)
            if not processed_query.strip():
                return []

        try:
            q_vec = self.vectorizer.transform([processed_query])
            if q_vec.nnz == 0:
                return []

            # Cosine similarity via sparse dot product
            scores = self.doc_matrix.dot(q_vec.T).toarray().ravel()
            
            # Non-zero scores only
            non_zero_indices = np.flatnonzero(scores > 0.0)
            if len(non_zero_indices) == 0:
                return []

            if filter_doc_ids is not None:
                non_zero_indices = np.array([idx for idx in non_zero_indices if idx in filter_doc_ids], dtype=int)
                if len(non_zero_indices) == 0:
                    return []

            # Top-k selection
            actual_limit = min(limit, len(non_zero_indices))
            if len(non_zero_indices) > actual_limit:
                sub_scores = scores[non_zero_indices]
                partition_idx = np.argpartition(sub_scores, -actual_limit)[-actual_limit:]
                top_indices = non_zero_indices[partition_idx]
                # Sort the top slice
                top_indices = top_indices[np.argsort(-scores[top_indices])]
            else:
                top_indices = non_zero_indices[np.argsort(-scores[non_zero_indices])]

            results = [(self.doc_isbns[idx], float(scores[idx])) for idx in top_indices]
            return results

        except Exception as e:
            logger.error(f"Error during TF-IDF retrieval: {e}")
            return []

    def save(
        self,
        vec_path: Path = TFIDF_VECTORIZER_PATH,
        matrix_path: Path = TFIDF_MATRIX_PATH
    ):
        """Persist vectorizer, sparse matrix, and ISBN mapping."""
        vec_path.parent.mkdir(parents=True, exist_ok=True)
        with open(vec_path, "wb") as f:
            pickle.dump({"vectorizer": self.vectorizer, "doc_isbns": self.doc_isbns}, f)
        sp.save_npz(matrix_path, self.doc_matrix)
        logger.info(f"Saved TF-IDF artifacts to {vec_path} and {matrix_path}")

    def load(
        self,
        vec_path: Path = TFIDF_VECTORIZER_PATH,
        matrix_path: Path = TFIDF_MATRIX_PATH
    ) -> bool:
        """Load TF-IDF artifacts from disk."""
        if not (vec_path.exists() and matrix_path.exists()):
            logger.warning(f"TF-IDF artifacts missing at {vec_path}")
            return False
        try:
            with open(vec_path, "rb") as f:
                data = pickle.load(f)
                self.vectorizer = data["vectorizer"]
                self.doc_isbns = data["doc_isbns"]
            self.doc_matrix = sp.load_npz(matrix_path)
            self._is_available = True
            logger.info("TF-IDF retriever loaded successfully.")
            return True
        except Exception as e:
            logger.error(f"Failed to load TF-IDF artifacts: {e}")
            return False
