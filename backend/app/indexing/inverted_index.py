"""
Metadata Inverted Index module for Book Search IR.
Constructs term-to-document postings, term frequencies, and document frequencies.
Supports inspection, boolean candidate retrieval, and index serialization.
"""

import pickle
import math
from typing import Dict, List, Tuple, Set, Any, Optional
import pandas as pd
from pathlib import Path

from backend.app.preprocessing.preprocessor import (
    preprocess_lexical,
    build_searchable_document
)
from backend.app.config import INVERTED_INDEX_PATH


class InvertedIndex:
    def __init__(self):
        # term -> list of (doc_id, term_frequency)
        self.postings: Dict[str, List[Tuple[int, int]]] = {}
        # term -> document frequency (number of documents containing term)
        self.doc_freq: Dict[str, int] = {}
        # doc_id -> total token count in document
        self.doc_lengths: List[int] = []
        # doc_id -> ISBN
        self.doc_id_to_isbn: List[str] = []
        # ISBN -> doc_id
        self.isbn_to_doc_id: Dict[str, int] = {}
        # Total number of documents
        self.num_docs: int = 0
        self.avg_doc_len: float = 0.0

    def build(self, df_books: pd.DataFrame, max_docs: Optional[int] = None) -> "InvertedIndex":
        """
        Builds the inverted index from a books DataFrame.
        Uses title, author, publisher, and year as searchable metadata.
        """
        if max_docs and max_docs < len(df_books):
            df = df_books.iloc[:max_docs].copy()
        else:
            df = df_books

        self.num_docs = len(df)
        self.doc_lengths = [0] * self.num_docs
        self.doc_id_to_isbn = ["" for _ in range(self.num_docs)]
        self.isbn_to_doc_id = {}
        self.postings = {}
        self.doc_freq = {}

        total_tokens = 0

        for doc_id, row in enumerate(df.itertuples(index=False)):
            isbn = str(row.isbn)
            self.doc_id_to_isbn[doc_id] = isbn
            self.isbn_to_doc_id[isbn] = doc_id

            doc_text = build_searchable_document(
                title=getattr(row, "title", ""),
                author=getattr(row, "author", ""),
                publisher=getattr(row, "publisher", ""),
                year=getattr(row, "year", "")
            )

            tokens = preprocess_lexical(doc_text, remove_stops=True, apply_stem=True)
            doc_len = len(tokens)
            self.doc_lengths[doc_id] = doc_len
            total_tokens += doc_len

            # Count term frequencies in this document
            tf_map: Dict[str, int] = {}
            for t in tokens:
                tf_map[t] = tf_map.get(t, 0) + 1

            for term, tf in tf_map.items():
                if term not in self.postings:
                    self.postings[term] = []
                    self.doc_freq[term] = 0
                self.postings[term].append((doc_id, tf))
                self.doc_freq[term] += 1

        self.avg_doc_len = (total_tokens / self.num_docs) if self.num_docs > 0 else 0.0
        return self

    def get_postings(self, term: str) -> List[Tuple[int, int]]:
        """Returns the postings list for a given term: [(doc_id, tf), ...]."""
        return self.postings.get(term, [])

    def get_doc_freq(self, term: str) -> int:
        """Returns the document frequency for a given term."""
        return self.doc_freq.get(term, 0)

    def lookup_term(self, term: str) -> Dict[str, Any]:
        """Provides technical inspection data for a single term."""
        norm_tokens = preprocess_lexical(term, remove_stops=False, apply_stem=True)
        if not norm_tokens:
            return {"term": term, "normalized_term": "", "df": 0, "sample_postings": []}
        
        n_term = norm_tokens[0]
        postings = self.get_postings(n_term)
        return {
            "term": term,
            "normalized_term": n_term,
            "df": len(postings),
            "sample_postings": [
                {"doc_id": doc_id, "isbn": self.doc_id_to_isbn[doc_id], "tf": tf}
                for doc_id, tf in postings[:10]
            ]
        }

    def inspect_query(self, query: str) -> Dict[str, Any]:
        """Demonstrates how a query is tokenized, stemmed, and mapped to inverted index postings."""
        query_tokens = preprocess_lexical(query, remove_stops=True, apply_stem=True)
        matched_terms = {}
        all_matching_doc_ids: Set[int] = set()

        for t in query_tokens:
            postings = self.get_postings(t)
            doc_ids = [doc_id for doc_id, _ in postings]
            all_matching_doc_ids.update(doc_ids)
            matched_terms[t] = {
                "term_doc_frequency": len(postings),
                "matched_doc_count": len(postings),
                "sample_doc_ids": doc_ids[:5],
                "sample_isbns": [self.doc_id_to_isbn[did] for did in doc_ids[:5]]
            }

        return {
            "query": query,
            "processed_tokens": query_tokens,
            "terms_matched": matched_terms,
            "total_candidate_documents": len(all_matching_doc_ids)
        }

    def get_candidate_doc_ids(self, query_tokens: List[str]) -> Set[int]:
        """Returns the set of candidate doc_ids that contain at least one query token."""
        candidates: Set[int] = set()
        for token in query_tokens:
            postings = self.get_postings(token)
            for doc_id, _ in postings:
                candidates.add(doc_id)
        return candidates

    def get_statistics(self) -> Dict[str, Any]:
        """Returns structural statistics of the inverted index."""
        total_postings_entries = sum(len(p) for p in self.postings.values())
        return {
            "total_documents": self.num_docs,
            "vocabulary_size": len(self.postings),
            "total_postings_entries": total_postings_entries,
            "avg_document_length": round(self.avg_doc_len, 2)
        }

    def save(self, file_path: Path = INVERTED_INDEX_PATH):
        """Serializes the inverted index to disk."""
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, file_path: Path = INVERTED_INDEX_PATH) -> "InvertedIndex":
        """Loads a serialized inverted index from disk."""
        if not file_path.exists():
            raise FileNotFoundError(f"Inverted index artifact not found at {file_path}")
        with open(file_path, "rb") as f:
            return pickle.load(f)
