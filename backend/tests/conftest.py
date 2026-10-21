"""
Pytest fixtures for Book Search IR tests.
Provides small, deterministic fixtures for unit testing without relying on large files.
"""

import pytest
import pandas as pd
from typing import List, Dict, Any

from backend.app.retrieval.tfidf_retriever import TFIDFRetriever
from backend.app.retrieval.bm25_retriever import BM25Retriever
from backend.app.indexing.inverted_index import InvertedIndex
from backend.app.preprocessing.preprocessor import (
    preprocess_lexical,
    preprocess_lexical_string,
    build_searchable_document
)


@pytest.fixture
def sample_books_df() -> pd.DataFrame:
    """Fixture providing a deterministic small book catalog."""
    return pd.DataFrame([
        {
            "isbn": "0439139597",
            "title": "Harry Potter and the Goblet of Fire",
            "author": "J. K. Rowling",
            "year": "2000",
            "publisher": "Scholastic",
            "image_url_m": "http://example.com/hp4.jpg",
            "rating_count": 500,
            "avg_rating": 8.5
        },
        {
            "isbn": "043935806X",
            "title": "Harry Potter and the Order of the Phoenix",
            "author": "J. K. Rowling",
            "year": "2003",
            "publisher": "Scholastic",
            "image_url_m": "http://example.com/hp5.jpg",
            "rating_count": 450,
            "avg_rating": 8.3
        },
        {
            "isbn": "0385504209",
            "title": "The Da Vinci Code",
            "author": "Dan Brown",
            "year": "2003",
            "publisher": "Doubleday",
            "image_url_m": "http://example.com/davinci.jpg",
            "rating_count": 600,
            "avg_rating": 7.8
        },
        {
            "isbn": "0446310786",
            "title": "To Kill a Mockingbird",
            "author": "Harper Lee",
            "year": "1988",
            "publisher": "Warner Books",
            "image_url_m": "http://example.com/mockingbird.jpg",
            "rating_count": 300,
            "avg_rating": 9.0
        },
        {
            "isbn": "0060935464",
            "title": "To Kill a Mockingbird (Special Edition)",
            "author": "Harper Lee",
            "year": "2002",
            "publisher": "Harper Perennial",
            "image_url_m": "http://example.com/mockingbird_sp.jpg",
            "rating_count": 100,
            "avg_rating": 8.9
        },
        {
            "isbn": "0451524934",
            "title": "1984",
            "author": "George Orwell",
            "year": "1990",
            "publisher": "Signet Classic",
            "image_url_m": "http://example.com/1984.jpg",
            "rating_count": 250,
            "avg_rating": 8.7
        }
    ])


@pytest.fixture
def sample_inverted_index(sample_books_df) -> InvertedIndex:
    index = InvertedIndex()
    index.build(sample_books_df)
    return index


@pytest.fixture
def sample_tfidf_retriever(sample_books_df) -> TFIDFRetriever:
    retriever = TFIDFRetriever()
    isbns = sample_books_df["isbn"].tolist()
    docs = [
        preprocess_lexical_string(
            build_searchable_document(row.title, row.author, row.publisher, row.year)
        )
        for row in sample_books_df.itertuples(index=False)
    ]
    retriever.fit(docs, isbns)
    return retriever


@pytest.fixture
def sample_bm25_retriever(sample_books_df) -> BM25Retriever:
    retriever = BM25Retriever()
    isbns = sample_books_df["isbn"].tolist()
    corpus = [
        preprocess_lexical(
            build_searchable_document(row.title, row.author, row.publisher, row.year)
        )
        for row in sample_books_df.itertuples(index=False)
    ]
    retriever.fit(corpus, isbns)
    return retriever
