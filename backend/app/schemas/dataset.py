"""
Dataset statistics and technical schema.
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel


class DatasetStatsResponse(BaseModel):
    books_loaded: int
    books_retained: int
    books_removed: int
    duplicates_removed: int
    ratings_loaded: int
    users_loaded: int
    explicit_ratings_count: int
    implicit_ratings_count: int
    missing_values_before_clean: Dict[str, int]
    year_distribution: Dict[str, int]
    top_authors: Dict[str, int]
    top_publishers: Dict[str, int]
    rating_distribution: Dict[str, int]


class InvertedIndexStatsResponse(BaseModel):
    total_documents: int
    vocabulary_size: int
    total_postings_entries: int
    avg_document_length: float


class InvertedIndexInspectionResponse(BaseModel):
    query: str
    processed_tokens: List[str]
    terms_matched: Dict[str, Any]
    total_candidate_documents: int
