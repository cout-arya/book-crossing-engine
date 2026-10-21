"""
Search request and response Pydantic models.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class SearchResultItem(BaseModel):
    isbn: str
    title: str
    author: str
    year: str
    publisher: str
    image_url_m: Optional[str] = None
    image_url_l: Optional[str] = None
    rating_count: int = 0
    avg_rating: float = 0.0
    score: float
    rank: int
    retrieval_model: str
    component_ranks: Optional[Dict[str, int]] = None


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Search query string")
    model: str = Field(
        default="hybrid",
        description="Retrieval model: tfidf, bm25, semantic, or hybrid"
    )
    limit: int = Field(default=10, ge=1, le=100, description="Max results to return")
    author_filter: Optional[str] = Field(default=None, description="Optional author filter")
    year_filter: Optional[str] = Field(default=None, description="Optional publication year filter")


class SearchResponse(BaseModel):
    query: str
    model: str
    results: List[SearchResultItem]
    total_results: int
    latency_ms: float
    model_availability: Dict[str, bool]
    message: Optional[str] = None
