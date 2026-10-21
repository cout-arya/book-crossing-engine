"""
Dataset and Inverted Index inspection endpoints.
"""

from fastapi import APIRouter, Query, HTTPException
from backend.app.services.dataset_service import DatasetService
from backend.app.services.search_service import SearchService

router = APIRouter()


@router.get("/dataset/stats")
def get_dataset_stats():
    stats = DatasetService.load_stats()
    if not stats:
        raise HTTPException(
            status_code=404,
            detail="Dataset statistics not found. Please run preprocessing first."
        )
    return stats


@router.get("/indexing/stats")
def get_indexing_stats():
    search_service = SearchService.get_instance()
    if not search_service.is_initialized:
        search_service.initialize()

    if search_service.inverted_index is None:
        raise HTTPException(
            status_code=404,
            detail="Inverted index artifact not loaded."
        )
    return search_service.inverted_index.get_statistics()


@router.get("/indexing/inspect")
def inspect_query_indexing(query: str = Query(..., min_length=1, description="Query to inspect")):
    search_service = SearchService.get_instance()
    if not search_service.is_initialized:
        search_service.initialize()

    if search_service.inverted_index is None:
        raise HTTPException(
            status_code=404,
            detail="Inverted index artifact not loaded."
        )
    return search_service.inverted_index.inspect_query(query)
