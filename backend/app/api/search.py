"""
Search API endpoints.
"""

from fastapi import APIRouter, HTTPException, Query
from backend.app.schemas.search import SearchRequest, SearchResponse
from backend.app.services.search_service import SearchService

router = APIRouter()


@router.post("/search", response_model=SearchResponse)
def execute_search(request: SearchRequest):
    search_service = SearchService.get_instance()
    
    valid_models = {"tfidf", "bm25", "semantic", "dense", "hybrid", "rrf"}
    if request.model.lower() not in valid_models:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model '{request.model}'. Supported models: tfidf, bm25, dense (alias: semantic), hybrid (alias: rrf)"
        )

    response = search_service.search(
        query=request.query,
        model_name=request.model,
        limit=request.limit,
        author_filter=request.author_filter,
        year_filter=request.year_filter
    )
    return response


@router.get("/search", response_model=SearchResponse)
def execute_search_get(
    query: str = Query(..., min_length=1),
    model: str = Query(default="hybrid"),
    limit: int = Query(default=10, ge=1, le=100),
    author: str = Query(default=None),
    year: str = Query(default=None)
):
    """GET convenience endpoint for browser testing and simple clients."""
    req = SearchRequest(
        query=query,
        model=model,
        limit=limit,
        author_filter=author,
        year_filter=year
    )
    return execute_search(req)
