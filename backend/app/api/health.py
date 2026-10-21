"""
Health and readiness endpoints.
Distinguishes between a running API and an IR search engine with loaded artifacts.
"""

from fastapi import APIRouter
from backend.app.services.search_service import SearchService

router = APIRouter()


@router.get("/health")
def get_health():
    search_service = SearchService.get_instance()
    readiness = search_service.get_readiness_status()
    
    status_str = "ready" if readiness["overall_engine_ready"] else "degraded"
    
    return {
        "status": status_str,
        "service": "Book Search IR API",
        "version": "1.0.0",
        "readiness_details": readiness
    }
