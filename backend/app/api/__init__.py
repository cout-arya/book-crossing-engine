from fastapi import APIRouter
from backend.app.api.health import router as health_router
from backend.app.api.dataset import router as dataset_router
from backend.app.api.search import router as search_router
from backend.app.api.evaluation import router as evaluation_router

api_router = APIRouter(prefix="/api")
api_router.include_router(health_router, tags=["Health"])
api_router.include_router(dataset_router, tags=["Dataset & Indexing"])
api_router.include_router(search_router, tags=["Search"])
api_router.include_router(evaluation_router, tags=["Evaluation"])
