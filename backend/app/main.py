"""
FastAPI Application Entry Point for Project 15: Book Search using Book-Crossing.
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api import api_router
from backend.app.services.search_service import SearchService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("book_search_api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Eagerly initialize models and caches on startup
    logger.info("Initializing search engine retrieval models and metadata caches...")
    search_service = SearchService.get_instance()
    try:
        search_service.initialize()
    except Exception as e:
        logger.warning(f"Could not initialize all artifacts on startup: {e}")
    yield
    logger.info("Shutting down Book Search IR API.")


app = FastAPI(
    title="Book Search Information Retrieval API",
    description="IR system for Book-Crossing catalog with TF-IDF, BM25, Sentence Transformer, and Hybrid RRF.",
    version="1.0.0",
    lifespan=lifespan
)

# CORS Middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)


@app.get("/")
def read_root():
    return {
        "title": "Project 15: Book Search using Book-Crossing",
        "description": "Information Retrieval engine supporting TF-IDF, BM25, Sentence Transformer, and Hybrid RRF",
        "docs_url": "/docs",
        "health_check": "/api/health"
    }


if __name__ == "__main__":
    import uvicorn
    from backend.app.config import HOST, PORT
    uvicorn.run("backend.app.main:app", host=HOST, port=PORT, reload=True)
