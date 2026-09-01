from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import Dict, Any

from app.config import settings
from app.api.analyze import router as analyze_router
from app.api.similarity import router as similarity_router
from app.api.extract import router as extract_router
from app.api.summarize import router as summarize_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown routines."""
    # Startup tasks
    print(f"Starting {settings.APP_NAME} v{settings.APP_VERSION} (Debug: {settings.DEBUG})")
    yield
    # Shutdown tasks
    print(f"Shutting down {settings.APP_NAME}")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "Customer Support Text Analysis API Gateway. Combines local NLP (preprocessing, POS, "
        "NER, TF-IDF, embeddings via Qdrant) with external LLM fallback for document intelligence."
    ),
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# Configure CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register Core API Routers (Direct endpoints + API v1 prefix)
app.include_router(analyze_router)
app.include_router(similarity_router)
app.include_router(extract_router)
app.include_router(summarize_router)

# Also mount under /api/v1 prefix for versioned access
app.include_router(analyze_router, prefix=settings.API_PREFIX)
app.include_router(similarity_router, prefix=settings.API_PREFIX)
app.include_router(extract_router, prefix=settings.API_PREFIX)
app.include_router(summarize_router, prefix=settings.API_PREFIX)


@app.get(
    "/",
    status_code=status.HTTP_200_OK,
    summary="Gateway Root Information",
    tags=["System"],
)
async def root() -> Dict[str, Any]:
    """Returns gateway status, version, and interactive documentation links."""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "online",
        "docs_url": "/docs",
        "endpoints": [
            "/analyze",
            "/similarity",
            "/extract",
            "/summarize",
            "/health",
        ],
    }


@app.get(
    "/health",
    status_code=status.HTTP_200_OK,
    summary="Gateway Health Check",
    tags=["System"],
)
async def health_check() -> Dict[str, Any]:
    """Health check endpoint for container orchestrators and status monitoring."""
    return {
        "status": "healthy",
        "app_name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "stage": "Day 2 - Skeleton & Schema Freeze Complete",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )
