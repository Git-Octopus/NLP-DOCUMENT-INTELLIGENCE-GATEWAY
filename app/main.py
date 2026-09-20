import time
import logging
from typing import Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.api.analyze import router as analyze_router
from app.api.similarity import router as similarity_router
from app.api.extract import router as extract_router
from app.api.summarize import router as summarize_router

# Configure Centralized Logging
logging.basicConfig(
    level=logging.INFO if not settings.DEBUG else logging.DEBUG,
    format="%(asctime)s - [%(levelname)s] - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("gateway")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown routines."""
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION} (Debug: {settings.DEBUG})")
    yield
    logger.info(f"Shutting down {settings.APP_NAME}")


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

# Middleware: Request & Response Logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = round((time.time() - start_time) * 1000, 2)
    
    logger.info(
        f"Method: {request.method} Path: {request.url.path} "
        f"Status: {response.status_code} Duration: {duration}ms"
    )
    return response


# Global Exception Handler: HTTPException
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.warning(
        f"HTTPException on {request.method} {request.url.path} - "
        f"Status: {exc.status_code} Detail: {exc.detail}"
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "HTTPException",
            "status_code": exc.status_code,
            "detail": exc.detail,
        },
    )


# Global Exception Handler: Unhandled Internal Server Errors
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(
        f"Unhandled Exception on {request.method} {request.url.path}: {str(exc)}",
        exc_info=True,
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "InternalServerError",
            "status_code": 500,
            "detail": "An unexpected server error occurred. Please check logs.",
        },
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
        "stage": "Logging & Central Error Handling Active",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )
