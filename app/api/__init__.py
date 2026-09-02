from app.api.analyze import router as analyze_router
from app.api.similarity import router as similarity_router
from app.api.extract import router as extract_router
from app.api.summarize import router as summarize_router

__all__ = [
    "analyze_router",
    "similarity_router",
    "extract_router",
    "summarize_router",
]
