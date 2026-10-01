"""Day1: 헬스체크 엔드포인트."""
from fastapi import APIRouter

from serving_app import model_loader

router = APIRouter()


@router.get("/health")
def health():
    from serving_app.monitoring import jobs

    model_loaded = model_loader._model_cache is not None
    return {
        "status": "ok",
        "model_loaded": model_loaded,
        "model_version": model_loader.current_version(),
        "loading_mode": _current_loading_mode(),
        "training": jobs.job_status().get("status", "idle"),
    }


def _current_loading_mode() -> str:
    import os

    return os.getenv("LOADING_MODE", "lazy")
