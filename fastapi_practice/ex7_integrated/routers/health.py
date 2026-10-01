"""1. 앱/라우터 구조 (HAIC 프로젝트의 routers/health.py와 같은 자리) - prefix 없는 라우터"""
from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
def health():
    return {"status": "ok"}
