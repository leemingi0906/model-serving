"""
1. 앱/라우터 구조 실습

이 파일 하나로 "1. 앱/라우터 구조" 절의 문법을 전부 확인할 수 있습니다.
- FastAPI() 앱 인스턴스
- APIRouter(prefix=...) 로 기능별로 라우터 나누기
- @router.get() 데코레이터
- def 핸들러 vs async def 핸들러
- app.include_router() 로 조립

실행 (이 파일이 있는 폴더에서):
    uvicorn ex1_router_structure:app --reload

확인:
    http://localhost:8000/docs 에서 아래 3개를 Try it out으로 호출해보세요.
    - GET /notes/ping        : 즉시 응답 (평범한 def)
    - GET /notes/slow-ping   : 0.5초 뒤 응답 (async def + await)
    - GET /health            : notes_router와 다른 prefix를 가진 라우터

서버 콘솔에 요청 로그가 찍히는 타이밍을 눈으로 비교해보면
def와 async def의 차이가 더 잘 느껴집니다.
"""
import asyncio

from fastapi import APIRouter, FastAPI

app = FastAPI(title="1. 앱/라우터 구조 실습")

# 기능별로 라우터를 나눕니다 (HAIC 프로젝트의 routers/data.py, routers/logs.py와 같은 패턴).
notes_router = APIRouter(prefix="/notes")
system_router = APIRouter()  # prefix 없음 - HAIC의 routers/health.py, routers/predict.py와 같은 패턴


@notes_router.get("/ping")
def ping():
    # 평범한 def 핸들러 - 함수 안에서 I/O를 기다릴 필요가 없을 때는 이렇게 씁니다.
    return {"message": "pong", "style": "sync def"}


@notes_router.get("/slow-ping")
async def slow_ping():
    # async def + await - 함수 안에서 시간이 걸리는 I/O를 기다려야 할 때 씁니다.
    # (실제로는 DB 조회, 외부 API 호출, 파일 읽기 같은 작업이 이 자리에 옵니다.)
    await asyncio.sleep(0.5)
    return {"message": "pong", "style": "async def", "waited": "0.5s"}


@system_router.get("/health")
def health():
    return {"status": "ok"}


# 라우터 조립 - 어떤 순서로 include_router()하든 이 예제에서는 경로가 겹치지 않아
# 문제가 없지만, 실제 프로젝트(main.py)에서는 이 등록 순서가 경로 우선순위를 결정합니다.
app.include_router(notes_router)
app.include_router(system_router)
