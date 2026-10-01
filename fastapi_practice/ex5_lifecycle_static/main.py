"""
5. 앱 생명주기 & 정적 파일 실습

- @app.on_event("startup") : 서버가 뜰 때 한 번만 실행되는 훅
- StaticFiles(directory=..., html=True) + app.mount("/", ...) : HTML을 같은 서버에서 서빙
- 라우터를 먼저 등록하고 StaticFiles를 마지막에 mount해야 하는 이유

실행 (이 파일이 있는 ex5_lifecycle_static/ 폴더 안에서 - static/ 폴더를 같이 찾아야 합니다):
    uvicorn main:app --reload

확인:
    1) 서버를 띄우자마자 콘솔에 [startup] 로그가 한 번만 찍히는지 확인
    2) http://localhost:8000/          -> static/index.html이 뜨는지
    3) http://localhost:8000/ping      -> API가 정상 응답하는지 (정적 파일에 안 가로채이는지)
"""
import os
import time

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="5. 앱 생명주기 & 정적 파일 실습")


@app.get("/ping")
def ping():
    return {"message": "pong"}


@app.on_event("startup")
def startup():
    # 서버가 뜰 때 딱 한 번만 실행됩니다. (HAIC 프로젝트의 model_loader eager 로딩과 같은 자리)
    print(f"[startup] 서버 준비 완료 - {time.strftime('%H:%M:%S')}")


_STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

# 반드시 라우터(엔드포인트)를 먼저 등록한 뒤 마지막에 mount해야 합니다.
# StaticFiles를 먼저 mount하면 /ping 같은 API 경로도 정적 파일 요청으로
# 먼저 매칭되어 버립니다 (Starlette는 등록된 순서대로 경로를 검사합니다).
app.mount("/", StaticFiles(directory=_STATIC_DIR, html=True), name="static")
