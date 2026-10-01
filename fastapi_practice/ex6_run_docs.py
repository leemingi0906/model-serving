"""
6. 실행과 자동 문서 실습

- uvicorn 모듈경로:app변수 --reload 형식으로 실행
- /docs (Swagger UI) 에서 코드 한 줄 안 짜고 직접 호출
- 환경변수로 동작을 바꾸는 패턴 (HAIC 프로젝트의 MODEL_SOURCE, LOADING_MODE와 같은 방식)

실행 A (기본값):
    uvicorn ex6_run_docs:app --reload

실행 B (환경변수를 바꿔서 다시 실행 - 서버를 껐다 켜야 반영됩니다):
    GREETING=hi uvicorn ex6_run_docs:app --reload
    (Windows PowerShell: $env:GREETING="hi"; uvicorn ex6_run_docs:app --reload)

확인:
    http://localhost:8000/docs 에서 GET /greet 를 Try it out으로 호출해
    실행 A와 실행 B에서 greeting 값이 다르게 나오는지 비교해보세요.
"""
import os

from fastapi import FastAPI

app = FastAPI(title="6. 실행과 자동 문서 실습")

# 환경변수로 서버 동작을 바꾸는 패턴 - HAIC 프로젝트의
# MODEL_SOURCE=mlflow, LOADING_MODE=eager 와 정확히 같은 방식입니다.
GREETING = os.getenv("GREETING", "hello")


@app.get("/greet")
def greet():
    return {
        "greeting": GREETING,
        "hint": "GREETING=hi uvicorn ex6_run_docs:app --reload 로 다시 띄워서 값이 바뀌는지 보세요",
    }
