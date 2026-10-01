"""
7. 통합 실습 - 지금까지의 6개 그룹을 모두 연결한 미니 버전.

HAIC 프로젝트(serving_app/)와 똑같은 뼈대를 아주 작은 "메모 API"로
축소했습니다. 파일 구조까지 그대로 대응됩니다.

    ex7_integrated/main.py            <->  serving_app/main.py
    ex7_integrated/schemas.py         <->  serving_app/schemas.py
    ex7_integrated/routers/notes.py   <->  serving_app/routers/data.py, predict.py
    ex7_integrated/routers/health.py  <->  serving_app/routers/health.py
    ex7_integrated/static/index.html  <->  serving_app/static/index.html

실행 (반드시 이 ex7_integrated/ 폴더 안에서 - routers, schemas를 같은
위치에서 import하기 때문입니다):
    uvicorn main:app --reload

확인:
    1) http://localhost:8000/         : static/index.html 대시보드
    2) http://localhost:8000/docs     : POST /notes -> GET /notes/{id} -> POST /notes/upload 순서로 호출
    3) 서버 콘솔에 [startup] 로그가 한 번 찍히는지
"""
import os

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from routers import health, notes

app = FastAPI(title="7. 통합 실습 - 메모 API")

# 1. 앱/라우터 구조: 기능별로 나뉜 라우터를 조립
app.include_router(notes.router)
app.include_router(health.router)


# 5. 앱 생명주기: 서버가 뜰 때 한 번 실행
@app.on_event("startup")
def startup():
    print("[startup] 메모 API 준비 완료")


_STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

# 5. 정적 파일: 반드시 라우터 등록 뒤에 mount (등록 순서 = 경로 우선순위)
app.mount("/", StaticFiles(directory=_STATIC_DIR, html=True), name="static")
