"""
2. 요청을 받는 3가지 방법 실습

- 2-1. 경로 파라미터  : GET /notes/{note_id}
- 2-2. 요청 바디      : POST /notes  (Pydantic 모델)
- 2-3. 파일 업로드    : POST /notes/upload

실행 (이 파일이 있는 폴더에서):
    uvicorn ex2_request_methods:app --reload

확인:
    http://localhost:8000/docs 에서
    1) POST /notes 로 메모를 하나 만들고
    2) 그 응답의 note_id로 GET /notes/{note_id}를 호출해 같은 내용이 나오는지,
    3) 텍스트 파일을 하나 만들어 POST /notes/upload 로 업로드했을 때
       파일 내용 글자 수(chars)가 맞게 나오는지 확인해보세요.
"""
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel

app = FastAPI(title="2. 요청을 받는 3가지 방법 실습")

# 메모리 저장소 (DB 대신 - 실습용. 서버를 재시작하면 초기화됩니다)
NOTES: dict[int, str] = {1: "첫 번째 메모", 2: "두 번째 메모"}


# ── 2-1. 경로 파라미터 ─────────────────────────────────────────
# URL의 {note_id} 부분이 함수 인자 note_id로 그대로 들어옵니다.
# 중괄호 안 이름과 함수 인자 이름이 같아야 자동으로 연결됩니다.
@app.get("/notes/{note_id}")
def get_note(note_id: int):
    return {"note_id": note_id, "content": NOTES.get(note_id, "(해당 메모 없음)")}


# ── 2-2. 요청 바디 ─────────────────────────────────────────────
# Pydantic 모델을 함수 인자의 타입으로 선언하면, FastAPI가 JSON 바디를
# 자동으로 파싱·검증해서 넘겨줍니다.
class NoteCreate(BaseModel):
    content: str


@app.post("/notes")
def create_note(note: NoteCreate):
    new_id = max(NOTES) + 1
    NOTES[new_id] = note.content
    return {"note_id": new_id, "content": note.content}


# ── 2-3. 파일 업로드 ───────────────────────────────────────────
# UploadFile + File(...) 로 받고, await로 파일 내용을 읽습니다.
# 파일을 다루는 I/O이므로 이 핸들러만 async def입니다.
@app.post("/notes/upload")
async def upload_note(file: UploadFile = File(...)):
    raw = await file.read()
    text = raw.decode("utf-8")
    new_id = max(NOTES) + 1
    NOTES[new_id] = text
    return {"note_id": new_id, "filename": file.filename, "chars": len(text)}
