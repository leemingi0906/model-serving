"""
2. 요청을 받는 3가지 방법 + 4. 응답과 에러 처리
(HAIC 프로젝트의 routers/data.py, routers/predict.py와 같은 자리)
"""
from fastapi import APIRouter, File, HTTPException, UploadFile

from schemas import NoteCreate, NoteResponse

router = APIRouter(prefix="/notes")

# 메모리 저장소 (DB 대신 - 실습용. 서버를 재시작하면 초기화됩니다)
_NOTES: dict[int, dict] = {}
_NEXT_ID = 1


# 2-2. 요청 바디: Pydantic 모델(NoteCreate)이 JSON 바디를 자동 검증
@router.post("", response_model=NoteResponse)
def create_note(note: NoteCreate):
    global _NEXT_ID
    note_id = _NEXT_ID
    _NEXT_ID += 1
    _NOTES[note_id] = {"title": note.title, "content": note.content, "tags": note.tags}
    return NoteResponse(id=note_id, title=note.title, content=note.content, tag_count=len(note.tags))


# 2-1. 경로 파라미터 + 4. 에러 처리(404)
@router.get("/{note_id}", response_model=NoteResponse)
def get_note(note_id: int):
    n = _NOTES.get(note_id)
    if n is None:
        raise HTTPException(status_code=404, detail=f"note_id={note_id}를 찾을 수 없습니다")
    return NoteResponse(id=note_id, title=n["title"], content=n["content"], tag_count=len(n["tags"]))


# 2-3. 파일 업로드 - 유일한 async def 핸들러
@router.post("/upload")
async def upload_note(file: UploadFile = File(...)):
    global _NEXT_ID
    raw = await file.read()
    text = raw.decode("utf-8")
    note_id = _NEXT_ID
    _NEXT_ID += 1
    _NOTES[note_id] = {"title": file.filename, "content": text, "tags": []}
    return {"note_id": note_id, "filename": file.filename, "chars": len(text)}


@router.get("")
def list_notes():
    return [
        {"id": nid, "title": n["title"], "tag_count": len(n["tags"])}
        for nid, n in sorted(_NOTES.items())
    ]
