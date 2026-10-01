"""
SolarCast 일 발전량 CSV 업로드 (routers/data.py)

컬럼: Date, generation_kwh (quality 등 추가 컬럼은 무시). 여러 번 업로드하면 계속 쌓이고,
학습(train_and_register.py, fine_tune)은 항상 가장 최근 파일 하나를 사용합니다(data/storage.py).
"""
import csv
import io
import os
import time

from fastapi import APIRouter, File, HTTPException, UploadFile

from data.features import SEQ_LEN, load_rows
from data.storage import UPLOAD_DIR, latest_upload
from serving_app.monitoring.drift_detector import WINDOW_SIZE

router = APIRouter(prefix="/data")

REQUIRED_COLUMNS = {"Date", "generation_kwh"}
MIN_ROWS = SEQ_LEN + WINDOW_SIZE  # 시퀀스 구성 + 드리프트 판정 윈도우에 필요한 최소 행 수


@router.post("/upload")
async def upload(file: UploadFile = File(...)):
    raw = await file.read()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise HTTPException(400, "UTF-8로 인코딩된 CSV 파일만 업로드할 수 있습니다.")

    reader = csv.DictReader(io.StringIO(text))
    if not REQUIRED_COLUMNS.issubset(set(reader.fieldnames or [])):
        raise HTTPException(400, f"CSV에 {sorted(REQUIRED_COLUMNS)} 컬럼이 모두 있어야 합니다.")
    rows = list(reader)
    if len(rows) < MIN_ROWS:
        raise HTTPException(400, f"최소 {MIN_ROWS}행 이상의 데이터가 필요합니다.")

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    dest = os.path.join(UPLOAD_DIR, f"solar_{int(time.time())}.csv")
    with open(dest, "w", encoding="utf-8", newline="") as f:
        f.write(text)

    return {"filename": os.path.basename(dest), "rows": len(rows)}


@router.get("/window")
def window(start: str | None = None, n: int = SEQ_LEN + WINDOW_SIZE, scale: float = 1.0):
    """
    대시보드 드리프트 시뮬레이션용: 업로드된 실제 데이터에서 n 일 구간을 잘라 돌려준다.
      start 없음 -> 마지막 n 일 (정상 배치)
      start=2026-06-20 -> 그 날부터 n 일 (장마철 배치)
      scale=1.8 -> 값에 배율 적용 (신규 발전소 편입 시나리오)
    """
    try:
        rows = load_rows(latest_upload())
    except FileNotFoundError:
        raise HTTPException(400, "업로드된 데이터가 없습니다.")
    if start:
        idx = next((i for i, r in enumerate(rows) if r["Date"] >= start), None)
        if idx is None or idx + n > len(rows):
            raise HTTPException(400, f"{start} 부터 {n}일 구간을 만들 수 없습니다.")
        sel = rows[idx : idx + n]
    else:
        sel = rows[-n:]
    return {
        "start_date": sel[0]["Date"],
        "end_date": sel[-1]["Date"],
        "values": [round(r["Gen"] * scale, 1) for r in sel],
    }


@router.get("/status")
def status():
    try:
        path = latest_upload()
    except FileNotFoundError:
        return {"exists": False}

    rows = load_rows(path)
    gens = [r["Gen"] for r in rows]
    return {
        "exists": True,
        "filename": os.path.basename(path),
        "rows": len(rows),
        "start_date": rows[0]["Date"],
        "end_date": rows[-1]["Date"],
        "min_generation_kwh": min(gens),
        "max_generation_kwh": max(gens),
        "mean_generation_kwh": round(sum(gens) / len(gens), 1),
        "missing_suspect_days": sum(1 for r in rows if r.get("quality") != "ok"),
    }
