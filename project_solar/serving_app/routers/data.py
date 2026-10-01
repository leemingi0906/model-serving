"""
SolarCast v2 발전 실적 업로드 / 상태 / 시뮬레이션용 구간 조회 (routers/data.py)

업로드 CSV 컬럼: plant_id, time('YYYY-MM-DD HH:00', HH=01..24), generation_kwh.  .csv 또는 .csv.gz
기상(관측 obs, 하루 전 예보 d1)과 발전소 레지스트리는 data/ 에 동봉되어 있어 업로드하지 않는다.
"""
import csv
import gzip
import io
import os
import time
from datetime import date, timedelta

from fastapi import APIRouter, File, HTTPException, UploadFile

from data.features import HIST_HOURS, HORIZON, load_generation, load_plants, PLANTS_PATH, hour_keys
from data.storage import UPLOAD_DIR, latest_upload, load_recent
from serving_app.monitoring.drift_detector import WINDOW_DAYS

router = APIRouter(prefix="/data")

REQUIRED_COLUMNS = {"plant_id", "time", "generation_kwh"}
MIN_ROWS = HIST_HOURS + HORIZON * WINDOW_DAYS  # 한 발전소가 드리프트 판정까지 가는 데 필요한 최소 시간 수

_gen_cache: tuple[str, dict] | None = None


def generation() -> dict:
    """업로드 최신 파일 + 운영 중 수신분(recent) 을 합친 발전 실적 (파일이 바뀌면 다시 읽음)"""
    global _gen_cache
    path = latest_upload()
    if _gen_cache is None or _gen_cache[0] != path:
        _gen_cache = (path, load_generation(path))
    gen = {pid: dict(rows) for pid, rows in _gen_cache[1].items()}
    for pid, rows in load_recent().items():
        gen.setdefault(pid, {}).update(rows)
    return gen


@router.post("/upload")
async def upload(file: UploadFile = File(...)):
    raw = await file.read()
    is_gz = raw[:2] == b"\x1f\x8b"
    try:
        text = (gzip.decompress(raw) if is_gz else raw).decode("utf-8-sig")
    except (UnicodeDecodeError, OSError):
        raise HTTPException(400, "UTF-8 CSV(.csv / .csv.gz) 파일만 업로드할 수 있습니다.")

    reader = csv.DictReader(io.StringIO(text))
    if not REQUIRED_COLUMNS.issubset(set(reader.fieldnames or [])):
        raise HTTPException(400, f"CSV에 {sorted(REQUIRED_COLUMNS)} 컬럼이 모두 있어야 합니다.")
    n, bad = 0, 0
    plants = load_plants(PLANTS_PATH)
    for r in reader:
        n += 1
        try:
            v = float(r["generation_kwh"])
        except ValueError:
            bad += 1
            continue
        p = plants.get(r["plant_id"])
        if v < 0 or (p and v > p["capacity_kw"] * 1.05):  # 용량의 105% 초과 = 계측 오류 -> 거부 (데이터 오류는 모델까지 안 감)
            bad += 1
    if n < MIN_ROWS:
        raise HTTPException(400, f"최소 {MIN_ROWS}행 이상의 데이터가 필요합니다.")
    if bad > n * 0.01:
        raise HTTPException(400, f"범위 밖 값이 {bad}행({bad / n:.1%}) - 설비용량 초과/음수/비수치. 파일을 확인하세요.")

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    dest = os.path.join(UPLOAD_DIR, f"solar_{int(time.time())}.csv.gz")
    with gzip.open(dest, "wt", encoding="utf-8", newline="") as f:
        f.write(text)
    return {"filename": os.path.basename(dest), "rows": n, "rejected_rows": bad}


@router.get("/status")
def status():
    try:
        path = latest_upload()
    except FileNotFoundError:
        return {"exists": False}
    gen = generation()
    plants = load_plants(PLANTS_PATH)
    times = [t for rows in gen.values() for t in rows]
    per_plant = {}
    for pid, rows in gen.items():
        zero_days = len({t[:10] for t in rows}) - len({t[:10] for t, v in rows.items() if v > 0})
        per_plant[pid] = {"name": plants.get(pid, {}).get("name", pid), "hours": len(rows), "zero_days": zero_days,
                          "capacity_kw": plants.get(pid, {}).get("capacity_kw")}
    return {"exists": True, "filename": os.path.basename(path), "plants": len(gen), "rows": len(times),
            "start": min(times)[:10] if times else None, "end": max(times)[:10] if times else None,
            "per_plant": per_plant}


@router.get("/window")
def window(plant_id: str, start: str | None = None, n_days: int = WINDOW_DAYS, scale: float = 1.0,
           scale_from_day: int = 0):
    """
    시뮬레이션용 실적 구간: start 부터 n_days 일 + 앞 3일 이력 (총 n_days+3 일, 시간별).
      start 없음      -> 업로드 데이터의 마지막 n_days 일
      scale, scale_from_day -> (0-based) scale_from_day 번째 평가일부터 발전량에 scale 을 곱해 돌려줌
                              (설비 고장 = 0.5, 전 발전소 변화 = 1.25 등)
    """
    gen = generation().get(plant_id)
    if not gen:
        raise HTTPException(400, f"'{plant_id}' 실적이 없습니다. 업로드를 확인하세요.")
    days = sorted({t[:10] for t in gen})
    if start:
        if start not in days:
            raise HTTPException(400, f"{start} 는 데이터에 없습니다 ({days[0]} ~ {days[-1]}).")
        i = days.index(start)
    else:
        i = len(days) - n_days
    if i < 3 or i + n_days > len(days):
        raise HTTPException(400, f"{n_days}일 + 이력 3일 구간을 만들 수 없습니다.")
    sel = days[i - 3: i + n_days]
    records = []
    for j, ds in enumerate(sel):
        f = scale if (j >= 3 + scale_from_day and scale != 1.0) else 1.0
        for k in hour_keys(date.fromisoformat(ds)):
            if k in gen:
                records.append({"time": k, "generation_kwh": round(gen[k] * f, 3)})
    return {"plant_id": plant_id, "start": sel[3], "end": sel[-1], "n_days": n_days, "scale": scale,
            "scale_from_day": scale_from_day, "records": records}
