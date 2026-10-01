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
TIME_RE = __import__("re").compile(r"^\d{4}-\d{2}-\d{2} (0[1-9]|1\d|2[0-4]):00$")
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
    n, bad, bad_time, unknown, dup = 0, 0, 0, set(), 0
    plants = load_plants(PLANTS_PATH)
    seen: set[tuple[str, str]] = set()
    for r in reader:
        n += 1
        if not TIME_RE.match(r["time"] or ""):
            bad_time += 1
            continue
        key = (r["plant_id"], r["time"])
        if key in seen:  # 같은 발전소·시각 중복 = 원본 문제. 조용히 평균 내지 않고 거부한다
            dup += 1
        seen.add(key)
        try:
            v = float(r["generation_kwh"])
        except ValueError:
            bad += 1
            continue
        p = plants.get(r["plant_id"])
        if p is None:
            unknown.add(r["plant_id"])
        if v != v or v < 0 or (p and v > p["capacity_kw"] * 1.05):  # NaN/음수/용량 105% 초과 = 계측 오류 -> 거부
            bad += 1
    if n < MIN_ROWS:
        raise HTTPException(400, f"최소 {MIN_ROWS}행 이상의 데이터가 필요합니다.")
    if bad_time:
        raise HTTPException(400, f"time 형식 오류 {bad_time}행 - 'YYYY-MM-DD HH:00' (HH=01..24, 구간 끝 시각) 이어야 합니다.")
    if dup:
        raise HTTPException(400, f"발전소·시각 중복 {dup}행 - 원본에서 중복을 정리한 뒤 올리세요.")
    if unknown:
        raise HTTPException(400, f"data/plants.csv 에 없는 발전소: {sorted(unknown)[:5]} - 레지스트리를 먼저 갱신하세요.")
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
           scale_from_day: int = 0, heat_loss: float = 0.0, heat_base: float = 28.0):
    """
    시뮬레이션용 실적 구간: start 부터 n_days 일 + 앞 3일 이력 (총 n_days+3 일, 시간별).
      start 없음      -> 업로드 데이터의 마지막 n_days 일
      scale, scale_from_day -> (0-based) scale_from_day 번째 평가일부터 발전량에 scale 을 곱해 돌려줌
                              (설비 고장 = 0.5, 전 발전소 변화 = 1.25 등)
      heat_loss, heat_base  -> 기후 변화(건기·폭염화) 모의: 관측 기온이 heat_base(°C) 를 1°C 넘을 때마다 발전량을
                              heat_loss 만큼 더 깎음 (factor = 1 - heat_loss x max(0, T - heat_base), 하한 0.4).
                              같은 일사량에서도 더운 시간대 발전이 과거보다 낮아진 "관계 변화"를 흉내낸다.
    """
    generation()  # 캐시 갱신
    gen = _gen_cache[1].get(plant_id) if _gen_cache else None  # 업로드 원본만: 운영 중 저장된(변형 주입된) recent 는 섞지 않는다 (시나리오 반복 시 중첩 방지)
    if not gen:
        raise HTTPException(400, f"'{plant_id}' 실적이 없습니다. 업로드를 확인하세요.")
    wloc = None
    if heat_loss > 0:
        from serving_app.routers.predict import weather_obs  # 관측 기상 캐시 재사용
        from data.features import WEATHER_SCALE, gen_time_to_weather_time
        plant = load_plants().get(plant_id)
        wloc = weather_obs().get(plant["loc"], {}) if plant else {}

    def heat_factor(k: str) -> float:
        if not wloc:
            return 1.0
        w = wloc.get(gen_time_to_weather_time(k))
        if not w or w[2] != w[2]:  # 결측(nan)
            return 1.0
        t = w[2] * WEATHER_SCALE["temperature_2m"]
        return max(0.4, 1.0 - heat_loss * max(0.0, t - heat_base))
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
        inject = j >= 3 + scale_from_day
        f = scale if (inject and scale != 1.0) else 1.0
        for k in hour_keys(date.fromisoformat(ds)):
            if k in gen:
                fh = heat_factor(k) if (inject and heat_loss > 0) else 1.0
                records.append({"time": k, "generation_kwh": round(gen[k] * f * fh, 3)})
    return {"plant_id": plant_id, "start": sel[3], "end": sel[-1], "n_days": n_days, "scale": scale,
            "scale_from_day": scale_from_day, "heat_loss": heat_loss, "heat_base": heat_base, "records": records}
