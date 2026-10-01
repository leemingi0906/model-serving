"""
Open-Meteo 에서 발전소별 시간별 기상 데이터를 받아 CSV 로 저장한다 (로컬 PC 에서 실행).

    python team_solar/scripts/fetch_open_meteo.py                 # plants.csv 의 use=1 발전소 전부
    python team_solar/scripts/fetch_open_meteo.py --only samcheonpo_2
    python team_solar/scripts/fetch_open_meteo.py --start 2023-01-01 --end 2026-08-31

받는 것 (발전소 위치마다 3종, 같은 위치는 한 번만 호출):
    obs   : 과거 관측(재분석) archive-api          -> 학습용 "그날 실제 날씨"
    fcst  : 과거 예보 아카이브 historical-forecast  -> 학습/평가용 "그날 예보" (예보-관측 격차 측정)
    d1    : 하루 전 발행 예보 previous-runs (…_previous_day1) -> 제도의 D-1 10시 제출 상황과 같은 리드타임

출력: team_solar/data_raw/weather/{obs|fcst|d1}_{lat}_{lon}.csv
      컬럼: time(KST, 매시 정각 = 직전 1시간 구간), shortwave_radiation(W/m2), direct_radiation, diffuse_radiation,
            cloud_cover(%), temperature_2m(C), relative_humidity_2m(%), precipitation(mm), wind_speed_10m(km/h), snowfall(cm)

주의
    - 시각 정렬: Open-Meteo 의 복사량은 "직전 1시간 평균"이라 01:00 값 = 00:00~01:00 구간 = 발전 데이터의 "1시 발전량"과 일치.
    - 무료 API: 키 없음, 하루 1만 호출. 발전소 위치 10곳 x 3종 = 30회라 여유 있음.
    - 이 클라우드 세션은 외부 API 가 차단되어 있어 로컬에서 실행해야 한다.
"""
import argparse
import csv
import os
import sys
import time

import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # team_solar/
PLANTS = os.path.join(ROOT, "plants.csv")
OUT_DIR = os.path.join(ROOT, "data_raw", "weather")

HOURLY = [
    "shortwave_radiation", "direct_radiation", "diffuse_radiation",
    "cloud_cover", "temperature_2m", "relative_humidity_2m",
    "precipitation", "wind_speed_10m", "snowfall",
]
SOURCES = {
    "obs": "https://archive-api.open-meteo.com/v1/archive",
    "fcst": "https://historical-forecast-api.open-meteo.com/v1/forecast",
    "d1": "https://previous-runs-api.open-meteo.com/v1/forecast",
}


def fetch(kind: str, lat: float, lon: float, start: str, end: str) -> list[dict]:
    hourly = HOURLY if kind != "d1" else [f"{h}_previous_day1" for h in HOURLY]
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": start, "end_date": end,
        "hourly": ",".join(hourly),
        "timezone": "Asia/Seoul",
    }
    for attempt in range(6):
        try:
            r = requests.get(SOURCES[kind], params=params, timeout=180)
        except (requests.ConnectionError, requests.Timeout) as e:  # 서버가 연결을 끊거나 응답이 늦을 때
            print(f"    연결 오류({e.__class__.__name__}), {15 * (attempt + 1)}초 후 재시도 {attempt + 1}/6")
            time.sleep(15 * (attempt + 1))
            continue
        if r.status_code in (429, 500, 502, 503, 504):  # rate limit / 서버 일시 오류
            print(f"    HTTP {r.status_code}, {15 * (attempt + 1)}초 후 재시도 {attempt + 1}/6")
            time.sleep(15 * (attempt + 1))
            continue
        r.raise_for_status()
        js = r.json()
        h = js["hourly"]
        times = h["time"]
        rows = []
        for i, t in enumerate(times):
            row = {"time": t}
            for name in hourly:
                row[name.replace("_previous_day1", "")] = h[name][i]
            rows.append(row)
        return rows
    raise RuntimeError(f"{kind} {lat},{lon}: 재시도 초과")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2023-01-01")
    ap.add_argument("--end", default="2026-08-31")
    ap.add_argument("--only", help="plant_id 하나만")
    ap.add_argument("--kinds", default="obs,fcst,d1")
    args = ap.parse_args()

    with open(PLANTS, encoding="utf-8") as f:
        plants = [r for r in csv.DictReader(f)]
    targets = [p for p in plants if p["use"] == "1" and p["lat"] and p["lon"]]
    if args.only:
        targets = [p for p in targets if p["plant_id"] == args.only]
    if not targets:
        sys.exit("plants.csv 에 use=1 이고 lat/lon 이 채워진 발전소가 없습니다.")

    os.makedirs(OUT_DIR, exist_ok=True)
    locations = {}
    for p in targets:
        key = (round(float(p["lat"]), 3), round(float(p["lon"]), 3))
        locations.setdefault(key, []).append(p["plant_id"])

    print(f"발전소 {len(targets)}곳 -> 위치 {len(locations)}곳, 기간 {args.start}~{args.end}")
    for (lat, lon), ids in locations.items():
        for kind in args.kinds.split(","):
            out = os.path.join(OUT_DIR, f"{kind}_{lat}_{lon}.csv")
            if os.path.exists(out):
                print(f"  skip (있음) {out}")
                continue
            rows = fetch(kind, lat, lon, args.start, args.end)
            with open(out, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["time"] + HOURLY)
                w.writeheader()
                w.writerows(rows)
            print(f"  {kind:4s} {lat},{lon} ({', '.join(ids)}) -> {len(rows)}행 -> {os.path.relpath(out, ROOT)}")
            time.sleep(1)
    print("완료. git add team_solar/data_raw/weather && git commit && git push 해주세요.")


if __name__ == "__main__":
    main()
