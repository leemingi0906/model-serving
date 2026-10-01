"""
v2 학습 전 데이터 정합성 검토.

    python team_solar/scripts/validate_data.py

검사 항목
  1) 기상 24개 파일: 행수(1339일x24=32136), 시각 연속성, 결측, 값 범위, 야간 복사량 0, 일 최대 일사
  2) obs / fcst / d1 세 소스 간 상관 (예보 품질이 쓸 만한지)
  3) plants.csv 매핑: use=1 발전소마다 기상 파일 존재, 위치 반올림 일치
  4) 발전 시간별 데이터: 발전소별 행수, 결측, 이용률 범위(0~1), 야간 발전 0, 정지 구간
  5) 발전 x 기상 조인: 시각 키 일치율, 삼천포2 일사량-발전량 상관 (정렬이 맞으면 0.8 이상)
"""
import csv
import glob
import gzip
import math
import os
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W = os.path.join(ROOT, "data_raw", "weather")
VARS = ["shortwave_radiation", "direct_radiation", "diffuse_radiation", "cloud_cover",
        "temperature_2m", "relative_humidity_2m", "precipitation", "wind_speed_10m", "snowfall"]
RANGES = {"shortwave_radiation": (0, 1200), "direct_radiation": (0, 1100), "diffuse_radiation": (0, 700),
          "cloud_cover": (0, 100), "temperature_2m": (-30, 45), "relative_humidity_2m": (0, 100),
          "precipitation": (0, 200), "wind_speed_10m": (0, 150), "snowfall": (0, 50)}
EXPECTED_ROWS = 1339 * 24
problems = []


def corr(a, b):
    n = len(a)
    if n < 2:
        return float("nan")
    ma, mb = sum(a) / n, sum(b) / n
    sa = math.sqrt(sum((x - ma) ** 2 for x in a)); sb = math.sqrt(sum((y - mb) ** 2 for y in b))
    if sa == 0 or sb == 0:
        return float("nan")
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / (sa * sb)


def load_weather(path):
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows


# ---------- 1) 기상 파일 ----------
print("=== 1) 기상 파일 ===")
weather = {}  # (kind, lat, lon) -> {time: row}
for path in sorted(glob.glob(os.path.join(W, "*.csv"))):
    name = os.path.basename(path)[:-4]
    kind, lat, lon = name.split("_")
    rows = load_weather(path)
    times = [r["time"] for r in rows]
    missing = {v: sum(1 for r in rows if r[v] in ("", "None")) for v in VARS}
    out_of_range = {}
    for v in VARS:
        lo, hi = RANGES[v]
        vals = [float(r[v]) for r in rows if r[v] not in ("", "None")]
        bad = sum(1 for x in vals if x < lo or x > hi)
        if bad:
            out_of_range[v] = bad
    # 시각 연속성: 1시간 간격, 중복 없음
    dup = len(times) - len(set(times))
    # 야간(02시) 복사량 0 비율
    night = [float(r["shortwave_radiation"]) for r in rows if r["time"].endswith("T02:00") and r["shortwave_radiation"] not in ("", "None")]
    night_zero = sum(1 for x in night if x == 0) / len(night) if night else float("nan")
    # 일 최대 일사(W/m2) 상위
    daymax = defaultdict(float)
    for r in rows:
        if r["shortwave_radiation"] not in ("", "None"):
            daymax[r["time"][:10]] = max(daymax[r["time"][:10]], float(r["shortwave_radiation"]))
    peak = max(daymax.values()) if daymax else float("nan")
    flag = []
    if len(rows) != EXPECTED_ROWS:
        flag.append(f"행수 {len(rows)} != {EXPECTED_ROWS}")
    if dup:
        flag.append(f"중복 시각 {dup}")
    tot_missing = sum(missing.values())
    if tot_missing:
        flag.append(f"결측 {tot_missing} ({ {k: v for k, v in missing.items() if v} })")
    if out_of_range:
        flag.append(f"범위 밖 {out_of_range}")
    if not (night_zero > 0.99):
        flag.append(f"야간 복사량 0 비율 {night_zero:.2f}")
    status = "OK" if not flag else "확인: " + "; ".join(flag)
    print(f"  {name:<26} rows={len(rows)} {times[0]}~{times[-1]} 피크GHI={peak:.0f}W/m2  {status}")
    if flag:
        problems.append(f"{name}: {status}")
    weather[(kind, lat, lon)] = {r["time"]: r for r in rows}

# ---------- 2) 소스 간 상관 ----------
print("\n=== 2) obs vs fcst vs d1 (shortwave_radiation 상관, 낮 시간만) ===")
locs = sorted({(lat, lon) for (_, lat, lon) in weather})
for lat, lon in locs:
    o, f, d = weather.get(("obs", lat, lon)), weather.get(("fcst", lat, lon)), weather.get(("d1", lat, lon))
    if not (o and f and d):
        problems.append(f"{lat},{lon}: 3종 중 누락"); continue
    keys = [t for t in o if t in f and t in d and o[t]["shortwave_radiation"] not in ("", "None") and f[t]["shortwave_radiation"] not in ("", "None") and d[t]["shortwave_radiation"] not in ("", "None") and float(o[t]["shortwave_radiation"]) > 0]
    xo = [float(o[t]["shortwave_radiation"]) for t in keys]
    xf = [float(f[t]["shortwave_radiation"]) for t in keys]
    xd = [float(d[t]["shortwave_radiation"]) for t in keys]
    print(f"  {lat},{lon}: obs~fcst r={corr(xo, xf):.3f}  obs~d1 r={corr(xo, xd):.3f}  (n={len(keys)})")

# ---------- 3) plants.csv 매핑 ----------
print("\n=== 3) plants.csv 매핑 ===")
with open(os.path.join(ROOT, "plants.csv"), encoding="utf-8") as fh:
    plants = list(csv.DictReader(fh))
use_plants = [p for p in plants if p["use"] == "1"]
loc_of = {}
for p in use_plants:
    key = (f"{round(float(p['lat']), 3)}", f"{round(float(p['lon']), 3)}")
    # 파일명은 str(round(...)) 그대로이므로 37.24 같은 경우 trailing zero 없음
    key = (str(round(float(p["lat"]), 3)), str(round(float(p["lon"]), 3)))
    ok = all((k, key[0], key[1]) in weather for k in ("obs", "fcst", "d1"))
    loc_of[p["plant_id"]] = key
    cap = p["capacity_kw_official"] or p["capacity_kw_est"]
    print(f"  {p['plant_id']:<16} cap={float(cap):>8.1f}kW loc={key} 기상={'OK' if ok else '없음'}")
    if not ok:
        problems.append(f"{p['plant_id']}: 기상 파일 없음 {key}")

# ---------- 4) 발전 시간별 ----------
print("\n=== 4) 발전 시간별 (all_plants_hourly.csv.gz) ===")
gen = defaultdict(dict)  # plant_id -> time -> (kwh, cf)
with gzip.open(os.path.join(ROOT, "data", "all_plants_hourly.csv.gz"), "rt", encoding="utf-8") as fh:
    for r in csv.DictReader(fh):
        gen[r["plant_id"]][r["time"]] = (float(r["generation_kwh"]), float(r["cf"]) if r["cf"] else float("nan"))
for p in use_plants:
    pid = p["plant_id"]
    g = gen.get(pid, {})
    cfs = [v[1] for v in g.values()]
    over1 = sum(1 for c in cfs if c > 1.0)
    night = [v[0] for t, v in g.items() if t.endswith(" 02:00")]
    night_nonzero = sum(1 for x in night if x > 0)
    # 정지 구간: 일 합계 0 인 날 수와 최장 연속
    daily = defaultdict(float)
    for t, v in g.items():
        daily[t[:10]] += v[0]
    days = sorted(daily)
    zero_days = [d for d in days if daily[d] <= 0]
    longest, run, prev = 0, 0, None
    for d in days:
        if daily[d] <= 0:
            run += 1; longest = max(longest, run)
        else:
            run = 0
    flag = []
    if len(g) != EXPECTED_ROWS:
        flag.append(f"행수 {len(g)}")
    if over1:
        flag.append(f"CF>1 {over1}건")
    if night_nonzero:
        flag.append(f"02시 발전>0 {night_nonzero}일")
    print(f"  {pid:<16} rows={len(g)} 평균CF={sum(cfs)/len(cfs):.3f} 0kWh일={len(zero_days)} 최장연속정지={longest}일  {'OK' if not flag else '확인: ' + '; '.join(flag)}")
    if flag:
        problems.append(f"{pid}: {'; '.join(flag)}")

# ---------- 5) 조인 ----------
print("\n=== 5) 발전 x 기상 조인 (시각 키 'YYYY-MM-DD HH:00' <-> 'YYYY-MM-DDTHH:00') ===")
for p in use_plants:
    pid = p["plant_id"]
    lat, lon = loc_of[pid]
    o = weather.get(("obs", lat, lon), {})
    g = gen.get(pid, {})
    # 발전 time 'YYYY-MM-DD 01:00' .. '24:00' -> 기상 'YYYY-MM-DDT01:00' .. ; 24:00 은 다음날 T00:00
    matched, xs, ys = 0, [], []
    for t, (kwh, cf) in g.items():
        day, hh = t.split(" ")
        if hh == "24:00":
            import datetime as dt
            nd = (dt.date.fromisoformat(day) + dt.timedelta(days=1)).isoformat()
            wt = f"{nd}T00:00"
        else:
            wt = f"{day}T{hh}"
        w = o.get(wt)
        if w and w["shortwave_radiation"] not in ("", "None"):
            matched += 1
            xs.append(float(w["shortwave_radiation"])); ys.append(cf)
    r = corr(xs, ys)
    # 정렬 검증: 한 시간 밀린 조인과 비교 (맞게 정렬됐다면 원래가 더 높아야 함)
    xs2, ys2 = [], []
    for t, (kwh, cf) in g.items():
        day, hh = t.split(" ")
        h = int(hh[:2]) - 1  # 한 시간 앞 기상
        if 1 <= h <= 23:
            w = o.get(f"{day}T{h:02d}:00")
            if w and w["shortwave_radiation"] not in ("", "None"):
                xs2.append(float(w["shortwave_radiation"])); ys2.append(cf)
    r2 = corr(xs2, ys2)
    note = "" if r >= r2 else "  <- 한 시간 밀린 조인이 더 높음: 정렬 재확인"
    print(f"  {pid:<16} 매칭 {matched}/{len(g)}  GHI~CF r={r:.3f}  (1h 앞 조인 r={r2:.3f}){note}")
    if matched < len(g) - 1 or r < 0.7:
        problems.append(f"{pid}: 조인 매칭 {matched}/{len(g)}, r={r:.3f}")

print("\n=== 결론 ===")
if problems:
    print("확인 필요 항목:")
    for x in problems:
        print("  -", x)
else:
    print("문제 없음. v2 학습 진행 가능.")
