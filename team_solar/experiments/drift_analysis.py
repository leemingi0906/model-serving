"""
드리프트 원인 분석 (실제 데이터). project_solar/ 에서 실행:
    python ../team_solar/experiments/drift_analysis.py

  A. D-1 예보 입력 vs 관측 입력 성능 격차 (서빙 조건 그대로)
  B. 발전소별 월별 PR = 실제 / 모델 기대치(관측 기상)  -> 여러 발전소가 같은 달에 같은 방향으로 움직이면 "모델 드리프트" 후보
  C. 영흥#5·영흥1호기 2026-08 저성능 구간 해부: 일별 PR 변화점, 시간대별 비율(아침/점심/오후), 피크 제한 여부, 같은 부지 대조군(영흥2, 영흥#5 등)
결과: team_solar/experiments/drift_analysis.json + 콘솔 요약
"""
import os, sys, json, math
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.path.insert(0, os.getcwd())
import numpy as np
from datetime import date, timedelta
from collections import defaultdict
from tensorflow import keras
from serving_app import lstm_model  # noqa: F401 (custom loss 등록)
from data.features import (load_plants, load_generation, load_weather, build_dataset, split_by_date, hour_keys,
                           WEATHER_OBS_PATH, WEATHER_D1_PATH, PLANTS_PATH)
from data.metrics import day_error_rate, summarize
from serving_app.train_and_register import TEST_SPLIT

plants = load_plants(PLANTS_PATH)
gen = load_generation("data/sample_solar_hourly.csv.gz")
use = {p: g for p, g in gen.items() if plants[p]["hourly_ok"]}
model = keras.models.load_model("serving_app/models/solarcast_v2.keras")
out = {}


def predict_all(weather):
    Xh, Xf, Xd, Xp, Y, meta = build_dataset(plants, use, weather)
    P = model.predict([Xh, Xf, Xd, Xp], verbose=0, batch_size=256)
    P = np.where(Xf[:, :, -1] > 0, np.clip(P, 0, 1), 0.0)  # 야간 마스킹 (서빙과 동일)
    return P, Y, meta


# ---------- A. D-1 예보 vs 관측 ----------
w_obs = load_weather(WEATHER_OBS_PATH)
w_d1 = load_weather(WEATHER_D1_PATH)
P_obs, Y, meta = predict_all(w_obs)
P_d1, Y2, meta2 = predict_all(w_d1)
assert meta == meta2
te = [i for i, (_, d) in enumerate(meta) if d >= TEST_SPLIT]


def metrics_for(P, idx):
    errs = []
    for i in idx:
        cap = plants[meta[i][0]]["capacity_kw"]
        errs.append(day_error_rate([float(v) * cap for v in P[i]], [float(v) * cap for v in Y[i]], cap))
    return summarize(errs), errs


m_obs, e_obs = metrics_for(P_obs, te)
m_d1, e_d1 = metrics_for(P_d1, te)
out["A_forecast_gap"] = {"obs": m_obs, "d1": m_d1}
print("=== A. 입력 기상에 따른 성능 (테스트 2025-09~2026-08) ===")
print(f"  관측 기상(검증 조건):   mean_error={m_obs['mean_error']}%  pass8={m_obs['pass_rate_8']}")
print(f"  하루 전 예보(서빙 조건): mean_error={m_d1['mean_error']}%  pass8={m_d1['pass_rate_8']}")
# 월별
bym = defaultdict(lambda: [[], []])
for k, i in enumerate(te):
    mo = meta[i][1][:7]
    if e_obs[k] is not None: bym[mo][0].append(e_obs[k])
    if e_d1[k] is not None: bym[mo][1].append(e_d1[k])
out["A_monthly"] = {mo: {"obs": round(float(np.mean(v[0])), 2), "d1": round(float(np.mean(v[1])), 2)} for mo, v in sorted(bym.items())}
print("  월별 obs/d1: " + " ".join(f"{mo[2:]}:{v['obs']}/{v['d1']}" for mo, v in out["A_monthly"].items()))

# ---------- B. 발전소별 월별 PR (관측 기대치) ----------
pr_day = defaultdict(dict)   # pid -> date -> PR
err_day = defaultdict(dict)
for i, (pid, d) in enumerate(meta):
    cap = plants[pid]["capacity_kw"]
    exp = float(P_obs[i].sum()) * cap; act = float(Y[i].sum()) * cap
    if exp >= 0.5 * cap:
        pr_day[pid][d] = act / exp
    err_day[pid][d] = day_error_rate([float(v) * cap for v in P_obs[i]], [float(v) * cap for v in Y[i]], cap)
months = sorted({d[:7] for d in {dd for p in pr_day.values() for dd in p}})
pr_month = {}
for pid in sorted(use):
    pr_month[pid] = {}
    for mo in months:
        vals = [v for d, v in pr_day[pid].items() if d.startswith(mo)]
        if len(vals) >= 10:
            pr_month[pid][mo] = round(float(np.median(vals)), 3)
out["B_pr_month"] = pr_month
print("\n=== B. 발전소별 월별 PR 중앙값 (1.0 = 기상 기대치와 일치) - 테스트 기간 ===")
test_months = [m for m in months if m >= TEST_SPLIT[:7]]
print("  plant           " + " ".join(m[2:] for m in test_months))
for pid in sorted(use):
    print(f"  {pid:<15} " + " ".join(f"{pr_month[pid].get(m, float('nan')):5.2f}" for m in test_months))
# 동시성: 같은 달에 PR < 0.9 또는 > 1.1 인 발전소 비율
fleet = {}
for mo in test_months:
    vals = [pr_month[p].get(mo) for p in use if pr_month[p].get(mo) is not None]
    low = sum(v < 0.9 for v in vals); high = sum(v > 1.1 for v in vals)
    fleet[mo] = {"n": len(vals), "low": low, "high": high, "median": round(float(np.median(vals)), 3) if vals else None}
out["B_fleet"] = fleet
print("  동시성(PR<0.9 / >1.1 발전소 수): " + " ".join(f"{mo[2:]}:{f['low']}/{f['high']}(med {f['median']})" for mo, f in fleet.items()))

# ---------- C. 영흥 해부 ----------
def hourly_ratio(pid, start, end):
    """시간대별 실제/기대 비율 (낮 시간만), 피크 비율"""
    cap = plants[pid]["capacity_kw"]
    act = defaultdict(float); exp = defaultdict(float); mx_act = 0.0
    for i, (p, d) in enumerate(meta):
        if p != pid or not (start <= d <= end): continue
        for h in range(24):
            act[h] += float(Y[i][h]) * cap; exp[h] += float(P_obs[i][h]) * cap
            mx_act = max(mx_act, float(Y[i][h]) * cap)
    ratio = {h + 1: round(act[h] / exp[h], 2) for h in range(24) if exp[h] > 0.02 * cap * max(1, (date.fromisoformat(end) - date.fromisoformat(start)).days)}
    return ratio, round(mx_act / cap, 3)


def changepoint(pid, start, end):
    """일별 PR 시계열에서 전/후 평균 차이가 가장 큰 날짜"""
    days = sorted(d for d in pr_day[pid] if start <= d <= end)
    vals = [pr_day[pid][d] for d in days]
    best = None
    for k in range(5, len(vals) - 5):
        a, b = np.mean(vals[:k]), np.mean(vals[k:])
        score = abs(a - b)
        if best is None or score > best[0]:
            best = (score, days[k], round(float(a), 3), round(float(b), 3))
    return best, days, vals


out["C"] = {}
print("\n=== C. 영흥 저성능 구간 해부 (2026-05-01 ~ 2026-08-31) ===")
for pid in ["yeongheung5_1", "yeongheung_1", "yeongheung_2"]:
    cp, days, vals = changepoint(pid, "2026-05-01", "2026-08-31")
    r_before, pk_before = hourly_ratio(pid, "2025-08-01", "2025-08-31")
    r_after, pk_after = hourly_ratio(pid, "2026-08-01", "2026-08-31")
    zero_days = sum(1 for d in days if d >= "2026-08-01" and all(gen[pid].get(k, 0) == 0 for k in hour_keys(date.fromisoformat(d))))
    out["C"][pid] = {"changepoint": {"date": cp[1], "pr_before": cp[2], "pr_after": cp[3]}, "pr_daily": dict(zip(days, [round(v, 3) for v in vals])),
                     "hourly_ratio_2025_08": r_before, "hourly_ratio_2026_08": r_after, "peak_cf_2025_08": pk_before, "peak_cf_2026_08": pk_after,
                     "zero_days_2026_08": zero_days}
    print(f"  {pid}: 변화점 {cp[1]} (PR {cp[2]} -> {cp[3]}), 2025-08 피크 이용률 {pk_before} -> 2026-08 {pk_after}, 2026-08 0kWh 일수 {zero_days}")
    print("     시간대별 실제/기대 2025-08: " + " ".join(f"{h}:{r_before[h]}" for h in sorted(r_before) if 7 <= h <= 19))
    print("     시간대별 실제/기대 2026-08: " + " ".join(f"{h}:{r_after[h]}" for h in sorted(r_after) if 7 <= h <= 19))

json.dump(out, open("../team_solar/experiments/drift_analysis.json", "w"), ensure_ascii=False, indent=1)
print("\n-> team_solar/experiments/drift_analysis.json")

# ---------- D. 전 발전소 일별 PR·오차 (테스트 기간) 덤프 - 페이지 차트용 ----------
dump = {pid: {"pr": {d: round(v, 3) for d, v in sorted(pr_day[pid].items()) if d >= TEST_SPLIT},
              "err": {d: (None if v is None else round(v, 2)) for d, v in sorted(err_day[pid].items()) if d >= TEST_SPLIT}} for pid in use}
json.dump(dump, open("../team_solar/experiments/pr_daily_test.json", "w"), ensure_ascii=False)
print("-> team_solar/experiments/pr_daily_test.json")
