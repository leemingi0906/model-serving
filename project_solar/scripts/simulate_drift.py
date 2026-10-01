"""
Day3 드리프트 시뮬레이션 (SolarCast v2) - 실제 실적을 구간별로 잘라 주입하고 원인 분류·대응을 확인한다.

    시나리오       데이터                                        기대 판정 -> 대응
    normal         삼천포2 최근 21일 (2026-08)                      ok
    monsoon        삼천포2 장마철 2026-06-20 ~ 07-10 (21일)          ok 또는 weather (PR 정상) -> 재학습 안 함
    equipment      삼천포2 최근 21일, 8일째부터 발전량 x0.5           equipment -> [ALERT] 재학습 금지
    fleet_shift    발전소 3곳 최근 21일 x1.25 (전 발전소 동시 변화)   model_drift -> 재학습 -> 게이트 -> 재배포
    --- 실제 데이터에서 찾은 사례 (주입 없음) ---
    real_gwangyang 광양항 2025-10-05~ (PR 0.3 지속, 10/6~11 완전 정지)       equipment -> [ALERT]
    real_sc2_dec   삼천포2 2025-12-05~ (PR 0.96 -> 0.46 급락, 3주 뒤 복구)   equipment -> [ALERT]
    real_yh5       영흥#5 2026-08-11~ (2026-03 부터 PR 0.96 -> 0.64 점진 하락) equipment (장기 기록 있으면 soiling) -> [ALERT]

사전 준비: MODEL_SOURCE=mlflow uvicorn serving_app.main:app --port 8010 서버가 떠 있고, 실적이 업로드되어 있어야 한다.
실행: python scripts/simulate_drift.py [--only normal,monsoon,...]
"""
import argparse
import os
import sys

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE = os.getenv("API_BASE", "http://localhost:8010")
N_DAYS = 21
MONSOON_START = "2026-06-20"
FLEET = ["samcheonpo_2", "samcheonpo_3", "gyeongsang_1"]  # 경남 3곳: "같은 지역 발전소 동시 변화"


def fetch_window(plant_id, **params):
    r = requests.get(f"{BASE}/data/window", params={"plant_id": plant_id, "n_days": N_DAYS, **params}, timeout=60)
    r.raise_for_status()
    return r.json()


def send(plant_id, label, records, persist=True, reset_state=True, check=True):
    r = requests.post(f"{BASE}/predict/batch-test",
                      json={"plant_id": plant_id, "label": label, "records": records, "persist": persist,
                            "reset_state": reset_state, "check": check}, timeout=1800)
    r.raise_for_status()
    res = r.json()
    s, d = res["summary"], res["drift_check"]
    print(f"[{label:11s}] {plant_id:<14} mean_error={s['mean_error']}% pass8={s['pass_rate_8']} PR={s['pr_mean']}  "
          f"-> status={d.get('status')} action={d.get('action')}"
          + (f" retrain={d['retrain'].get('promoted')} new_error={d['retrain'].get('mean_error')}" if "retrain" in d else ""))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="normal,monsoon,equipment,fleet_shift,real_gwangyang,real_sc2_dec,real_yh5")
    args = ap.parse_args()
    run = set(args.only.split(","))

    if "normal" in run:
        w = fetch_window("samcheonpo_2")
        print(f"[1] 정상: {w['start']}~{w['end']}")
        send("samcheonpo_2", "normal", w["records"])

    if "monsoon" in run:
        w = fetch_window("samcheonpo_2", start=MONSOON_START)
        print(f"[2] 장마철: {w['start']}~{w['end']}")
        send("samcheonpo_2", "monsoon", w["records"], persist=False)

    if "equipment" in run:
        w = fetch_window("samcheonpo_2", scale=0.5, scale_from_day=7)
        print(f"[3] 설비 고장(8일째부터 x0.5): {w['start']}~{w['end']}")
        send("samcheonpo_2", "equipment", w["records"], persist=False)

    if "fleet_shift" in run:
        print(f"[4] 전 발전소 변화(x1.25): {', '.join(FLEET)}")
        for i, pid in enumerate(FLEET):  # 3곳 기록을 모두 쌓은 뒤 마지막에 한 번 판정 (동시성 판단)
            w = fetch_window(pid, scale=1.25)
            send(pid, "fleet_shift", w["records"], check=(i == len(FLEET) - 1))

    for key, pid, start, desc in [("real_gwangyang", "gwangyang_1", "2025-10-05", "광양항 2025-10 실제 저성능"),
                                  ("real_sc2_dec", "samcheonpo_2", "2025-12-05", "삼천포2 2025-12 실제 급락"),
                                  ("real_yh5", "yeongheung5_1", "2026-08-11", "영흥#5 2026-08 실제 점진 손실")]:
        if key in run:
            w = fetch_window(pid, start=start)
            print(f"[실제] {desc}: {w['start']}~{w['end']}")
            send(pid, key, w["records"], persist=False)

    print("[끝] logs/aiops.log 와 /predict/drift-state 를 확인하세요.")


if __name__ == "__main__":
    main()
