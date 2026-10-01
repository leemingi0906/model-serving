"""
Day3 드리프트 감지 시뮬레이션 (SolarCast)

HAIC 실습은 랜덤워크로 "변동성 3배" 드리프트를 만들었지만, 태양광은 실제 실적 데이터가 있으므로
기획안의 드리프트 신호 후보를 그대로 실제 구간/실제 사건으로 재현합니다.

    배치           데이터                                  기대 결과
    normal         업로드 CSV 의 마지막 35일 (늦여름)         nRMSE <= 8%  -> ok
    monsoon        장마철 2026-06-20 ~ 07-24 (35일, 실제)      계절 변동은 학습 분포 안 -> 대개 ok (경계 관찰용)
    new_plant      마지막 35일 x 1.8 (신규 발전소 편입 가정)     학습 범위 밖 -> nRMSE > 8% -> 재학습 트리거

사전 준비: uvicorn serving_app.main:app --port 8010 서버가 떠 있고, CSV 가 업로드되어 있어야 합니다.
실행: python scripts/simulate_drift.py
"""
import os
import sys

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.features import load_rows, SEQ_LEN, DAILY_CAPACITY_KWH
from data.storage import latest_upload
from serving_app.monitoring.drift_detector import WINDOW_SIZE

API_URL = os.getenv("API_URL", "http://localhost:8010/predict/batch-test")

BATCH_N = SEQ_LEN + WINDOW_SIZE  # 14 + 21 = 35 -> 배치 하나당 정확히 21개의 (predicted, actual) 쌍
MONSOON_START = "2026-06-20"
NEW_PLANT_SCALE = 1.8  # 집합자원에 0.8 MW 급 발전소가 추가로 편입된 상황


def pick_range(rows: list[dict], start: str, n: int = BATCH_N) -> list[float]:
    idx = next(i for i, r in enumerate(rows) if r["Date"] >= start)
    return [r["Gen"] for r in rows[idx : idx + n]]


def send_batch(values: list[float], label: str) -> dict:
    """배치를 /predict/batch-test 에 일괄 전송하고 드리프트 판정 결과를 출력한다."""
    values = [min(v, DAILY_CAPACITY_KWH) for v in values]  # 스키마 상한(설비용량x24h) 안으로
    resp = requests.post(API_URL, json={"values": values, "label": label}, timeout=900)
    resp.raise_for_status()
    result = resp.json()
    print(f"[{label:9s}] drift_check = {result['drift_check']}")
    return result


def main():
    rows = load_rows(latest_upload())
    print(f"[1] 업로드 데이터: {rows[0]['Date']} ~ {rows[-1]['Date']} ({len(rows)}일)")

    print("[2] 정상 입력(최근 35일) 전송...")
    normal = [r["Gen"] for r in rows[-BATCH_N:]]
    send_batch(normal, label="normal")

    print(f"[3] 장마철 실제 구간({MONSOON_START}~) 전송...")
    send_batch(pick_range(rows, MONSOON_START), label="monsoon")

    print(f"[4] 신규 발전소 편입(x{NEW_PLANT_SCALE}) 전송...")
    send_batch([v * NEW_PLANT_SCALE for v in normal], label="new_plant")

    print("[5] 결과 확인: logs/aiops.log 에서 [WARN] drift detected -> [INFO] retrain triggered -> [OK] 를 확인하세요.")


if __name__ == "__main__":
    main()
