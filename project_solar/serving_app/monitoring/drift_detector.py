"""
[Day3] 드리프트 감지 + 원인 분류 - serving_app/monitoring/drift_detector.py (SolarCast v2)

HAIC 실습은 "오차 크면 드리프트 -> 재학습" 한 가지였다. v2 는 두 지표를 같이 본다.
    오차율  : 모델이 틀리고 있나          (최근 21일 일 오차율 평균 > 임계값 = max(8%, 검증 오차 x 1.25))
    성능비 PR: 발전소가 이상한가          (실제 / 기상 기대치, 1.0 이 정상)
그리고 "한 발전소만인가, 여러 발전소가 같이인가"를 가린다.

판정표 (기획안 "드리프트 원인 분류와 대응")
    오차 <= 임계값                                         -> ok          (대응 없음)
    오차 > 임계값, 같은 시기 여러 발전소 동시 또는 PR 이 1.15 초과 -> model_drift (최근 30일 fine-tuning -> 게이트 -> 재배포)
    오차 > 임계값, 장기(60일+) PR 단조 하락 또는 창 안 완만 하락   -> soiling     (오염·열화·점진 손실: 세척/점검 알림, 재학습 안 함)
    오차 > 임계값, PR 최근 7일 < 0.75 (급락이든 지속 저하든)      -> equipment   (설비 이상: 알림, 재학습 금지)
    오차 > 임계값, PR 정상, 이 발전소만                          -> weather     (예보 오차 알림, 재학습 안 함)
  실제 데이터 검증(2026-10-01): 광양항 2025-09~10 PR 0.37 지속 -> equipment, 삼천포2 2025-12 PR 0.96->0.46 급락 -> equipment,
  영흥#5 2026-03~08 PR 0.96->0.64 점진 하락 -> 21일 창에서는 equipment, 장기 기록이 쌓이면 soiling.
"""
import json
import os
from collections import defaultdict

from data.metrics import ERROR_THRESHOLD

# 드리프트 임계값: 검증된 Production 성능(테스트 기간 일 오차율 평균)의 1.25 배, 단 제도 기준 8% 미만으로는 내려가지 않음.
# 모델의 평상시 오차가 9% 인데 8% 를 임계값으로 쓰면 정상 운영에서도 늘 "드리프트"가 뜨므로(통계적 공정관리의 기본),
# "검증 당시보다 25% 이상 나빠졌다"를 알람 조건으로 쓴다. production_metrics.json 은 train_and_register 가 승격 때 쓴다.
DRIFT_FACTOR = 1.25
PRODUCTION_METRICS_PATH = "serving_app/models/production_metrics.json"


def error_threshold() -> float:
    try:
        with open(PRODUCTION_METRICS_PATH, encoding="utf-8") as f:
            base = json.load(f).get("mean_error")
        if base:
            return round(max(ERROR_THRESHOLD, DRIFT_FACTOR * float(base)), 2)
    except (OSError, ValueError):
        pass
    return ERROR_THRESHOLD


WINDOW_DAYS = 21        # 드리프트 판정 창
PR_RECENT_DAYS = 7      # PR "최근" 구간
PR_EQUIPMENT = 0.75     # 이 아래면 설비 이상 의심
PR_EQUIPMENT_DROP = 0.20
PR_SOILING = 0.92
PR_SOILING_DROP = 0.05
PR_FLEET_HIGH = 1.15    # 실제가 기대치보다 계속 많으면 모델이 세상을 못 따라가는 것 (용량 증설·효율 개선·계절)
MIN_DAYS = 7            # 이보다 적으면 판단 보류
FLEET_SHARE = 0.5       # 같은 시기 발전소 중 이 비율 이상이 동시에 오차 초과면 모델 드리프트
FLEET_DATE_TOLERANCE = 7  # "같은 시기" = 최근 기록 날짜 차이 7일 이내
LONG_DAYS = 60          # 장기 추세(오염·열화) 판단에 필요한 최소 누적 일수

# 발전소별 일 단위 기록: {plant_id: [{"date", "day_error", "pr"}, ...]}  (predict.py 가 채운다)
records: dict[str, list[dict]] = defaultdict(list)


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def window(plant_id: str) -> list[dict]:
    return records[plant_id][-WINDOW_DAYS:]


def compute_window_error(plant_id: str) -> float | None:
    return _mean([r["day_error"] for r in window(plant_id)])


def fleet_share_over_threshold(ref_date: str | None = None) -> tuple[float, int]:
    """
    같은 시기(ref_date 기준 ±FLEET_DATE_TOLERANCE 일 안에 최근 기록이 있는) 발전소 중 창 오차가 임계값을 넘는 비율.
    시기가 다른 기록(예: 작년 가을 광양항, 올여름 영흥)을 한데 묶으면 "동시 변화"가 아니므로 날짜로 거른다.
    """
    over, n = 0, 0
    for pid in records:
        w = window(pid)
        if len([r for r in w if r["day_error"] is not None]) < MIN_DAYS:
            continue
        if ref_date and abs(_days_between(w[-1]["date"], ref_date)) > FLEET_DATE_TOLERANCE:
            continue
        n += 1
        if compute_window_error(pid) > error_threshold():
            over += 1
    return (over / n if n else 0.0), n


def _days_between(a: str, b: str) -> int:
    from datetime import date

    return (date.fromisoformat(a) - date.fromisoformat(b)).days


def long_trend(plant_id: str) -> dict | None:
    """
    장기 추세 (기록이 LONG_DAYS 이상 쌓였을 때): 앞 30일 PR 평균 vs 중간 vs 최근 7일 PR 평균.
    21일 창만으로는 "창 시작 전에 이미 떨어진 것"과 "서서히 떨어지는 것"을 못 가르므로, 운영 중 누적된 기록으로 보완한다.
    """
    hist = [r["pr"] for r in records[plant_id] if r["pr"] is not None]
    if len(hist) < LONG_DAYS:
        return None
    first, recent = _mean(hist[:30]), _mean(hist[-PR_RECENT_DAYS:])
    mid = _mean(hist[len(hist) // 2 - 7: len(hist) // 2 + 7])
    gradual = first is not None and mid is not None and recent is not None and first > mid > recent  # 단조 하락
    return {"pr_first30": round(first, 3), "pr_mid": round(mid, 3), "pr_recent": round(recent, 3), "gradual": gradual}


def classify(plant_id: str) -> dict:
    """
    돌려줄 것: {"status": "ok" | "insufficient" | "weather" | "equipment" | "soiling" | "model_drift",
               "action": "none" | "alert" | "retrain", "window_error": .., "pr_recent": .., "pr_before": .., "fleet_share": ..}
    """
    w = window(plant_id)
    errs = [r["day_error"] for r in w if r["day_error"] is not None]
    out = {"plant_id": plant_id, "n_days": len(errs)}
    if len(errs) < MIN_DAYS:
        out.update(status="insufficient", action="none", window_error=_mean(errs))
        return out

    err = _mean(errs)
    prs = [r["pr"] for r in w if r["pr"] is not None]
    pr_recent = _mean(prs[-PR_RECENT_DAYS:]) if prs else None
    pr_before = _mean(prs[:-PR_RECENT_DAYS]) if len(prs) > PR_RECENT_DAYS + 2 else None
    pr_window = _mean(prs) if prs else None
    share, n_plants = fleet_share_over_threshold(ref_date=w[-1]["date"])
    trend = long_trend(plant_id)
    thr = error_threshold()
    out.update(window_error=round(err, 2), threshold=thr, pr_recent=None if pr_recent is None else round(pr_recent, 3),
               pr_window=None if pr_window is None else round(pr_window, 3),
               pr_before=None if pr_before is None else round(pr_before, 3), fleet_share=round(share, 2),
               fleet_plants=n_plants, long_trend=trend)

    if err <= thr:
        out.update(status="ok", action="none")
        return out

    drop = (pr_before - pr_recent) if (pr_before is not None and pr_recent is not None) else None
    # 1) 같은 시기 여러 발전소가 동시에 틀리거나, 실제가 기대치를 계속 크게 웃돌면 -> 모델이 세상을 못 따라가는 것
    if (n_plants >= 2 and share >= FLEET_SHARE) or (pr_window is not None and pr_window > PR_FLEET_HIGH):
        out.update(status="model_drift", action="retrain")
    # 2) 장기 기록이 있고 PR 이 수개월에 걸쳐 단조 하락 -> 오염·열화·점진적 설비 손실 (재학습 금지)
    elif trend and trend["gradual"] and trend["pr_first30"] - trend["pr_recent"] >= PR_SOILING_DROP \
            and pr_recent is not None and pr_recent < PR_SOILING:
        out.update(status="soiling", action="alert")
    # 3) PR 이 낮다 (창 안에서 급락했든, 창 시작 전부터 낮든) -> 설비 이상. 창 안 급락이면 abrupt=True
    elif pr_recent is not None and pr_recent < PR_EQUIPMENT:
        out.update(status="equipment", action="alert", abrupt=bool(drop is not None and drop >= PR_EQUIPMENT_DROP))
    # 4) 창 안에서 완만히 미끄러짐 (0.75~0.92, -0.05 이상)
    elif pr_recent is not None and pr_recent < PR_SOILING and drop is not None and drop >= PR_SOILING_DROP:
        out.update(status="soiling", action="alert")
    # 5) PR 정상인데 오차만 큼 -> 날씨·예보 문제
    else:
        out.update(status="weather", action="alert")
    return out


def is_drift(plant_id: str) -> bool:
    """HAIC 호환: 재학습이 필요한 드리프트인지 True/False"""
    return classify(plant_id)["action"] == "retrain"
