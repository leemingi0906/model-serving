"""
[Day3] 드리프트 -> 원인별 대응 - serving_app/monitoring/retrain_trigger.py (SolarCast v2)

logs/aiops.log 에 남는 줄 (대시보드가 읽으므로 앞머리 토큰은 바꾸지 말 것)
    ok          : (기록 없음)
    weather     : [WARN] drift detected (cause=weather, ...) - forecast quality alert, no retrain
    equipment   : [WARN] drift detected (cause=equipment, ...)  +  [ALERT] equipment anomaly ... - retrain blocked
    soiling     : [WARN] drift detected (cause=soiling, ...)    +  [ALERT] soiling suspected ... - cleaning recommended
    model_drift : [WARN] drift detected - triggering retrain
                  [INFO] retrain triggered (window=last_30_days)
                  [OK] new_error=4.12% - production promoted: SolarCast_Hourly v2     (또는 게이트 실패 [WARN])
"""
import logging
from datetime import date, timedelta

from serving_app.monitoring.drift_detector import PR_EQUIPMENT, classify, records

logger = logging.getLogger("aiops")

FINE_TUNE_DAYS = 30


def _fmt(c: dict) -> str:
    return (f"plant={c['plant_id']}, err={c.get('window_error')}% (thr {c.get('threshold')}%), pr_recent={c.get('pr_recent')}, "
            f"pr_before={c.get('pr_before')}, fleet_share={c.get('fleet_share')}")


def check_and_trigger(plant_id: str) -> dict:
    c = classify(plant_id)
    status, action = c["status"], c["action"]

    if status in ("ok", "insufficient"):
        return c

    if status == "weather":
        logger.warning(f"[WARN] drift detected (cause=weather, {_fmt(c)}) - forecast quality alert, no retrain")
        return c
    if status == "equipment":
        logger.warning(f"[WARN] drift detected (cause=equipment, {_fmt(c)})")
        logger.warning(f"[ALERT] equipment anomaly suspected at {plant_id} (PR {c['pr_recent']}) - retrain blocked, operator notified")
        return c
    if status == "soiling":
        logger.warning(f"[WARN] drift detected (cause=soiling, {_fmt(c)})")
        logger.warning(f"[ALERT] soiling suspected at {plant_id} (PR {c['pr_recent']}) - cleaning recommended, no retrain")
        return c

    # model_drift -> 재학습
    logger.warning(f"[WARN] drift detected - triggering retrain (cause=model_drift, {_fmt(c)})")
    from serving_app.train_and_register import fine_tune
    from serving_app import model_loader

    last_day = max(r["date"] for r in records[plant_id])
    end = date.fromisoformat(last_day)
    start = end - timedelta(days=FINE_TUNE_DAYS - 1)
    # 재학습 대상: 오차가 큰 발전소 전부 (한 발전소만 보고 전체 모델을 고치지 않는다).
    # 단, 같은 시기에 PR 이 설비 이상 범위(< 0.75)인 발전소는 동시성 때문에 model_drift 로 묶여도 제외한다.
    # 고장 난 발전소 실적으로 fine-tune 하면 고장을 정상으로 학습하기 때문.
    affected, excluded = {plant_id}, []
    for pid in records:
        c2 = classify(pid)
        if c2["status"] not in ("model_drift", "weather"):
            continue
        if c2.get("pr_recent") is not None and c2["pr_recent"] < PR_EQUIPMENT:
            excluded.append(f"{pid}(PR {c2['pr_recent']})")
        else:
            affected.add(pid)
    affected = sorted(affected)
    logger.info(f"[INFO] retrain triggered (window=last_{FINE_TUNE_DAYS}_days, plants={','.join(affected)})"
                + (f" - excluded low-PR plants: {', '.join(excluded)}" if excluded else ""))

    result = fine_tune(affected, start.isoformat(), end.isoformat())
    c["retrain"] = result
    if result.get("promoted"):
        logger.info(f"[OK] new_error={result['mean_error']:.2f}% - production promoted: SolarCast_Hourly v{result['version']}")
        model_loader.invalidate_cache()
        for pid in affected:  # 새 모델 기준으로 다시 쌓기 시작
            records[pid].clear()
    else:
        logger.warning(f"[WARN] retrain gate failed (new_error={result.get('mean_error')}) - keeping current Production")
    return c
