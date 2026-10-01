"""
[Day3] 드리프트 -> 원인별 대응 - serving_app/monitoring/retrain_trigger.py (SolarCast v2)

logs/aiops.log 에 남는 줄 (대시보드가 읽으므로 앞머리 토큰은 바꾸지 말 것)
    ok          : (기록 없음)
    weather     : [WARN] drift detected (cause=weather, ...) - forecast quality alert, no retrain
    equipment   : [WARN] drift detected (cause=equipment, ...)  +  [ALERT] equipment anomaly ... - retrain blocked
    soiling     : [WARN] drift detected (cause=soiling, ...)    +  [ALERT] soiling suspected ... - cleaning recommended
    model_drift : [WARN] drift detected - triggering retrain
                  [INFO] retrain triggered (window=last_30_days, plants=..., job=...)
                  [OK] new_error=4.12% - production promoted: SolarCast_Hourly v2     (또는 게이트 실패 [WARN])

재학습은 백그라운드 작업(monitoring/jobs.py)으로 돈다. 요청은 job_id 를 받고 바로 끝나며, 학습 중에도
/predict 는 기존 Production 으로 응답한다. RETRAIN_MODE=sync 면 요청 안에서 끝까지 기다린다 (스크립트·테스트용).
"""
import logging
import os
from datetime import date, timedelta

from serving_app.monitoring import jobs
from serving_app.monitoring.drift_detector import PR_EQUIPMENT, classify, records

logger = logging.getLogger("aiops")

FINE_TUNE_DAYS = 30


def _fmt(c: dict) -> str:
    return (f"plant={c['plant_id']}, err={c.get('window_error')}% (thr {c.get('threshold')}%), pr_recent={c.get('pr_recent')}, "
            f"pr_before={c.get('pr_before')}, fleet_share={c.get('fleet_share')}")


def select_retrain_plants(plant_id: str, c: dict) -> tuple[list[str], list[str]]:
    """
    재학습 대상: 같은 시기에 model_drift/weather 인 발전소 전부 (한 발전소만 보고 전체 모델을 고치지 않는다).
    단, PR 이 설비 이상 범위(< 0.75)인 발전소는 동시성 때문에 묶여도 제외한다 - 고장 실적로 fine-tune 하면 고장을
    정상으로 학습하기 때문. 저PR 발전소가 후보의 절반 이상이면 공통 원인(기후·제도 변화)이므로 전부 포함한다.
    """
    candidates = {plant_id: c}
    for pid in records:
        if pid == plant_id:
            continue
        c2 = classify(pid)
        if c2["status"] in ("model_drift", "weather"):
            candidates[pid] = c2
    low = {pid for pid, cc in candidates.items() if cc.get("pr_recent") is not None and cc["pr_recent"] < PR_EQUIPMENT}
    if len(low) * 2 >= len(candidates):
        low = set()
    excluded = [f"{pid}(PR {candidates[pid]['pr_recent']})" for pid in sorted(low)]
    return sorted(set(candidates) - low), excluded


def _retrain_job(affected: list[str], start: str, end: str) -> dict:
    from serving_app import model_loader
    from serving_app.train_and_register import fine_tune

    result = fine_tune(affected, start, end)
    if result.get("promoted"):
        logger.info(f"[OK] new_error={result['mean_error']:.2f}% - production promoted: SolarCast_Hourly v{result['version']}")
        model_loader.invalidate_cache()
        for pid in affected:  # 새 모델 기준으로 다시 쌓기 시작
            records[pid].clear()
    else:
        logger.warning(f"[WARN] retrain gate failed (new_error={result.get('mean_error')}) - keeping current Production")
    return result


def check_and_trigger(plant_id: str) -> dict:
    c = classify(plant_id)
    status = c["status"]

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
    end = date.fromisoformat(max(r["date"] for r in records[plant_id]))
    start = end - timedelta(days=FINE_TUNE_DAYS - 1)
    affected, excluded = select_retrain_plants(plant_id, c)
    meta = {"kind": "fine_tune", "plants": affected, "window": f"{start.isoformat()}~{end.isoformat()}"}

    runner = jobs.run_sync if os.getenv("RETRAIN_MODE", "async") == "sync" else jobs.submit
    job = runner(lambda: _retrain_job(affected, start.isoformat(), end.isoformat()), **meta)
    if job.get("deduplicated"):
        logger.info(f"[INFO] retrain already running (job={job['job_id']}) - request merged")
    else:
        logger.info(f"[INFO] retrain triggered (window=last_{FINE_TUNE_DAYS}_days, plants={','.join(affected)}, job={job['job_id']})"
                    + (f" - excluded low-PR plants: {', '.join(excluded)}" if excluded else ""))
    c["retrain"] = {"job_id": job.get("job_id"), "status": job.get("status"), "plants": affected,
                    **({k: v for k, v in job.get("result", {}).items()} if job.get("status") == "completed" else {})}
    return c
