"""
[Day3] 드리프트 -> 자동 재학습 - serving_app/monitoring/retrain_trigger.py (SolarCast)

전체 흐름
   드리프트 감지(nRMSE > 8%) -> 경고 로그 -> 최근 30일 데이터 -> fine-tuning
     -> 게이트 통과(nRMSE <= 8%)? - 예   -> 새 버전 Production 승격 + 캐시 비움 + 성공 로그
                                   - 아니오 -> 기존 Production 유지 (서비스는 멈추지 않음)

logs/aiops.log 에 아래 3줄이 순서대로 찍히면 성공입니다 (대시보드가 이 문장을 읽으므로 글자를 바꾸지 마세요).
     [WARN] drift detected - triggering retrain
     [INFO] retrain triggered (window=last_30_days)
     [OK] new_nrmse=4.12% - production promoted: SolarCast_Predictor v2
"""
import logging

from serving_app.monitoring.drift_detector import is_drift, compute_nrmse, WINDOW_SIZE

logger = logging.getLogger("aiops")

FINE_TUNE_DAYS = 30  # 기획안 운영 설계: 최근 30일로 fine-tuning


def check_and_trigger(recent_predictions: list[dict]) -> dict:
    """
    받는 것  : 최근 예측 기록 [{"predicted": ..., "actual": ...}, ...]
    돌려줄 것: {"status": "ok", "nrmse": 5.1}  또는
               {"status": "retrain_triggered", "promoted": True/False, "nrmse": 9.3, "new_nrmse": 4.1}
    """
    window_nrmse = round(compute_nrmse(recent_predictions[-WINDOW_SIZE:]), 2)
    if not is_drift(recent_predictions):
        return {"status": "ok", "nrmse": window_nrmse}

    logger.warning("[WARN] drift detected - triggering retrain")

    from data.features import load_rows, SEQ_LEN
    from data.storage import latest_upload
    from serving_app.train_and_register import fine_tune

    logger.info(f"[INFO] retrain triggered (window=last_{FINE_TUNE_DAYS}_days)")

    # 최근 30일의 정답으로 학습하려면 창문(14일)까지 포함해 44행이 필요
    rows = load_rows(latest_upload())[-(SEQ_LEN + FINE_TUNE_DAYS):]
    result = fine_tune(rows)  # Production 가중치에서 warm start

    if result["promoted"]:
        logger.info(
            f"[OK] new_nrmse={result['nrmse']:.2f}% - production promoted: SolarCast_Predictor v{result['version']}"
        )
        from serving_app import model_loader

        model_loader.invalidate_cache()  # 다음 /predict 가 새 Production 을 불러오게
        return {"status": "retrain_triggered", "promoted": True, "nrmse": window_nrmse, "new_nrmse": result["nrmse"]}

    logger.warning(f"[WARN] retrain gate failed: new_nrmse={result['nrmse']:.2f}% > 8% - keeping current Production")
    return {"status": "retrain_triggered", "promoted": False, "nrmse": window_nrmse, "new_nrmse": result["nrmse"]}
