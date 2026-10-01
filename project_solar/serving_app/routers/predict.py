"""
[Day1 -> Day3] 예측 API - serving_app/routers/predict.py (SolarCast v2)

   POST /predict             : 발전소 1곳, 다음 날 24시간 발전량 (제도 제출 포맷)
   POST /predict/batch-test  : 시간별 실적을 받아 날짜별로 사후 평가 -> 오차율·PR 기록 -> 드리프트 원인 분류 -> 대응
"""
from datetime import date, timedelta

from fastapi import APIRouter, HTTPException

from data.features import HIST_HOURS, HORIZON, hour_keys, build_future, load_weather, WEATHER_OBS_PATH, WEATHER_SCALE
from data.metrics import day_error_rate, performance_ratio, summarize
from data.storage import append_recent
from serving_app import model_loader
from serving_app.schemas import (PredictRequest, PredictResponse, BatchTestRequest, BatchTestResponse, DayResult)
from serving_app.monitoring.drift_detector import records
from serving_app.monitoring.retrain_trigger import check_and_trigger

router = APIRouter()

_weather_obs: dict | None = None


def weather_obs() -> dict:
    global _weather_obs
    if _weather_obs is None:
        _weather_obs = load_weather(WEATHER_OBS_PATH)
    return _weather_obs


def _resolve_plant(plant_id: str, capacity_kw=None, lat=None, lon=None) -> dict:
    p = model_loader.plants().get(plant_id)
    if p is None:
        if capacity_kw is None or lat is None or lon is None:
            raise HTTPException(422, f"미등록 발전소 '{plant_id}': capacity_kw, lat, lon 을 함께 보내세요.")
        return {"plant_id": plant_id, "capacity_kw": capacity_kw, "lat": lat, "lon": lon,
                "loc": f"{round(lat, 3)}_{round(lon, 3)}"}
    p = dict(p)
    if capacity_kw:
        p["capacity_kw"] = capacity_kw
    return p


@router.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """
    받는 것: plant_id, date(D), history_kwh[72] (D-3~D-1), forecast[24] (D 의 기상 예보)
    돌려줄 것: hourly_kwh[24], day_total_kwh, model_version
    """
    plant = _resolve_plant(req.plant_id, req.capacity_kw, req.lat, req.lon)
    model = model_loader.get_model()
    rows = [[w.ghi, w.cloud_cover, w.temperature] for w in req.forecast]
    hourly = model.predict_day(plant, date.fromisoformat(req.date), req.history_kwh, rows)
    return PredictResponse(plant_id=req.plant_id, date=req.date, hourly_kwh=hourly,
                           day_total_kwh=round(sum(hourly), 1), model_version=model.version)


@router.post("/predict/batch-test", response_model=BatchTestResponse)
def batch_test(req: BatchTestRequest):
    """
    실적 사후 평가 + 드리프트 시뮬레이션.
      records = 연속된 시간별 실적 (최소 72 + 24 시간). 첫 3일은 이력으로만 쓰고, 4일째부터 하루씩:
        이력 72h + 그날 "관측 기상" -> 모델 예측 = 기상 기대치(expected)
        일 오차율 = 제도식, PR = 실제/기대치  -> drift_detector.records 에 누적
      마지막에 check_and_trigger -> ok / weather / equipment / soiling / model_drift(재학습)
    """
    plant = _resolve_plant(req.plant_id)
    model = model_loader.get_model()
    wloc = weather_obs().get(plant["loc"], {})
    cap = plant["capacity_kw"]

    gen = {r.time: r.generation_kwh for r in req.records}
    days = sorted({t[:10] for t in gen})
    results: list[DayResult] = []
    for ds in days[3:]:
        d = date.fromisoformat(ds)
        hist, ok = [], True
        for off in (3, 2, 1):
            for k in hour_keys(d - timedelta(days=off)):
                if k not in gen:
                    ok = False
                    break
                hist.append([min(max(gen[k] / cap, 0.0), 1.0)])
        actual = [gen.get(k) for k in hour_keys(d)]
        if not ok or any(a is None for a in actual):
            continue
        future = build_future(plant, d, wloc)
        if future is None:
            continue
        from data.features import doy_features

        cf = model.predict_cf(hist, future, doy_features(d), req.plant_id)
        expected = [c * cap for c in cf]
        err = day_error_rate(expected, actual, cap)
        pr = performance_ratio(actual, expected, cap)
        records[req.plant_id].append({"date": ds, "day_error": err, "pr": pr, "label": req.label})
        results.append(DayResult(date=ds, actual_kwh=round(sum(actual), 1), expected_kwh=round(sum(expected), 1),
                                 day_error=None if err is None else round(err, 2),
                                 pr=None if pr is None else round(pr, 3)))

    if not results:
        raise HTTPException(400, "평가할 수 있는 날이 없습니다 (이력 72시간 + 그날 24시간 + 기상이 모두 있어야 합니다).")

    if req.persist:
        append_recent(req.plant_id, [{"time": r.time, "generation_kwh": r.generation_kwh} for r in req.records])

    summary = summarize([r.day_error for r in results])
    summary["pr_mean"] = round(sum(r.pr for r in results if r.pr is not None) / max(1, sum(1 for r in results if r.pr is not None)), 3)
    drift_check = check_and_trigger(req.plant_id)
    drift_check["label"] = req.label
    return BatchTestResponse(plant_id=req.plant_id, label=req.label, days=results, summary=summary, drift_check=drift_check)


@router.get("/predict/drift-state")
def drift_state():
    """발전소별 최근 기록 요약 (대시보드·디버깅용)"""
    from serving_app.monitoring.drift_detector import classify

    return {pid: classify(pid) for pid in records}
