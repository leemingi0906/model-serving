"""
[Day1 -> Day3] 예측 API - serving_app/routers/predict.py (SolarCast)

   [Day1] POST /predict             : 14일치 일 발전량 -> 다음 날 발전량 1개 (kWh)
   [Day3] POST /predict/batch-test  : 긴 발전량 목록 -> 슬라이딩 윈도우로 여러 번 예측 -> 드리프트 검사
"""
from fastapi import APIRouter

from data.features import SEQ_LEN  # = 14
from serving_app import model_loader
from serving_app.schemas import PredictRequest, PredictResponse, BatchTestRequest, BatchTestResponse
from serving_app.monitoring.drift_detector import WINDOW_SIZE
from serving_app.monitoring.retrain_trigger import check_and_trigger

router = APIRouter()

# (Day3) 최근 예측 기록. 드리프트 판단은 최근 WINDOW_SIZE(21)건만 보므로 그만큼만 유지한다.
recent_predictions: list[dict] = []


@router.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """
    받는 것  : {"sequence": [{"generation_kwh": 3320.8}, ... 14개]}
    돌려줄 것: {"predicted_kwh": 3410.5, "model_version": "production-v1"}
    """
    model = model_loader.get_model()
    sequence = [p.model_dump() for p in req.sequence]
    predicted = model.predict_one(sequence)
    return PredictResponse(predicted_kwh=round(predicted, 1), model_version=model.version)


@router.post("/predict/batch-test", response_model=BatchTestResponse)
def batch_test(req: BatchTestRequest):
    """
    받는 것  : {"values": [3320.8, 3918.0, ... 35개], "label": "monsoon"}
    돌려줄 것: {"predictions": [예측값 21개], "drift_check": {"status": "ok", "nrmse": 5.1} 또는 재학습 결과}

    슬라이딩 윈도우: 값 35개 -> 14개씩 잘라 "그다음 날"을 예측하고 실제 값과 비교 -> 21번 예측
    """
    model = model_loader.get_model()
    predictions: list[float] = []

    values = req.values
    for i in range(len(values) - SEQ_LEN):
        window = values[i : i + SEQ_LEN]
        sequence = [{"generation_kwh": v} for v in window]
        pred = model.predict_one(sequence)
        actual = values[i + SEQ_LEN]
        predictions.append(round(pred, 1))
        recent_predictions.append({"predicted": pred, "actual": actual})

    recent_predictions[:] = recent_predictions[-WINDOW_SIZE:]  # 최근 21건만 유지

    drift_check = check_and_trigger(recent_predictions)
    drift_check["label"] = req.label
    return BatchTestResponse(predictions=predictions, drift_check=drift_check)
