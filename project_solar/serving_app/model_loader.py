"""
[Day1 -> Day2] 모델 불러오기 - serving_app/model_loader.py (SolarCast v2)

환경변수
   LOADING_MODE = lazy(기본) | eager
   MODEL_SOURCE = local(기본, Day1) | mlflow(Day2~)
"""
import os
import time
from datetime import date

from data.features import SolarScaler, doy_features, normalize_weather_row, load_plants, PLANTS_PATH
from data.solar import day_profile

LOCAL_MODEL_PATH = "serving_app/models/solarcast_v2.keras"
SCALER_PATH = "serving_app/models/scaler.pkl"
MODEL_NAME = "SolarCast_Hourly"
MLFLOW_MODEL_URI = f"models:/{MODEL_NAME}/Production"

_model_cache = None
_plants_cache: dict | None = None


def plants() -> dict:
    global _plants_cache
    if _plants_cache is None:
        _plants_cache = load_plants(PLANTS_PATH)
    return _plants_cache


class LoadedModel:
    def __init__(self, keras_model, scaler: SolarScaler, version: str):
        self._keras_model = keras_model
        self.scaler = scaler
        self.version = version

    def predict_cf(self, hist_cf: list[list[float]], future: list[list[float]], doy: list[float]) -> list[float]:
        """정규화된 입력 -> 24시간 이용률"""
        import numpy as np

        out = self._keras_model.predict(
            [np.array([hist_cf], "float32"), np.array([future], "float32"), np.array([doy], "float32")], verbose=0
        )[0]
        return [float(min(max(v, 0.0), 1.0)) for v in out]

    def predict_day(self, plant: dict, day: date, history_kwh: list[float], forecast_rows: list[list[float]]) -> list[float]:
        """
        kWh 단위 입출력. forecast_rows = [[ghi, cloud, temp], ...] 24개 (정규화 전).
        흐름: kWh -> 이용률 변환 -> 기상 정규화 + 태양고도 추가 -> 모델 -> 이용률 x 용량 = kWh
        """
        cap = plant["capacity_kw"]
        hist = [[min(max(v / cap, 0.0), 1.0)] for v in history_kwh]
        elev = day_profile(plant["lat"], plant["lon"], day)
        future = [normalize_weather_row(*row) + [elev[i]] for i, row in enumerate(forecast_rows)]
        cf = self.predict_cf(hist, future, doy_features(day))
        return [round(c * cap, 1) for c in cf]


def _load_from_local() -> LoadedModel:
    from tensorflow import keras

    return LoadedModel(keras.models.load_model(LOCAL_MODEL_PATH), SolarScaler.load(SCALER_PATH), "v2-local")


def _load_from_mlflow() -> LoadedModel:
    import mlflow.tensorflow

    keras_model = mlflow.tensorflow.load_model(MLFLOW_MODEL_URI)
    return LoadedModel(keras_model, SolarScaler.load(SCALER_PATH), _production_version_label())


def _production_version_label() -> str:
    try:
        from mlflow.tracking import MlflowClient

        versions = MlflowClient().get_latest_versions(MODEL_NAME, stages=["Production"])
        if versions:
            return f"production-v{versions[0].version}"
    except Exception:
        pass
    return "production"


def invalidate_cache() -> None:
    """재학습으로 새 Production 이 승격되면 캐시를 비워 다음 요청이 새 모델을 불러오게 한다."""
    global _model_cache
    _model_cache = None


def _load_model() -> LoadedModel:
    return _load_from_mlflow() if os.getenv("MODEL_SOURCE", "local") == "mlflow" else _load_from_local()


def load_eager() -> LoadedModel:
    start = time.time()
    model = _load_model()
    print(f"[eager] model loaded in {time.time() - start:.3f}s at startup")
    global _model_cache
    _model_cache = model
    return model


def get_model() -> LoadedModel:
    global _model_cache
    if _model_cache is None:
        start = time.time()
        _model_cache = _load_model()
        print(f"[lazy] model loaded in {time.time() - start:.3f}s on first request")
    return _model_cache
