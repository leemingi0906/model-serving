"""
[Day1 -> Day2] 모델 불러오기 - serving_app/model_loader.py (SolarCast)

서버가 예측에 쓸 모델을 "어디서(local / mlflow), 언제(lazy / eager)" 불러올지 정하고 예측 한 건을 수행합니다.
HAIC 실습과 구조가 같고, 스케일러 도구 이름만 SolarScaler(transform_point / inverse_gen) 로 바뀌었습니다.

환경변수
   LOADING_MODE = lazy(기본) | eager
   MODEL_SOURCE = local(기본, Day1) | mlflow(Day2~)
"""
import os
import time

from data.features import SolarScaler

LOCAL_MODEL_PATH = "serving_app/models/solarcast_v1.keras"
SCALER_PATH = "serving_app/models/scaler.pkl"
MODEL_NAME = "SolarCast_Predictor"
MLFLOW_MODEL_URI = f"models:/{MODEL_NAME}/Production"

_model_cache = None


class LoadedModel:
    """모델 + 스케일러 + 버전을 한 묶음으로 포장한 상자."""

    def __init__(self, keras_model, scaler: SolarScaler, version: str):
        self._keras_model = keras_model
        self.scaler = scaler
        self.version = version

    def predict_one(self, sequence: list[dict]) -> float:
        """
        14일치 일 발전량으로 다음 날 발전량 1개를 예측합니다.
        받는 것  : sequence = [{"generation_kwh": 3320.8}, ... 14개]  (오래된 날 -> 최근 날)
        돌려줄 것: 다음 날 예상 발전량 (kWh, 0 미만은 0 으로 자름)
        """
        import numpy as np

        scaled = [self.scaler.transform_point(p["generation_kwh"]) for p in sequence]  # ① 0~1 변환
        x = np.array([scaled], dtype="float32")  # ② (1, SEQ_LEN, 1)
        pred_scaled = float(self._keras_model.predict(x, verbose=0)[0][0])  # ③ 예측 (0~1)
        return max(0.0, self.scaler.inverse_gen(pred_scaled))  # ④ kWh 복원 (발전량은 음수 불가)


def _load_from_local() -> LoadedModel:
    """Day1: 로컬 파일에서 모델과 스케일러를 불러온다."""
    from tensorflow import keras

    keras_model = keras.models.load_model(LOCAL_MODEL_PATH)
    scaler = SolarScaler.load(SCALER_PATH)
    return LoadedModel(keras_model=keras_model, scaler=scaler, version="v1-local")


def _load_from_mlflow() -> LoadedModel:
    """Day2: MLflow Model Registry 의 Production 단계 모델 + 로컬 scaler.pkl"""
    import mlflow.tensorflow

    keras_model = mlflow.tensorflow.load_model(MLFLOW_MODEL_URI)
    scaler = SolarScaler.load(SCALER_PATH)  # 스케일러는 Day1에 fit 한 로컬 파일 그대로
    return LoadedModel(keras_model=keras_model, scaler=scaler, version=_production_version_label())


def _production_version_label() -> str:
    """재배포 확인용: 현재 Production 단계의 레지스트리 버전 번호를 라벨에 붙인다 (예: production-v2)."""
    try:
        from mlflow.tracking import MlflowClient

        versions = MlflowClient().get_latest_versions(MODEL_NAME, stages=["Production"])
        if versions:
            return f"production-v{versions[0].version}"
    except Exception:
        pass
    return "production"


def invalidate_cache() -> None:
    """Day3: 재학습으로 새 Production 이 승격되면 캐시를 비워, 다음 /predict 가 새 모델을 다시 불러오게 한다."""
    global _model_cache
    _model_cache = None


def _load_model() -> LoadedModel:
    source = os.getenv("MODEL_SOURCE", "local")
    if source == "mlflow":
        return _load_from_mlflow()
    return _load_from_local()


def load_eager() -> LoadedModel:
    """Eager Loading: 서버가 켜질 때(main.py 의 startup) 바로 불러온다."""
    start = time.time()
    model = _load_model()
    print(f"[eager] model loaded in {time.time() - start:.3f}s at startup")
    global _model_cache
    _model_cache = model
    return model


def get_model() -> LoadedModel:
    """Lazy Loading: 첫 요청이 들어올 때만 불러오고, 이후에는 캐시를 재사용한다."""
    global _model_cache
    if _model_cache is None:
        start = time.time()
        _model_cache = _load_model()
        print(f"[lazy] model loaded in {time.time() - start:.3f}s on first request")
    return _model_cache
