"""
Day2: MLflow 로 SolarCast LSTM 을 학습 -> 기록(Tracking) -> 게이트 검증 -> 등록(Registry) -> Production 승격.
Day3: 드리프트 감지 후 Production 가중치에서 이어서 학습하는 fine-tuning 재학습.

HAIC 와 다른 점
    - 게이트 지표: RMSE $4.00 -> nRMSE 8% (설비용량 대비, 발전량 예측제도 정산 기준과 동일)
    - fine-tuning 창: 최근 21거래일 -> 최근 30일 (retrain_trigger.py)

실행:
    python scripts/train_baseline_v1.py     # 최초 1회 (scaler.pkl 생성)
    python serving_app/train_and_register.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlflow
import mlflow.tensorflow
import numpy as np
from mlflow.tracking import MlflowClient
from tensorflow import keras

from data.features import load_rows, build_sequences, train_test_split, SolarScaler, nrmse, CAPACITY_KW
from data.storage import latest_upload
from serving_app.lstm_model import build_model

SEED = 42
keras.utils.set_random_seed(SEED)

NRMSE_GATE = 8.0  # %  - 정산금 기준(오차율 8% 이하)과 같은 선
MODEL_NAME = "SolarCast_Predictor"
SCALER_PATH = "serving_app/models/scaler.pkl"
BASE_EPOCHS = 60
FINE_TUNE_EPOCHS = 10
FINE_TUNE_LR = 1e-4


def _prepare(rows: list[dict], scaler: SolarScaler):
    X, y = build_sequences(rows, scaler)
    X_train, y_train, X_test, y_test = train_test_split(X, y)
    X_train = np.array(X_train, dtype="float32")
    X_test = np.array(X_test, dtype="float32")
    y_train_scaled = np.array([scaler.scale_gen(v) for v in y_train], dtype="float32")
    return X_train, y_train_scaled, X_test, y_test


def _register_if_gate_passed(model, run_id: str, score: float) -> dict:
    result = {"run_id": run_id, "nrmse": score, "promoted": False}
    if score <= NRMSE_GATE:
        v = mlflow.register_model(f"runs:/{run_id}/model", MODEL_NAME)
        MlflowClient().transition_model_version_stage(name=MODEL_NAME, version=v.version, stage="Production")
        result["promoted"] = True
        result["version"] = v.version
        print(f"[GATE PASSED] nrmse={score:.2f}% <= {NRMSE_GATE:.0f}% -> {MODEL_NAME} v{v.version} promoted to Production")
    else:
        print(f"[GATE FAILED] nrmse={score:.2f}% > {NRMSE_GATE:.0f}% -> 배포 차단, 기존 Production 유지")
    return result


def _evaluate(model, scaler, X_test, y_test) -> tuple[float, float]:
    preds = [scaler.inverse_gen(p) for p in model.predict(X_test, verbose=0).flatten()]
    rmse_kwh = float(np.sqrt(np.mean((np.array(y_test) - np.array(preds)) ** 2)))
    return rmse_kwh, nrmse(y_test, preds)


def train_and_register(csv_path: str | None = None, rows: list[dict] | None = None) -> dict:
    """Day2: 처음부터(scratch) 학습. 업로드된 최신 CSV 전체를 사용합니다."""
    if rows is None:
        rows = load_rows(csv_path or latest_upload())
    scaler = SolarScaler.load(SCALER_PATH)
    X_train, y_train_scaled, X_test, y_test = _prepare(rows, scaler)

    with mlflow.start_run(run_name="base-train"):
        model = build_model()
        model.fit(X_train, y_train_scaled, epochs=BASE_EPOCHS, verbose=0)
        rmse_kwh, score = _evaluate(model, scaler, X_test, y_test)

        mlflow.log_param("mode", "scratch")
        mlflow.log_param("epochs", BASE_EPOCHS)
        mlflow.log_param("n_rows", len(rows))
        mlflow.log_param("capacity_kw", CAPACITY_KW)
        mlflow.log_metric("rmse_kwh", rmse_kwh)
        mlflow.log_metric("nrmse_pct", score)
        mlflow.tensorflow.log_model(model, name="model", input_example=X_train[:1])
        print(f"base-train: rmse={rmse_kwh:.0f} kWh, nrmse={score:.2f}% (test {len(y_test)}일)")

        return _register_if_gate_passed(model, mlflow.active_run().info.run_id, score)


def fine_tune(rows: list[dict]) -> dict:
    """Day3: 현재 Production 가중치에서 이어서(warm start) 넘겨받은 최근 rows 로 짧게 fine-tuning."""
    scaler = SolarScaler.load(SCALER_PATH)
    X_train, y_train_scaled, X_test, y_test = _prepare(rows, scaler)

    model = mlflow.tensorflow.load_model(f"models:/{MODEL_NAME}/Production")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=FINE_TUNE_LR), loss="mse")

    with mlflow.start_run(run_name="fine-tune"):
        model.fit(X_train, y_train_scaled, epochs=FINE_TUNE_EPOCHS, verbose=0)
        rmse_kwh, score = _evaluate(model, scaler, X_test, y_test)

        mlflow.log_param("mode", "fine-tune")
        mlflow.log_param("epochs", FINE_TUNE_EPOCHS)
        mlflow.log_param("n_rows", len(rows))
        mlflow.log_metric("rmse_kwh", rmse_kwh)
        mlflow.log_metric("nrmse_pct", score)
        mlflow.tensorflow.log_model(model, name="model", input_example=X_train[:1])
        print(f"fine-tune: rmse={rmse_kwh:.0f} kWh, nrmse={score:.2f}% (test {len(y_test)}일)")

        return _register_if_gate_passed(model, mlflow.active_run().info.run_id, score)


if __name__ == "__main__":
    train_and_register()
