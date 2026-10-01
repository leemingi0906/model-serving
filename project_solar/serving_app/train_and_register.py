"""
Day2: MLflow 로 SolarCast v2 모델을 학습 -> 기록 -> 게이트 검증 -> 등록 -> Production 승격.
Day3: 드리프트 감지 후 Production 가중치에서 이어서 학습하는 fine-tuning.

게이트 = 발전량 예측제도 오차율: 테스트 기간 일 오차율 평균 <= 8%  (data/metrics.py)
        + 통과율(8%, 6%)을 함께 기록해 발표 수치로 쓴다.

실행:
    python scripts/train_baseline_v1.py      # 최초 1회 (scaler.pkl)
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

from data.features import (SolarScaler, load_plants, load_generation, load_weather, build_dataset, split_by_date,
                           PLANTS_PATH, WEATHER_OBS_PATH)
from data.metrics import day_error_rate, summarize, ERROR_THRESHOLD
from data.storage import latest_upload, load_recent
from serving_app.lstm_model import build_model

SEED = 42
keras.utils.set_random_seed(SEED)

GATE_MEAN_ERROR = ERROR_THRESHOLD  # 8.0 %
MODEL_NAME = "SolarCast_Hourly"
SCALER_PATH = "serving_app/models/scaler.pkl"
TEST_SPLIT = "2025-09-01"  # 예측일 기준: 이전 = 학습, 이후 = 테스트 (최근 1년)
BASE_EPOCHS = 40
FINE_TUNE_EPOCHS = 10
FINE_TUNE_LR = 1e-4


def load_training_sources(plant_ids: list[str] | None = None, with_recent: bool = True):
    plants = load_plants(PLANTS_PATH)
    gen = load_generation(latest_upload())
    if with_recent:
        for pid, rows in load_recent().items():  # 운영 중 들어온 실적이 업로드 데이터를 덮어쓴다
            gen.setdefault(pid, {}).update(rows)
    weather = load_weather(WEATHER_OBS_PATH)
    use = [p for p, m in plants.items() if m["use"] and m["hourly_ok"] and (plant_ids is None or p in plant_ids)]
    return plants, {p: gen[p] for p in use if p in gen}, weather


def evaluate(model, Xh, Xf, Xd, Y, meta, plants) -> dict:
    """제도 오차율: 샘플(발전소, 날)마다 일 오차율 -> 평균·통과율. RMSE(이용률)도 참고로."""
    if len(Y) == 0:
        return {"n_days": 0, "mean_error": None, "pass_rate_8": None, "pass_rate_6": None, "rmse_cf": None}
    P = model.predict([Xh, Xf, Xd], verbose=0)
    errs = []
    for p, y, (pid, _) in zip(P, Y, meta):
        cap = plants[pid]["capacity_kw"]
        errs.append(day_error_rate([float(v) * cap for v in p], [float(v) * cap for v in y], cap))
    s = summarize(errs)
    s["rmse_cf"] = round(float(np.sqrt(np.mean((P - Y) ** 2))), 4)
    return s


def _register_if_gate_passed(run_id: str, metrics: dict) -> dict:
    score = metrics["mean_error"]
    result = {"run_id": run_id, "mean_error": score, "pass_rate_8": metrics["pass_rate_8"], "promoted": False}
    if score is not None and score <= GATE_MEAN_ERROR:
        v = mlflow.register_model(f"runs:/{run_id}/model", MODEL_NAME)
        MlflowClient().transition_model_version_stage(name=MODEL_NAME, version=v.version, stage="Production")
        result["promoted"] = True
        result["version"] = v.version
        print(f"[GATE PASSED] mean_error={score:.2f}% <= {GATE_MEAN_ERROR:.0f}% (pass_rate_8={metrics['pass_rate_8']}) "
              f"-> {MODEL_NAME} v{v.version} promoted to Production")
    else:
        print(f"[GATE FAILED] mean_error={score} > {GATE_MEAN_ERROR:.0f}% -> 배포 차단, 기존 Production 유지")
    return result


def _log_metrics(prefix: str, m: dict):
    for k, v in m.items():
        if v is not None:
            mlflow.log_metric(f"{prefix}{k}", v)


def train_and_register() -> dict:
    """Day2: 처음부터(scratch) 학습. hourly_ok 발전소 전체 통합."""
    SolarScaler.load(SCALER_PATH)  # scaler.pkl 존재 확인 (Day1 에서 생성)
    plants, gen, weather = load_training_sources()
    Xh, Xf, Xd, Y, meta = build_dataset(plants, gen, weather)
    tr, te = split_by_date(meta, TEST_SPLIT)
    print(f"samples: train {len(tr)} / test {len(te)}  (plants {len(gen)}, split {TEST_SPLIT})")

    with mlflow.start_run(run_name="base-train"):
        model = build_model()
        model.fit([Xh[tr], Xf[tr], Xd[tr]], Y[tr], epochs=BASE_EPOCHS, batch_size=64, verbose=0)
        m = evaluate(model, Xh[te], Xf[te], Xd[te], Y[te], [meta[i] for i in te], plants)

        mlflow.log_param("mode", "scratch")
        mlflow.log_param("epochs", BASE_EPOCHS)
        mlflow.log_param("n_train", len(tr))
        mlflow.log_param("n_test", len(te))
        mlflow.log_param("plants", ",".join(sorted(gen)))
        _log_metrics("test_", m)
        mlflow.tensorflow.log_model(model, name="model", input_example=[Xh[:1], Xf[:1], Xd[:1]])
        print(f"base-train: {m}")
        return _register_if_gate_passed(mlflow.active_run().info.run_id, m)


def fine_tune(plant_ids: list[str], start: str, end: str) -> dict:
    """
    Day3: Production 가중치에서 warm start, 최근 구간(start~end, 보통 30일)의 실적(업로드 + 운영 중 수신분)으로
    짧게 fine-tuning. 게이트는 그 구간의 마지막 20% 날짜로 평가한다.
    """
    plants, gen, weather = load_training_sources(plant_ids)
    Xh, Xf, Xd, Y, meta = build_dataset(plants, gen, weather, start=start, end=end)
    if len(Y) < 10:
        print(f"fine-tune skipped: 샘플 {len(Y)}개뿐")
        return {"run_id": None, "mean_error": None, "promoted": False, "reason": "insufficient_data"}
    days = sorted({d for _, d in meta})
    split = days[int(len(days) * 0.8)]
    tr, te = split_by_date(meta, split)

    model = mlflow.tensorflow.load_model(f"models:/{MODEL_NAME}/Production")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=FINE_TUNE_LR), loss="mse")

    with mlflow.start_run(run_name="fine-tune"):
        model.fit([Xh[tr], Xf[tr], Xd[tr]], Y[tr], epochs=FINE_TUNE_EPOCHS, batch_size=16, verbose=0)
        m = evaluate(model, Xh[te], Xf[te], Xd[te], Y[te], [meta[i] for i in te], plants)

        mlflow.log_param("mode", "fine-tune")
        mlflow.log_param("epochs", FINE_TUNE_EPOCHS)
        mlflow.log_param("window", f"{start}~{end}")
        mlflow.log_param("plants", ",".join(plant_ids))
        mlflow.log_param("n_train", len(tr))
        _log_metrics("test_", m)
        mlflow.tensorflow.log_model(model, name="model", input_example=[Xh[:1], Xf[:1], Xd[:1]])
        print(f"fine-tune: {m}")
        return _register_if_gate_passed(mlflow.active_run().info.run_id, m)


if __name__ == "__main__":
    train_and_register()
