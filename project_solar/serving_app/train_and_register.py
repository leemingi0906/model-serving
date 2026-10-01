"""
Day2: MLflow 로 SolarCast v2 모델을 학습 -> 기록 -> 게이트 검증 -> 등록 -> Production 승격.
Day3: 드리프트 감지 후 Production 가중치에서 이어서 학습하는 fine-tuning.

게이트 (기획안 "지표 하나로 통일")
    핵심 지표 = 일 오차율이 8% 이하인 날의 비율(pass_rate_8). 제도는 날마다 정산하므로 "돈을 받는 날의 비율"이
    서비스 KPI 이고, 눈 덮임·부분 정지처럼 기상으로 못 맞히는 날(오차 30~50%)이 평균을 끌어올리는 것에 둔감하다.
    - 첫 배포: pass_rate_8 >= GATE_MIN_PASS_RATE (0.45 = 기준선 persistence/GHI 선형 0.30 의 1.5배)
    - 이후   : 챔피언/챌린저 - 같은 테스트셋에서 현재 Production 의 pass_rate_8 이상 (동률 허용)
    - fine-tune: 최근 구간 held-out 날짜에서 현재 Production 보다 평균 오차가 낮을 때만 승격
    승격 시 serving_app/models/production_metrics.json 에 검증 수치를 남기고, 드리프트 임계값은 그 수치에서 유도한다.

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

GATE_MIN_PASS_RATE = 0.45  # 첫 배포 하한: 기준선(전날 그대로 0.30 / 일사량 선형 0.30) x 1.5
MODEL_NAME = "SolarCast_Hourly"
SCALER_PATH = "serving_app/models/scaler.pkl"
PRODUCTION_METRICS_PATH = "serving_app/models/production_metrics.json"
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


def load_production_model():
    """현재 Production 모델 (없으면 None)"""
    try:
        return mlflow.tensorflow.load_model(f"models:/{MODEL_NAME}/Production")
    except Exception:
        return None


def _save_production_metrics(version, metrics: dict, mode: str):
    import json

    payload = {"version": str(version), "mode": mode, "split": TEST_SPLIT, **{k: v for k, v in metrics.items()}}
    with open(PRODUCTION_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)


def _promote(run_id: str, metrics: dict, mode: str, result: dict, reason: str) -> dict:
    v = mlflow.register_model(f"runs:/{run_id}/model", MODEL_NAME)
    MlflowClient().transition_model_version_stage(name=MODEL_NAME, version=v.version, stage="Production")
    result.update(promoted=True, version=v.version)
    _save_production_metrics(v.version, metrics, mode)
    print(f"[GATE PASSED] {reason} -> {MODEL_NAME} v{v.version} promoted to Production")
    return result


def _register_if_gate_passed(run_id: str, metrics: dict, champion_metrics: dict | None) -> dict:
    """base 학습 게이트: pass_rate_8 이 하한(첫 배포) 또는 챔피언(현 Production) 이상일 때만 승격."""
    pr = metrics["pass_rate_8"]
    result = {"run_id": run_id, "mean_error": metrics["mean_error"], "pass_rate_8": pr, "promoted": False,
              "champion_pass_rate_8": None if champion_metrics is None else champion_metrics["pass_rate_8"]}
    if pr is None:
        print("[GATE FAILED] 평가할 날이 없음"); return result
    if champion_metrics is None:
        if pr >= GATE_MIN_PASS_RATE:
            return _promote(run_id, metrics, "scratch", result,
                            f"pass_rate_8={pr} >= {GATE_MIN_PASS_RATE} (첫 배포 하한, mean_error={metrics['mean_error']}%)")
        print(f"[GATE FAILED] pass_rate_8={pr} < {GATE_MIN_PASS_RATE} -> 배포 차단"); return result
    cp = champion_metrics["pass_rate_8"]
    if pr >= cp:
        return _promote(run_id, metrics, "scratch", result, f"pass_rate_8={pr} >= champion {cp} (mean_error={metrics['mean_error']}%)")
    print(f"[GATE FAILED] pass_rate_8={pr} < champion {cp} -> 배포 차단, 기존 Production 유지"); return result


def _register_finetune_if_better(run_id: str, metrics: dict, champion_metrics: dict) -> dict:
    """fine-tune 게이트: 같은 held-out 날짜에서 현재 Production 보다 평균 오차가 낮아야 승격."""
    new, cur = metrics["mean_error"], champion_metrics["mean_error"]
    result = {"run_id": run_id, "mean_error": new, "pass_rate_8": metrics["pass_rate_8"], "champion_mean_error": cur,
              "promoted": False}
    if new is not None and cur is not None and new <= cur:
        return _promote(run_id, metrics, "fine-tune", result, f"new_error={new:.2f}% <= current {cur:.2f}%")
    print(f"[GATE FAILED] new_error={new} > current {cur} -> 기존 Production 유지"); return result


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

    champion = load_production_model()
    champion_m = evaluate(champion, Xh[te], Xf[te], Xd[te], Y[te], [meta[i] for i in te], plants) if champion else None
    if champion_m:
        print(f"champion(Production) on same test: {champion_m}")

    with mlflow.start_run(run_name="base-train"):
        model = build_model()
        es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
        model.fit([Xh[tr], Xf[tr], Xd[tr]], Y[tr], epochs=BASE_EPOCHS, batch_size=64, verbose=0,
                  validation_split=0.1, callbacks=[es])
        m = evaluate(model, Xh[te], Xf[te], Xd[te], Y[te], [meta[i] for i in te], plants)

        mlflow.log_param("mode", "scratch")
        mlflow.log_param("epochs", len(model.history.history["loss"]))
        mlflow.log_param("n_train", len(tr))
        mlflow.log_param("n_test", len(te))
        mlflow.log_param("plants", ",".join(sorted(gen)))
        _log_metrics("test_", m)
        mlflow.tensorflow.log_model(model, name="model", input_example=[Xh[:1], Xf[:1], Xd[:1]])
        print(f"base-train: {m}")
        return _register_if_gate_passed(mlflow.active_run().info.run_id, m, champion_m)


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
    champion_m = evaluate(model, Xh[te], Xf[te], Xd[te], Y[te], [meta[i] for i in te], plants)
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
        _log_metrics("champion_", champion_m)
        mlflow.tensorflow.log_model(model, name="model", input_example=[Xh[:1], Xf[:1], Xd[:1]])
        print(f"fine-tune: {m}  (current Production on same days: {champion_m})")
        return _register_finetune_if_better(mlflow.active_run().info.run_id, m, champion_m)


if __name__ == "__main__":
    train_and_register()
