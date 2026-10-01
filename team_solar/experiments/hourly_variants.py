"""
v2 시간별 모델 변형 비교 (게이트 8% 를 넘기기 위한 탐색). project_solar/ 에서 실행:
    python ../team_solar/experiments/hourly_variants.py

변형
  base     : 현재 구성 (ghi, cloud, temp, sin_elev), mse, 40 epoch
  long     : base + 120 epoch + early stopping (val 10%)
  feat     : future 에 direct/diffuse/clear-sky index(ghi / (1000*sin_elev)) 추가, 120 epoch + ES
  feat_mae : feat + MAE loss (제도 지표가 절대오차이므로)
참고 기준선
  persist  : 전날 같은 시간 발전량
  ghi_lin  : 발전소별 CF = a * ghi_norm (최소제곱) - "기상만으로" 가능한 수준
"""
import os
import sys
import json
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.path.insert(0, os.getcwd())
import numpy as np
from tensorflow import keras

from data.features import (load_plants, load_generation, load_weather, build_dataset, split_by_date, HIST_HOURS, HORIZON,
                           WEATHER_OBS_PATH, PLANTS_PATH, hour_keys, gen_time_to_weather_time)
from data.metrics import day_error_rate, summarize
from serving_app.train_and_register import TEST_SPLIT

keras.utils.set_random_seed(42)
plants = load_plants(PLANTS_PATH)
gen = load_generation("data/sample_solar_hourly.csv.gz")
weather = load_weather(WEATHER_OBS_PATH)
use = {p: g for p, g in gen.items() if plants[p]["hourly_ok"]}
Xh, Xf, Xd, Y, meta = build_dataset(plants, use, weather)
tr, te = split_by_date(meta, TEST_SPLIT)
print("samples", len(tr), len(te))

# ---- 추가 피처 (direct, diffuse, clear-sky index) : obs 원본에서 다시 읽음
import csv, gzip
extra = {}
with gzip.open(WEATHER_OBS_PATH, "rt", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        pass  # obs.csv.gz 에는 3개 변수만 있음 -> team_solar 원본에서 direct/diffuse 를 읽는다
raw = {}
for pid, p in plants.items():
    if not p["hourly_ok"]:
        continue
    lat, lon = p["loc"].split("_")
    path = f"../team_solar/data_raw/weather/obs_{lat}_{lon}.csv"
    if p["loc"] in raw or not os.path.exists(path):
        continue
    raw[p["loc"]] = {r["time"]: (float(r["direct_radiation"] or 0), float(r["diffuse_radiation"] or 0)) for r in csv.DictReader(open(path))}


def extra_future(meta_rows, Xf_base):
    out = np.zeros((len(meta_rows), HORIZON, Xf_base.shape[2] + 3), "float32")
    out[:, :, : Xf_base.shape[2]] = Xf_base
    from datetime import date
    for i, (pid, ds) in enumerate(meta_rows):
        loc = plants[pid]["loc"]
        for h, k in enumerate(hour_keys(date.fromisoformat(ds))):
            d_, f_ = raw[loc].get(gen_time_to_weather_time(k), (0.0, 0.0))
            ghi = Xf_base[i, h, 0]; se = Xf_base[i, h, 3]
            out[i, h, -3] = d_ / 1000.0
            out[i, h, -2] = f_ / 1000.0
            out[i, h, -1] = min(ghi / (se + 0.05), 1.5)  # clear-sky index 근사
    return out


def build(n_future, loss):
    hi = keras.layers.Input(shape=(HIST_HOURS, 1)); fi = keras.layers.Input(shape=(HORIZON, n_future)); di = keras.layers.Input(shape=(2,))
    h = keras.layers.LSTM(32, return_sequences=True)(hi); h = keras.layers.LSTM(16)(h)
    f = keras.layers.Flatten()(fi); f = keras.layers.Dense(48, activation="relu")(f)
    z = keras.layers.Concatenate()([h, f, di]); z = keras.layers.Dense(64, activation="relu")(z)
    o = keras.layers.Dense(HORIZON, activation="sigmoid")(z)
    m = keras.Model([hi, fi, di], o); m.compile(optimizer=keras.optimizers.Adam(1e-3), loss=loss)
    return m


def evaluate(P, idx):
    errs = []
    for p, i in zip(P, idx):
        pid = meta[i][0]; cap = plants[pid]["capacity_kw"]
        errs.append(day_error_rate([float(v) * cap for v in p], [float(v) * cap for v in Y[i]], cap))
    return summarize(errs)


results = {}
# 기준선 1: persistence (전날 같은 시간) = hist 의 마지막 24개
P = Xh[te][:, -24:, 0]
results["persist"] = evaluate(P, te); print("persist ", results["persist"])
# 기준선 2: ghi 선형 (발전소별 a)
P = np.zeros((len(te), 24), "float32")
for pid in use:
    tr_i = [i for i in tr if meta[i][0] == pid]
    g = Xf[tr_i][:, :, 0].flatten(); y = Y[tr_i].flatten()
    a = float((g * y).sum() / max((g * g).sum(), 1e-6))
    for j, i in enumerate(te):
        if meta[i][0] == pid:
            P[j] = np.clip(a * Xf[i][:, 0], 0, 1)
results["ghi_lin"] = evaluate(P, te); print("ghi_lin ", results["ghi_lin"])

Xf2 = extra_future(meta, Xf)
variants = {
    "base": dict(Xf=Xf, loss="mse", epochs=40, es=False),
    "long": dict(Xf=Xf, loss="mse", epochs=120, es=True),
    "feat": dict(Xf=Xf2, loss="mse", epochs=120, es=True),
    "feat_mae": dict(Xf=Xf2, loss="mae", epochs=120, es=True),
}
only = sys.argv[1].split(",") if len(sys.argv) > 1 else list(variants)
for name in only:
    v = variants[name]
    t0 = time.time()
    keras.utils.set_random_seed(42)
    m = build(v["Xf"].shape[2], v["loss"])
    cb = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)] if v["es"] else []
    m.fit([Xh[tr], v["Xf"][tr], Xd[tr]], Y[tr], epochs=v["epochs"], batch_size=64, verbose=0,
          validation_split=0.1 if v["es"] else 0.0, callbacks=cb)
    P = m.predict([Xh[te], v["Xf"][te], Xd[te]], verbose=0)
    results[name] = evaluate(P, te)
    results[name]["epochs_run"] = len(m.history.history["loss"])
    results[name]["minutes"] = round((time.time() - t0) / 60, 1)
    print(name, results[name], flush=True)
    m.save(f"../team_solar/experiments/hourly_{name}.keras")

json.dump(results, open("../team_solar/experiments/hourly_variants_results.json", "w"), indent=1, ensure_ascii=False)
print("-> team_solar/experiments/hourly_variants_results.json")
