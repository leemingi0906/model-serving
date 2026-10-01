"""
v2 시간별 모델 2차 변형: 발전소 임베딩 / 낮 시간 가중 손실 / 앙상블.  project_solar/ 에서 실행.
    python ../team_solar/experiments/hourly_round2.py [emb,wloss,emb_wloss,ens3]
"""
import os, sys, json, time
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.path.insert(0, os.getcwd())
import numpy as np
import tensorflow as tf
from tensorflow import keras
from data.features import (load_plants, load_generation, load_weather, build_dataset, split_by_date, HIST_HOURS, HORIZON,
                           WEATHER_OBS_PATH, PLANTS_PATH)
from data.metrics import day_error_rate, summarize
from serving_app.train_and_register import TEST_SPLIT

keras.utils.set_random_seed(42)
plants = load_plants(PLANTS_PATH)
gen = load_generation("data/sample_solar_hourly.csv.gz")
weather = load_weather(WEATHER_OBS_PATH)
use = {p: g for p, g in gen.items() if plants[p]["hourly_ok"]}
PLANT_IDS = sorted(use)
Xh, Xf, Xd, Y, meta = build_dataset(plants, use, weather)
tr, te = split_by_date(meta, TEST_SPLIT)
Xp = np.zeros((len(meta), len(PLANT_IDS)), "float32")
for i, (pid, _) in enumerate(meta):
    Xp[i, PLANT_IDS.index(pid)] = 1.0
print("samples", len(tr), len(te), "plants", len(PLANT_IDS))


def daytime_weighted_mae(y_true, y_pred):
    w = tf.where(y_true >= 0.10, 1.0, 0.15)  # 제도 평가 대상(이용률 10% 이상) 시간에 집중
    return tf.reduce_sum(w * tf.abs(y_true - y_pred), axis=-1) / tf.reduce_sum(w, axis=-1)


def build(use_emb, loss, width=1.0):
    hi = keras.layers.Input(shape=(HIST_HOURS, 1)); fi = keras.layers.Input(shape=(HORIZON, Xf.shape[2])); di = keras.layers.Input(shape=(2,))
    ins = [hi, fi, di]
    h = keras.layers.LSTM(int(32 * width), return_sequences=True)(hi); h = keras.layers.LSTM(int(16 * width))(h)
    f = keras.layers.Flatten()(fi); f = keras.layers.Dense(int(48 * width), activation="relu")(f)
    parts = [h, f, di]
    if use_emb:
        pi = keras.layers.Input(shape=(len(PLANT_IDS),)); ins.append(pi)
        parts.append(keras.layers.Dense(8, activation="relu")(pi))
    z = keras.layers.Concatenate()(parts); z = keras.layers.Dense(int(64 * width), activation="relu")(z)
    o = keras.layers.Dense(HORIZON, activation="sigmoid")(z)
    m = keras.Model(ins, o); m.compile(optimizer=keras.optimizers.Adam(1e-3), loss=loss)
    return m


def inputs(idx, use_emb):
    x = [Xh[idx], Xf[idx], Xd[idx]]
    return x + [Xp[idx]] if use_emb else x


def evaluate(P, idx):
    errs = []
    for p, i in zip(P, idx):
        cap = plants[meta[i][0]]["capacity_kw"]
        errs.append(day_error_rate([float(v) * cap for v in p], [float(v) * cap for v in Y[i]], cap))
    return summarize(errs)


def train_one(use_emb, loss, seed, width=1.0):
    keras.utils.set_random_seed(seed)
    m = build(use_emb, loss, width)
    cb = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)]
    m.fit(inputs(tr, use_emb), Y[tr], epochs=120, batch_size=64, verbose=0, validation_split=0.1, callbacks=cb)
    return m, len(m.history.history["loss"])


results = {}
variants = {
    "emb": dict(emb=True, loss="mse"),
    "wloss": dict(emb=False, loss=daytime_weighted_mae),
    "emb_wloss": dict(emb=True, loss=daytime_weighted_mae),
    "emb_wloss_wide": dict(emb=True, loss=daytime_weighted_mae, width=1.5),
    "ens3": dict(emb=True, loss=daytime_weighted_mae, seeds=[1, 2, 3]),
}
only = sys.argv[1].split(",") if len(sys.argv) > 1 else list(variants)
for name in only:
    v = variants[name]; t0 = time.time()
    seeds = v.get("seeds", [42])
    preds, ep = [], []
    for s in seeds:
        m, e = train_one(v["emb"], v["loss"], s, v.get("width", 1.0))
        preds.append(m.predict(inputs(te, v["emb"]), verbose=0)); ep.append(e)
        m.save(f"../team_solar/experiments/hourly2_{name}_s{s}.keras")
    P = np.mean(preds, axis=0)
    results[name] = evaluate(P, te); results[name]["epochs_run"] = ep; results[name]["minutes"] = round((time.time() - t0) / 60, 1)
    print(name, results[name], flush=True)
    json.dump(results, open("../team_solar/experiments/hourly_round2_results.json", "w"), indent=1, ensure_ascii=False)
print("-> team_solar/experiments/hourly_round2_results.json")
