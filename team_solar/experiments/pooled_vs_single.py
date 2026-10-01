"""
실험: 발전소 통합 모델(이용률 정규화) vs 삼천포 2호기 단일 모델

    1) data_raw/monthly/*.csv -> 발전소별 일 발전량 (1~24시 합)  -> data/all_plants_daily.csv
    2) 설비용량 추정 = 기간 중 시간 발전량 최대치(kW)   (공식 용량이 확인되면 교체)
       이용률(CF) = 일 발전량 / (용량 x 24h)  -> 발전소 크기와 무관한 0~1 값, nRMSE 와 같은 단위
    3) 모델 (전부 project_solar 와 같은 3층 LSTM, SEQ_LEN=14, 입력 CF 1개, 60 epoch)
       A. single   : 삼천포 2호기 데이터만 학습
       B. pooled   : 모든 발전소 데이터를 합쳐 학습
       C. pooled+ft: B 가중치에서 삼천포 2호기 최근 데이터로 10 epoch warm-start  (Day3 fine_tune 과 동일)
       D. holdout  : 한 발전소를 학습에서 완전히 뺀 pooled 모델로 그 발전소 예측 (신규 발전소 편입 시나리오)
    4) 평가: 2025-09-01 ~ 2026-08-31, 지표 = nRMSE(%) = RMSE(CF) x 100  (= RMSE_kWh / (용량x24) x 100)

실행: python team_solar/experiments/pooled_vs_single.py   (저장소 루트에서, 3~5분)
"""
import csv
import glob
import json
import math
import os
import sys

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
from tensorflow import keras  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # team_solar/
SEQ_LEN = 14
SPLIT = "2025-09-01"
TARGET = ("삼천포태양광", "2")
HOLDOUT = ("광양항세방태양광", "1")
EPOCHS = 60
FT_EPOCHS, FT_LR, FT_DAYS = 10, 1e-4, 90
SEED = 42


# ---------- 1) 원본 -> 발전소별 일 발전량 ----------
def load_all_plants() -> dict[tuple, dict[str, list[float]]]:
    plants: dict[tuple, dict[str, list[float]]] = {}
    for f in sorted(glob.glob(os.path.join(ROOT, "data_raw", "monthly", "*.csv"))):
        with open(f, encoding="utf-8-sig") as fh:
            rd = csv.reader(fh)
            next(rd)
            for r in rd:
                r = [c.strip() for c in r]
                if len(r) < 27:
                    continue
                key = (r[0], r[1])
                hourly = [float(x or 0) for x in r[3:27]]
                plants.setdefault(key, {})[r[2][:10]] = hourly
    return plants


def main():
    keras.utils.set_random_seed(SEED)
    plants = load_all_plants()
    all_days = sorted({d for p in plants.values() for d in p})
    print(f"발전소 {len(plants)}곳, 기간 {all_days[0]} ~ {all_days[-1]} ({len(all_days)}일)")

    # 커버리지·용량 추정, 일 발전량 CSV 저장
    rows_out = []
    series: dict[tuple, tuple[list[str], np.ndarray, float]] = {}
    print(f"\n{'발전소':<18}{'호기':>4}{'일수':>6}{'0kWh일':>8}{'추정용량kW':>11}{'평균CF':>8}")
    for key, byday in sorted(plants.items()):
        days = sorted(byday)
        daily = np.array([sum(byday[d]) for d in days])
        cap = max(max(byday[d]) for d in days)
        zero_days = int((daily <= 0).sum())
        coverage_ok = days[0] <= "2023-01-03" and days[-1] >= "2026-08-29" and len(days) >= 1300
        mean_cf = float(daily.mean() / (cap * 24)) if cap > 0 else 0
        print(f"{key[0]:<18}{key[1]:>4}{len(days):>6}{zero_days:>8}{cap:>11.0f}{mean_cf*100:>7.1f}%  {'' if coverage_ok else '(기간 부족 -> 제외)'}")
        for d, v in zip(days, daily):
            rows_out.append((key[0], key[1], d, round(float(v), 1), round(cap, 1)))
        if coverage_ok and cap > 0 and zero_days < 60:
            series[key] = (days, daily / (cap * 24), cap)
    os.makedirs(os.path.join(ROOT, "data"), exist_ok=True)
    with open(os.path.join(ROOT, "data", "all_plants_daily.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["plant", "unit", "Date", "generation_kwh", "capacity_kw_est"])
        w.writerows(rows_out)
    print(f"\n학습에 쓰는 발전소: {len(series)}곳 -> data/all_plants_daily.csv 저장")

    # ---------- 2) 시퀀스 ----------
    def make_xy(key, start=None, end=None, exclude_before_split=False, only_after_split=False):
        days, cf, _ = series[key]
        X, y, d_list = [], [], []
        for i in range(len(cf) - SEQ_LEN):
            d = days[i + SEQ_LEN]
            if only_after_split and d < SPLIT:
                continue
            if exclude_before_split is False and not only_after_split and d >= SPLIT:
                continue
            if start and d < start:
                continue
            if end and d > end:
                continue
            X.append(cf[i : i + SEQ_LEN, None])
            y.append(cf[i + SEQ_LEN])
            d_list.append(d)
        return np.array(X, "float32"), np.array(y, "float32"), d_list

    def build():
        m = keras.Sequential([
            keras.layers.Input(shape=(SEQ_LEN, 1)),
            keras.layers.LSTM(32, return_sequences=True),
            keras.layers.LSTM(32, return_sequences=True),
            keras.layers.LSTM(16),
            keras.layers.Dense(16, activation="relu"),
            keras.layers.Dense(1),
        ])
        m.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
        return m

    def nrmse(model, X, y):
        p = model.predict(X, verbose=0).flatten()
        return float(np.sqrt(np.mean((np.clip(p, 0, None) - y) ** 2)) * 100)

    def persistence(X, y):
        return float(np.sqrt(np.mean((X[:, -1, 0] - y) ** 2)) * 100)

    tests = {k: make_xy(k, only_after_split=True) for k in series}
    results = {}

    # A. single
    Xa, ya, _ = make_xy(TARGET)
    mA = build(); mA.fit(Xa, ya, epochs=EPOCHS, verbose=0, batch_size=32)
    results["A_single(삼천포2)"] = {"삼천포2": nrmse(mA, *tests[TARGET][:2])}
    print(f"\nA single   : train {len(Xa)} seq")

    # B. pooled (all plants)
    Xb = np.concatenate([make_xy(k)[0] for k in series]); yb = np.concatenate([make_xy(k)[1] for k in series])
    mB = build(); mB.fit(Xb, yb, epochs=EPOCHS, verbose=0, batch_size=64)
    results["B_pooled(전체)"] = {f"{k[0]}#{k[1]}": nrmse(mB, *tests[k][:2]) for k in series}
    print(f"B pooled   : train {len(Xb)} seq ({len(series)}곳)")

    # C. pooled + fine-tune on 삼천포2 recent FT_DAYS before split
    days_t = series[TARGET][0]
    ft_start = days_t[max(0, days_t.index(SPLIT) - FT_DAYS - SEQ_LEN)] if SPLIT in days_t else None
    Xc, yc, _ = make_xy(TARGET, start=ft_start)
    mC = keras.models.clone_model(mB); mC.set_weights(mB.get_weights())
    mC.compile(optimizer=keras.optimizers.Adam(FT_LR), loss="mse")
    mC.fit(Xc, yc, epochs=FT_EPOCHS, verbose=0, batch_size=16)
    results["C_pooled+ft(삼천포2)"] = {"삼천포2": nrmse(mC, *tests[TARGET][:2])}
    print(f"C pooled+ft: fine-tune {len(Xc)} seq (최근 {FT_DAYS}일)")

    # D. holdout: pooled without HOLDOUT plant, test on it
    if HOLDOUT in series:
        Xd = np.concatenate([make_xy(k)[0] for k in series if k != HOLDOUT])
        yd = np.concatenate([make_xy(k)[1] for k in series if k != HOLDOUT])
        mD = build(); mD.fit(Xd, yd, epochs=EPOCHS, verbose=0, batch_size=64)
        results["D_holdout(광양항 제외 학습)"] = {
            "광양항(미학습)": nrmse(mD, *tests[HOLDOUT][:2]),
            "광양항_pooled전체모델": results["B_pooled(전체)"][f"{HOLDOUT[0]}#{HOLDOUT[1]}"],
            "광양항_persistence": persistence(*tests[HOLDOUT][:2]),
        }
        print(f"D holdout  : train {len(Xd)} seq (광양항 제외)")

    results["persistence"] = {f"{k[0]}#{k[1]}": persistence(*tests[k][:2]) for k in series}
    results["meta"] = {"split": SPLIT, "seq_len": SEQ_LEN, "epochs": EPOCHS, "plants": [f"{k[0]}#{k[1]}" for k in series],
                       "capacity_kw_est": {f"{k[0]}#{k[1]}": series[k][2] for k in series}}

    # ---------- 결과 ----------
    t = f"{TARGET[0]}#{TARGET[1]}"
    print("\n==== 삼천포 2호기 테스트 nRMSE (2025-09 ~ 2026-08, 용량x24 대비 %) ====")
    print(f"  persistence(전날 그대로)      {results['persistence'][t]:.2f}%")
    print(f"  A. 단일 모델                   {results['A_single(삼천포2)']['삼천포2']:.2f}%")
    print(f"  B. 통합 모델                   {results['B_pooled(전체)'][t]:.2f}%")
    print(f"  C. 통합 + 삼천포 fine-tune     {results['C_pooled+ft(삼천포2)']['삼천포2']:.2f}%")
    print("\n==== 발전소별: 통합 모델 vs persistence ====")
    for k in series:
        n = f"{k[0]}#{k[1]}"
        print(f"  {n:<22} pooled {results['B_pooled(전체)'][n]:5.2f}%   persistence {results['persistence'][n]:5.2f}%")
    if "D_holdout(광양항 제외 학습)" in results:
        d = results["D_holdout(광양항 제외 학습)"]
        print("\n==== 신규 발전소 시나리오 (광양항세방을 학습에서 뺀 통합 모델) ====")
        print(f"  미학습 발전소 예측 {d['광양항(미학습)']:.2f}%   vs 전체 학습 {d['광양항_pooled전체모델']:.2f}%   vs persistence {d['광양항_persistence']:.2f}%")

    os.makedirs(os.path.join(ROOT, "experiments"), exist_ok=True)
    with open(os.path.join(ROOT, "experiments", "pooled_vs_single_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print("\n-> experiments/pooled_vs_single_results.json")


if __name__ == "__main__":
    main()
