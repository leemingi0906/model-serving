"""
[Day1 사전 준비] SolarCast v2 로컬 baseline 모델 만들기 - scripts/train_baseline_v1.py

서버가 읽어 갈 파일 2개:
   serving_app/models/scaler.pkl           <- 정규화 상수 (Day1~3 공유)
   serving_app/models/solarcast_v2.keras   <- 학습된 모델

실행 순서
   1) python scripts/build_data.py 를 저장소 루트에서 한 번 (data/ 생성)      ※ 저장소에 결과가 이미 포함돼 있음
   2) uvicorn serving_app.main:app --host 0.0.0.0 --port 8010
   3) 대시보드에서 data/sample_solar_hourly.csv.gz 업로드
   4) python scripts/train_baseline_v1.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.features import SolarScaler, build_dataset, split_by_date
from serving_app.lstm_model import build_model
from serving_app.train_and_register import load_training_sources, evaluate, TEST_SPLIT, BASE_EPOCHS, GATE_MIN_PASS_RATE

MODEL_PATH = "serving_app/models/solarcast_v2.keras"
SCALER_PATH = "serving_app/models/scaler.pkl"


def main():
    from tensorflow import keras

    keras.utils.set_random_seed(42)
    SolarScaler().fit().save(SCALER_PATH)
    print(f"scaler(정규화 상수) -> {SCALER_PATH}")

    plants, gen, weather = load_training_sources(with_recent=False)
    Xh, Xf, Xd, Xp, Y, meta = build_dataset(plants, gen, weather)
    tr, te = split_by_date(meta, TEST_SPLIT)
    print(f"samples: train {len(tr)} / test {len(te)}  (plants {len(gen)})")

    model = build_model()
    es = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    model.fit([Xh[tr], Xf[tr], Xd[tr], Xp[tr]], Y[tr], epochs=BASE_EPOCHS, batch_size=64, verbose=0, validation_split=0.1, callbacks=[es])
    m = evaluate(model, Xh[te], Xf[te], Xd[te], Xp[te], Y[te], [meta[i] for i in te], plants)
    print(f"baseline v2: mean_error={m['mean_error']}%  pass_rate_8={m['pass_rate_8']}  pass_rate_6={m['pass_rate_6']}  "
          f"rmse_cf={m['rmse_cf']}  (배포 게이트: pass_rate_8 >= {GATE_MIN_PASS_RATE})")
    model.save(MODEL_PATH)
    print(f"saved -> {MODEL_PATH}")


if __name__ == "__main__":
    main()
