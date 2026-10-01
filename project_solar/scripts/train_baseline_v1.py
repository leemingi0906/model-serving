"""
[Day1 사전 준비] SolarCast 로컬 baseline 모델 만들기 - scripts/train_baseline_v1.py

업로드된 일 발전량 CSV 로 LSTM 을 학습시키고, 서버가 읽어 갈 파일 2개를 만들어 둡니다.
   serving_app/models/scaler.pkl           <- min-max 스케일러 (Day1~3 내내 계속 사용)
   serving_app/models/solarcast_v1.keras   <- 학습된 LSTM 모델

실행 순서
   1) uvicorn serving_app.main:app --host 0.0.0.0 --port 8010
   2) 대시보드(http://localhost:8010)에서 data/sample_samcheonpo2_daily.csv 업로드
   3) python scripts/train_baseline_v1.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.features import load_rows, build_sequences, train_test_split, SolarScaler, nrmse
from data.storage import latest_upload
from serving_app.lstm_model import build_model

MODEL_PATH = "serving_app/models/solarcast_v1.keras"
SCALER_PATH = "serving_app/models/scaler.pkl"
BASE_EPOCHS = 60  # 사전 실험에서 60 epoch 이면 수렴 (1,060 시퀀스 기준 CPU 1분 내외)
NRMSE_GATE = 8.0  # 발전량 예측제도 정산 기준(오차율 8%)과 같은 선


def main():
    import numpy as np
    from tensorflow import keras

    keras.utils.set_random_seed(42)

    # STEP 1. 데이터 읽기 - 가장 최근 업로드한 CSV
    rows = load_rows(latest_upload())

    # STEP 2. 스케일러 fit + 저장 ("여기서 딱 한 번만")
    scaler = SolarScaler().fit(rows)
    scaler.save(SCALER_PATH)
    print(f"scaler fit on {len(rows)}행 (gen {scaler.gen_min:.0f}~{scaler.gen_max:.0f} kWh) -> {SCALER_PATH}")

    # STEP 3~5. 시퀀스 생성, 시간순 분할, 정답 정규화
    X, y = build_sequences(rows, scaler)
    X_train, y_train, X_test, y_test = train_test_split(X, y)
    X_train = np.array(X_train, dtype="float32")
    X_test = np.array(X_test, dtype="float32")
    y_train_scaled = np.array([scaler.scale_gen(v) for v in y_train], dtype="float32")

    # STEP 6~7. 학습
    model = build_model()
    model.fit(X_train, y_train_scaled, epochs=BASE_EPOCHS, verbose=0)

    # STEP 8. 시험 - kWh 로 복원한 뒤 nRMSE(설비용량 대비 %) 계산
    preds = [scaler.inverse_gen(p) for p in model.predict(X_test, verbose=0).flatten()]
    rmse_kwh = (sum((a - b) ** 2 for a, b in zip(y_test, preds)) / len(y_test)) ** 0.5
    score = nrmse(y_test, preds)
    print(f"baseline v1 RMSE = {rmse_kwh:.0f} kWh, nRMSE = {score:.2f}%  (배포 게이트: {NRMSE_GATE:.0f}%)")

    model.save(MODEL_PATH)
    print(f"saved -> {MODEL_PATH}")


if __name__ == "__main__":
    main()
