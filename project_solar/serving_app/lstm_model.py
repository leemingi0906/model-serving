"""
SolarCast 다음 날 일 발전량 예측용 LSTM 아키텍처 (Day1 baseline 과 Day2 MLflow 학습이 공유).

HAIC 실습과 같은 3층 LSTM(32 -> 32 -> 16) + Dense 구조를 그대로 쓰고, 입력 모양만
(SEQ_LEN=14, N_FEATURES=1) 로 바뀌었습니다. 3.6년치(1,339일) 데이터에서 학습 시퀀스가
약 1,060개 나오므로 파라미터(약 1.6만 개) 대비 샘플은 HAIC 보다 오히려 넉넉합니다.
"""
from tensorflow import keras

from data.features import SEQ_LEN

N_FEATURES = 1  # (generation_kwh)  - 기상 피처를 붙이면 여기와 SolarScaler.transform_point 를 함께 늘립니다


def build_model() -> keras.Model:
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(SEQ_LEN, N_FEATURES)),
            keras.layers.LSTM(32, return_sequences=True),
            keras.layers.LSTM(32, return_sequences=True),
            keras.layers.LSTM(16),
            keras.layers.Dense(16, activation="relu"),
            keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="mse")
    return model
