"""
SolarCast v2 - 다음 날 24시간 이용률을 한 번에 출력하는 모델 (Day1 baseline 과 Day2 MLflow 학습이 공유).

    hist (72, 1)  --LSTM(32, return_sequences)--LSTM(16)--+
    future (24, 4) --Flatten--Dense(48, relu)------------+--concat--Dense(64, relu)--Dense(24, sigmoid)
    doy (2) ---------------------------------------------+

- 과거 72시간은 실습과 같은 LSTM 인코더가 읽는다 (최근 상태: 오염·고장·계절 수준).
- 내일 24시간 기상 + 태양고도는 "미래 입력"이라 시퀀스 재귀 없이 Dense 로 바로 섞는다.
- 출력 24개를 한 번에 내므로(direct multi-output) 1시간씩 이어 붙일 때의 오차 누적이 없다.
- sigmoid 출력 = 이용률 0~1. 발전량(kWh) = 출력 x 설비용량.
파라미터 약 1.5만 개, 10곳 x 3.6년 = 약 1.3만 샘플, CPU 40 epoch 수 분.
"""
from tensorflow import keras

from data.features import HIST_HOURS, HORIZON, N_FUTURE


def build_model() -> keras.Model:
    hist_in = keras.layers.Input(shape=(HIST_HOURS, 1), name="hist")
    fut_in = keras.layers.Input(shape=(HORIZON, N_FUTURE), name="future")
    doy_in = keras.layers.Input(shape=(2,), name="doy")

    h = keras.layers.LSTM(32, return_sequences=True)(hist_in)
    h = keras.layers.LSTM(16)(h)

    f = keras.layers.Flatten()(fut_in)
    f = keras.layers.Dense(48, activation="relu")(f)

    z = keras.layers.Concatenate()([h, f, doy_in])
    z = keras.layers.Dense(64, activation="relu")(z)
    out = keras.layers.Dense(HORIZON, activation="sigmoid", name="cf24")(z)

    model = keras.Model(inputs=[hist_in, fut_in, doy_in], outputs=out)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="mse")
    return model
