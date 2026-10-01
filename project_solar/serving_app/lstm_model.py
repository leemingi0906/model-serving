"""
SolarCast v2 - 다음 날 24시간 이용률을 한 번에 출력하는 모델 (Day1 baseline 과 Day2 MLflow 학습이 공유).

    hist (72, 1)  --LSTM(32, return_sequences)--LSTM(16)--+
    future (24, 4) --Flatten--Dense(48, relu)------------+--concat--Dense(64, relu)--Dense(24, sigmoid)
    doy (2) ---------------------------------------------+
    plant (10, one-hot) --Dense(8, relu)-----------------+

- 과거 72시간은 실습과 같은 LSTM 인코더가 읽는다 (최근 상태: 오염·고장·계절 수준).
- 내일 24시간 기상 + 태양고도는 "미래 입력"이라 시퀀스 재귀 없이 Dense 로 바로 섞는다.
- 발전소 one-hot 임베딩: 설비용량 추정 오차·방위각 같은 발전소 고유 보정. 미등록 발전소는 0 벡터(전역 평균).
- 손실 = 낮 시간 가중 MAE: 제도가 평가하는 시간(이용률 10% 이상)의 절대오차에 집중한다.
  (실험: mse 8.98% -> 가중 MAE 8.33% -> + 임베딩 7.99%, 8% 통과율 0.51 -> 0.60)
- 출력 24개를 한 번에 내므로(direct multi-output) 1시간씩 이어 붙일 때의 오차 누적이 없다.
"""
import tensorflow as tf
from tensorflow import keras

from data.features import HIST_HOURS, HORIZON, N_FUTURE, N_PLANTS


@keras.saving.register_keras_serializable(package="solarcast")
def daytime_weighted_mae(y_true, y_pred):
    """제도 평가 대상(이용률 >= 10%) 시간은 가중치 1, 그 외 0.15"""
    w = tf.where(y_true >= 0.10, 1.0, 0.15)
    return tf.reduce_sum(w * tf.abs(y_true - y_pred), axis=-1) / tf.reduce_sum(w, axis=-1)


def build_model(width: float = 1.0) -> keras.Model:
    hist_in = keras.layers.Input(shape=(HIST_HOURS, 1), name="hist")
    fut_in = keras.layers.Input(shape=(HORIZON, N_FUTURE), name="future")
    doy_in = keras.layers.Input(shape=(2,), name="doy")
    plant_in = keras.layers.Input(shape=(N_PLANTS,), name="plant")

    h = keras.layers.LSTM(int(32 * width), return_sequences=True)(hist_in)
    h = keras.layers.LSTM(int(16 * width))(h)

    f = keras.layers.Flatten()(fut_in)
    f = keras.layers.Dense(int(48 * width), activation="relu")(f)

    p = keras.layers.Dense(8, activation="relu")(plant_in)

    z = keras.layers.Concatenate()([h, f, doy_in, p])
    z = keras.layers.Dense(int(64 * width), activation="relu")(z)
    out = keras.layers.Dense(HORIZON, activation="sigmoid", name="cf24")(z)

    model = keras.Model(inputs=[hist_in, fut_in, doy_in, plant_in], outputs=out)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss=daytime_weighted_mae)
    return model
