"""
SolarCast - 태양광 일 발전량 데이터를 LSTM 입력용 시퀀스로 변환하는 공용 유틸리티.

HAIC 실습(project/data/features.py)과 같은 역할입니다. 바뀐 것은 세 가지뿐입니다.
    1) 피처: (종가, 거래량) 2개 -> 일 발전량(generation_kwh) 1개   (N_FEATURES=1)
    2) 윈도우: 20거래일 -> 14일 (태양광은 주말이 없는 달력일 기준, 2주)
    3) 단위: 달러 -> kWh.  정규화는 min-max 그대로 (LSTM은 스케일에 민감)

사전 실험(삼천포 2호기 2023-01~2026-08, 테스트 2025-09~2026-08):
    발전량 단일 피처 LSTM  RMSE 1,331 kWh  (전날 값 그대로 쓰는 persistence 1,650 kWh)
    + 계절 피처(sin/cos doy) RMSE 1,399 kWh  -> 도움이 안 되어 단일 피처로 확정.
    기상(일사량·운량)은 2단계 개선 항목이며, 이 파일의 N_FEATURES 와 SolarScaler 만 늘리면 됩니다.

입력 시퀀스: 최근 SEQ_LEN(14)일의 generation_kwh
타깃: 그다음 날의 generation_kwh
"""
import csv
import pickle

SEQ_LEN = 14  # LSTM 입력 윈도우 길이 (달력일) - 2주

# 설비용량(kW). 발전량 예측제도의 오차율 = |예측-실제| / 설비용량 이므로 모든 임계값의 분모가 됩니다.
# ※ 삼천포 2호기 공식 설비용량을 공개 자료에서 확인하지 못해, 시간 발전량 피크(847 kWh/h)로부터
#    1 MW 로 가정했습니다. 확인되면 이 값 하나만 바꾸면 됩니다.
CAPACITY_KW = 1000.0
# 일 발전량 기준 분모: 설비용량으로 24시간 내내 발전했을 때의 에너지(kWh/일)
DAILY_CAPACITY_KWH = CAPACITY_KW * 24


def load_rows(csv_path: str = "data/sample_samcheonpo2_daily.csv") -> list[dict]:
    """CSV -> [{"Date": "2023-01-01", "Gen": 3320.8}, ...]  (quality 컬럼이 있으면 그대로 보존)"""
    with open(csv_path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = [
            {
                "Date": r["Date"],
                "Gen": float(r["generation_kwh"]),
                "quality": r.get("quality", "ok"),
            }
            for r in reader
        ]
    return rows


class SolarScaler:
    """
    generation_kwh 를 [0, 1] 범위로 정규화하는 min-max 스케일러.

    HAICScaler 와 같은 규칙: Day1(train_baseline_v1.py)에서 한 번 fit 해서
    serving_app/models/scaler.pkl 로 저장하고, Day2 MLflow 학습과 Day3 fine-tuning 이
    같은 파일을 재사용합니다. (fine-tuning 때 다시 fit 하면 기존 가중치와 어긋납니다.)
    """

    def __init__(self):
        self.gen_min = self.gen_max = None

    def fit(self, rows: list[dict]) -> "SolarScaler":
        gens = [r["Gen"] for r in rows]
        self.gen_min, self.gen_max = min(gens), max(gens)
        return self

    def _scale(self, value: float, lo: float, hi: float) -> float:
        if hi == lo:
            return 0.0
        return (value - lo) / (hi - lo)

    def _unscale(self, value: float, lo: float, hi: float) -> float:
        return value * (hi - lo) + lo

    def transform_point(self, gen: float) -> list[float]:
        """하루치 입력 1개를 0~1로.  (N_FEATURES=1 이므로 길이 1 리스트)"""
        return [self._scale(gen, self.gen_min, self.gen_max)]

    def scale_gen(self, gen: float) -> float:
        """타깃(다음 날 발전량)을 학습용으로 정규화."""
        return self._scale(gen, self.gen_min, self.gen_max)

    def inverse_gen(self, scaled: float) -> float:
        """모델 출력(0~1)을 kWh 로 되돌린다."""
        return self._unscale(scaled, self.gen_min, self.gen_max)

    def save(self, path: str = "serving_app/models/scaler.pkl"):
        with open(path, "wb") as f:
            pickle.dump(self.__dict__, f)

    @classmethod
    def load(cls, path: str = "serving_app/models/scaler.pkl") -> "SolarScaler":
        scaler = cls()
        with open(path, "rb") as f:
            scaler.__dict__.update(pickle.load(f))
        return scaler


def build_sequences(rows: list[dict], scaler: SolarScaler, seq_len: int = SEQ_LEN):
    """
    rows(날짜순)에서 (SEQ_LEN, 1) 정규화 입력 시퀀스와 다음 날 발전량(kWh, 정규화 전) 타깃을 만든다.
    반환: X (n_samples, seq_len, 1), y (n_samples,)
    """
    scaled_points = [scaler.transform_point(r["Gen"]) for r in rows]
    gens = [r["Gen"] for r in rows]

    X, y = [], []
    for i in range(len(rows) - seq_len):
        X.append(scaled_points[i : i + seq_len])
        y.append(gens[i + seq_len])
    return X, y


def train_test_split(X: list, y: list, test_ratio: float = 0.2):
    """시간 순서를 유지한 채 앞부분을 train, 뒷부분을 test 로 나눈다 (미래 데이터 누수 방지)."""
    split_idx = int(len(X) * (1 - test_ratio))
    return X[:split_idx], y[:split_idx], X[split_idx:], y[split_idx:]


def nrmse(y_true, y_pred) -> float:
    """
    정규화 RMSE (%): RMSE / DAILY_CAPACITY_KWH * 100.
    발전량 예측제도의 오차율(|예측-실제| / 설비용량)을 일 단위로 옮긴 것으로,
    배포 게이트(train_and_register.py)와 드리프트 판정(drift_detector.py)이 모두 이 지표를 씁니다.
    """
    n = len(y_true)
    if n == 0:
        return 0.0
    mse = sum((float(a) - float(b)) ** 2 for a, b in zip(y_true, y_pred)) / n
    return (mse ** 0.5) / DAILY_CAPACITY_KWH * 100
