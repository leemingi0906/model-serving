"""
해아림 v2 - 시간별 통합 모델의 피처 정의 (학습·서빙·재학습·시뮬레이션이 모두 이 모듈을 쓴다).

한 샘플 = (발전소 p, 예측일 D)
    hist   : D-3 01:00 ~ D-1 24:00 의 이용률 72개                     (HIST_HOURS, 1)
    future : D 01:00 ~ 24:00 의 기상 3개(정규화) + sin(태양고도)        (HORIZON, N_FUTURE)
             학습·사후평가 = 관측(obs), 서빙 = 예보(d1 또는 실시간 예보)
    doy    : day-of-year sin/cos                                       (2,)
    target : D 01:00 ~ 24:00 의 이용률 24개                            (HORIZON,)
이용률(CF) = 발전량(kWh) / 설비용량(kW) : 발전소 크기와 무관한 0~1 값이라 발전소를 한 모델에 통합할 수 있다.

시각 규칙 (DATA_SCHEMA.md): 발전 "D HH:00" 은 (HH-1)~HH 구간, 기상 "DTHH:00" 도 직전 1시간 평균이라 같은 키로 조인.
발전 24:00 은 기상 D+1 T00:00.
"""
import csv
import gzip
import math
import pickle
from datetime import date, datetime, timedelta

from data.solar import day_profile

HIST_HOURS = 72
HORIZON = 24
WEATHER_VARS = ["shortwave_radiation", "cloud_cover", "temperature_2m"]
WEATHER_SCALE = {"shortwave_radiation": 1000.0, "cloud_cover": 100.0, "temperature_2m": 40.0}
N_FUTURE = len(WEATHER_VARS) + 1  # + sin(태양고도)
# 발전소 임베딩용 고정 순서 (학습에 쓴 hourly_ok 발전소). 등록되지 않은 발전소는 0 벡터 = "전역 평균 발전소"로 예측된다.
PLANT_IDS = ["doosan_1", "gumi_1", "gwangyang_1", "gyeongsang_1", "samcheonpo_2", "samcheonpo_3",
             "yecheon_1", "yeongheung5_1", "yeongheung_1", "yeongheung_2"]
N_PLANTS = len(PLANT_IDS)
MIN_GEN_FRACTION = 0.10  # 제도: 발전량이 설비용량의 10% 이상인 시간만 오차 평가

PLANTS_PATH = "data/plants.csv"
WEATHER_OBS_PATH = "data/weather/obs.csv.gz"
WEATHER_D1_PATH = "data/weather/d1.csv.gz"


# ---------------------------------------------------------------- 로딩
def _open(path):
    return gzip.open(path, "rt", encoding="utf-8") if path.endswith(".gz") else open(path, encoding="utf-8-sig")


def load_plants(path: str = PLANTS_PATH) -> dict[str, dict]:
    """plant_id -> {capacity_kw, lat, lon, loc, hourly_ok, name}"""
    out = {}
    with _open(path) as f:
        for r in csv.DictReader(f):
            if not r["lat"] or not r["lon"]:
                continue
            lat, lon = float(r["lat"]), float(r["lon"])
            out[r["plant_id"]] = {
                "plant_id": r["plant_id"],
                "name": f'{r["plant"]} {r["unit"]}호기',
                "capacity_kw": float(r["capacity_kw_official"] or r["capacity_kw_est"]),
                "lat": lat, "lon": lon,
                "loc": f"{round(lat, 3)}_{round(lon, 3)}",
                "use": r["use"] == "1",
                "hourly_ok": r.get("hourly_ok", "1") == "1",
            }
    return out


def load_generation(path: str) -> dict[str, dict[str, float]]:
    """CSV(plant_id,time,generation_kwh) -> plant_id -> {'YYYY-MM-DD HH:00': kWh}"""
    out: dict[str, dict[str, float]] = {}
    with _open(path) as f:
        for r in csv.DictReader(f):
            out.setdefault(r["plant_id"], {})[r["time"]] = float(r["generation_kwh"])
    return out


def load_weather(path: str) -> dict[str, dict[str, list[float]]]:
    """long CSV(loc,time,vars...) -> loc -> {'YYYY-MM-DDTHH:MM': [정규화된 var...]}"""
    out: dict[str, dict[str, list[float]]] = {}
    with _open(path) as f:
        for r in csv.DictReader(f):
            vals = []
            for v in WEATHER_VARS:
                x = r[v]
                vals.append(float(x) / WEATHER_SCALE[v] if x not in ("", "None") else float("nan"))
            out.setdefault(r["loc"], {})[r["time"]] = vals
    return out


def gen_time_to_weather_time(t: str) -> str:
    day, hh = t.split(" ")
    h = int(hh[:2])
    if h == 24:
        nd = date.fromisoformat(day) + timedelta(days=1)
        return f"{nd.isoformat()}T00:00"
    return f"{day}T{h:02d}:00"


def hour_keys(day: date) -> list[str]:
    return [f"{day.isoformat()} {h:02d}:00" for h in range(1, 25)]


def normalize_weather_row(ghi: float, cloud: float, temp: float) -> list[float]:
    return [ghi / WEATHER_SCALE["shortwave_radiation"], cloud / WEATHER_SCALE["cloud_cover"], temp / WEATHER_SCALE["temperature_2m"]]


def plant_vector(plant_id: str) -> list[float]:
    """발전소 one-hot (PLANT_IDS 순서). 미등록 발전소 -> 전부 0."""
    return [1.0 if plant_id == pid else 0.0 for pid in PLANT_IDS]


def doy_features(day: date) -> list[float]:
    a = 2 * math.pi * day.timetuple().tm_yday / 365.25
    return [math.sin(a), math.cos(a)]


# ---------------------------------------------------------------- 스케일러 (HAIC 구조 호환)
class SolarScaler:
    """
    v2 의 입력은 이미 0~1 근처(이용률, 정규화 기상)라 min-max fit 이 필요 없지만, HAIC 실습 구조(Day1 에서 한 번 fit 한
    scaler.pkl 을 Day1~3 이 공유)를 유지하기 위해 정규화 상수를 이 객체에 담아 저장한다. 서빙·재학습이 같은 상수를 쓰는지가
    "학습 시점 입력 = 서빙 시점 입력"의 보증이다.
    """

    def __init__(self):
        self.weather_scale = dict(WEATHER_SCALE)
        self.hist_hours = HIST_HOURS
        self.horizon = HORIZON

    def fit(self, _rows=None) -> "SolarScaler":
        return self

    def save(self, path: str = "serving_app/models/scaler.pkl"):
        with open(path, "wb") as f:
            pickle.dump(self.__dict__, f)

    @classmethod
    def load(cls, path: str = "serving_app/models/scaler.pkl") -> "SolarScaler":
        s = cls()
        with open(path, "rb") as f:
            s.__dict__.update(pickle.load(f))
        return s


# ---------------------------------------------------------------- 샘플 생성
def build_future(plant: dict, day: date, weather_loc: dict[str, list[float]]) -> list[list[float]] | None:
    """D 의 24시간 (기상 3 + sin고도) 행렬. 기상이 하나라도 비면 None."""
    elev = day_profile(plant["lat"], plant["lon"], day)
    rows = []
    for i, k in enumerate(hour_keys(day)):
        w = weather_loc.get(gen_time_to_weather_time(k))
        if w is None or any(math.isnan(x) for x in w):
            return None
        rows.append(w + [elev[i]])
    return rows


def build_day_sample(plant: dict, day: date, gen: dict[str, float], weather_loc: dict[str, list[float]]):
    """(hist[72][1], future[24][N_FUTURE], doy[2], target[24]) 또는 데이터가 비면 None."""
    cap = plant["capacity_kw"]
    hist = []
    for d_off in (3, 2, 1):
        for k in hour_keys(day - timedelta(days=d_off)):
            v = gen.get(k)
            if v is None:
                return None
            hist.append([min(max(v / cap, 0.0), 1.0)])
    target = []
    for k in hour_keys(day):
        v = gen.get(k)
        if v is None:
            return None
        target.append(min(max(v / cap, 0.0), 1.0))
    future = build_future(plant, day, weather_loc)
    if future is None:
        return None
    return hist, future, doy_features(day), target


def build_dataset(plants: dict, gen_by_plant: dict, weather: dict, plant_ids=None, start: str | None = None,
                  end: str | None = None):
    """
    반환: X_hist (n,72,1), X_fut (n,24,N_FUTURE), X_doy (n,2), X_plant (n,N_PLANTS), y (n,24), meta [(plant_id, 'YYYY-MM-DD'), ...]
    start/end 는 예측일 D 의 범위(포함).
    """
    import numpy as np

    Xh, Xf, Xd, Xp, Y, meta = [], [], [], [], [], []
    for pid, gen in gen_by_plant.items():
        if plant_ids is not None and pid not in plant_ids:
            continue
        plant = plants.get(pid)
        if plant is None:
            continue
        wloc = weather.get(plant["loc"], {})
        days = sorted({t[:10] for t in gen})
        for ds in days:
            if (start and ds < start) or (end and ds > end):
                continue
            s = build_day_sample(plant, date.fromisoformat(ds), gen, wloc)
            if s is None:
                continue
            h, f, d, t = s
            Xh.append(h); Xf.append(f); Xd.append(d); Xp.append(plant_vector(pid)); Y.append(t); meta.append((pid, ds))
    return (np.array(Xh, "float32"), np.array(Xf, "float32"), np.array(Xd, "float32"), np.array(Xp, "float32"),
            np.array(Y, "float32"), meta)


def split_by_date(meta: list[tuple[str, str]], split: str):
    """meta 의 예측일 기준으로 train(< split) / test(>= split) 인덱스."""
    tr = [i for i, (_, d) in enumerate(meta) if d < split]
    te = [i for i, (_, d) in enumerate(meta) if d >= split]
    return tr, te
