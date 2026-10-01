"""
SolarCast v2: FastAPI 요청/응답 스키마.

/predict 는 제도 제출 포맷(다음 날 1시간 단위 24개)을 그대로 돌려준다.
입력 검증이 학습 시점 피처(data/features.py)와 어긋나지 않도록 길이(72, 24)와 값 범위를 스키마 단에서 강제한다.
"""
from pydantic import BaseModel, Field

from data.features import HIST_HOURS, HORIZON


class HourWeather(BaseModel):
    ghi: float = Field(..., ge=0, le=1400, description="수평면 전일사 W/m2 (해당 시간 평균)")
    cloud_cover: float = Field(..., ge=0, le=100, description="전운량 %")
    temperature: float = Field(..., ge=-40, le=50, description="기온 C")


class PredictRequest(BaseModel):
    plant_id: str = Field(..., description="data/plants.csv 의 plant_id. 미등록 발전소면 capacity_kw·lat·lon 필수")
    date: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$", description="예측 대상일 (D)")
    history_kwh: list[float] = Field(..., min_length=HIST_HOURS, max_length=HIST_HOURS,
                                     description="D-3 01:00 ~ D-1 24:00 시간별 발전량 kWh (오래된 순)")
    forecast: list[HourWeather] = Field(..., min_length=HORIZON, max_length=HORIZON,
                                        description="D 01:00 ~ 24:00 기상 예보")
    capacity_kw: float | None = Field(None, gt=0)
    lat: float | None = Field(None, ge=-90, le=90)
    lon: float | None = Field(None, ge=-180, le=180)


class PredictResponse(BaseModel):
    plant_id: str
    date: str
    hourly_kwh: list[float]
    day_total_kwh: float
    model_version: str


class HourRecord(BaseModel):
    time: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2} \d{2}:00$", description="'YYYY-MM-DD HH:00', HH=01..24")
    generation_kwh: float = Field(..., ge=0)


class BatchTestRequest(BaseModel):
    # Day3 드리프트 시뮬레이션 / 실적 사후 평가: 연속된 시간별 실적을 보내면 서버가 날짜별로 잘라
    # (72시간 이력 -> 그날 24시간) 예측하고, 서버가 가진 그날의 관측 기상으로 기대치를 만들어 오차율·PR 을 계산한다.
    plant_id: str
    label: str = Field("batch", description="normal / monsoon / equipment / fleet_shift 등 로그 식별용")
    records: list[HourRecord] = Field(..., min_length=HIST_HOURS + HORIZON)
    persist: bool = Field(True, description="True 면 실적으로 저장 -> 재학습 데이터에 반영")
    reset_state: bool = Field(False, description="True 면 이 발전소의 드리프트 기록을 비우고 시작 (시나리오를 독립적으로 재현할 때)")
    check: bool = Field(True, description="False 면 기록만 쌓고 판정·대응은 하지 않음 (여러 발전소 배치를 모은 뒤 마지막에 한 번 판정)")


class DayResult(BaseModel):
    date: str
    actual_kwh: float
    expected_kwh: float
    day_error: float | None
    pr: float | None


class BatchTestResponse(BaseModel):
    plant_id: str
    label: str
    days: list[DayResult]
    summary: dict
    drift_check: dict
