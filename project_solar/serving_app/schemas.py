"""
SolarCast: FastAPI 요청/응답 Pydantic 스키마.

/predict 는 최근 SEQ_LEN(14)일의 일 발전량 시퀀스를 받아 다음 날 일 발전량(kWh)을 돌려줍니다.
학습 시점 피처(data/features.py)와 서빙 시점 입력이 어긋나지 않도록 길이(SEQ_LEN)와
값 범위(ge=0: 발전량은 음수가 될 수 없고, 야간·고장일은 0 가능)를 스키마 단에서 강제합니다.
"""
from pydantic import BaseModel, Field

from data.features import SEQ_LEN, DAILY_CAPACITY_KWH


class DailyPoint(BaseModel):
    # 설비용량으로 24시간 발전한 양을 넘는 값은 계측 오류이므로 거부 (le=DAILY_CAPACITY_KWH)
    generation_kwh: float = Field(..., ge=0, le=DAILY_CAPACITY_KWH, description="해당 일 발전량(kWh)")


class PredictRequest(BaseModel):
    sequence: list[DailyPoint] = Field(
        ...,
        min_length=SEQ_LEN,
        max_length=SEQ_LEN,
        description=f"가장 오래된 날 -> 가장 최근 날 순서의 최근 {SEQ_LEN}일 일 발전량",
    )


class PredictResponse(BaseModel):
    predicted_kwh: float
    model_version: str


class BatchTestRequest(BaseModel):
    # Day3 드리프트 시뮬레이션에서 사용 (scripts/simulate_drift.py 참고)
    # SEQ_LEN + N 개의 연속된 일 발전량을 보내면, 서버가 슬라이딩 윈도우로 잘라 N 건을 연속 예측한다.
    values: list[float] = Field(..., min_length=SEQ_LEN + 1)
    label: str = Field("batch", description="로그 식별용 배치 이름 (normal / monsoon / new_plant 등)")


class BatchTestResponse(BaseModel):
    predictions: list[float]
    drift_check: dict
