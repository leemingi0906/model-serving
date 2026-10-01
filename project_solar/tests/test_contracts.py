"""
계약 테스트 (TensorFlow·서버 없이 돈다):  python -m pytest -q
  - 제도 지표: 시간 오차율 = |예측-실제| / 설비용량, 이용률 10% 이상 시간만, 전부 야간이면 평가 불가
  - 드리프트 분류: 날씨 / 설비 / 모델 드리프트(동시성, PR>1.15) 분기와 재학습 대상 선별
  - 시간 정렬: 발전 "N시" 는 N-1~N 구간, 24:00 은 다음날 T00:00
  - 저장: 운영 중 수신 실적은 뒤에 쓴 값이 이긴다
  - 스키마: time 형식, 최소 길이
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.features import gen_time_to_weather_time, hour_keys  # noqa: E402
from data.metrics import day_error_rate, performance_ratio, summarize  # noqa: E402
from data.storage import append_recent, load_recent  # noqa: E402
from serving_app.monitoring import drift_detector as dd  # noqa: E402
from serving_app.monitoring.retrain_trigger import select_retrain_plants  # noqa: E402
from serving_app.schemas import BatchTestRequest, HourRecord  # noqa: E402
from datetime import date  # noqa: E402

from pydantic import ValidationError  # noqa: E402


# ---------------------------------------------------------------- 지표
def test_day_error_counts_only_hours_above_10pct_capacity():
    cap = 100.0
    actual = [0, 5, 20, 50]          # 0, 5 kWh 는 용량 10% 미만 -> 제외
    pred = [10, 15, 30, 40]          # 남는 두 시간 오차 10, 10 -> 10%
    assert day_error_rate(pred, actual, cap) == pytest.approx(10.0)


def test_day_error_is_none_when_no_daylight():
    assert day_error_rate([10, 10], [0, 0], 100.0) is None


def test_performance_ratio_and_summary():
    assert performance_ratio([50, 50], [100, 100], 100.0) == pytest.approx(0.5)
    assert performance_ratio([10, 10], [20, 20], 100.0) is None  # 기대 발전량이 용량의 절반도 안 되는 날은 PR 계산 안 함
    s = summarize([7.0, 9.0, None, 5.0])
    assert s["n_days"] == 3 and s["mean_error"] == pytest.approx(7.0) and s["pass_rate_8"] == pytest.approx(0.667, abs=1e-3)


# ---------------------------------------------------------------- 시간 정렬
def test_generation_hour_maps_to_weather_hour_end():
    assert gen_time_to_weather_time("2026-08-11 01:00") == "2026-08-11T01:00"
    assert gen_time_to_weather_time("2026-08-11 24:00") == "2026-08-12T00:00"
    assert hour_keys(date(2026, 8, 11))[0] == "2026-08-11 01:00" and hour_keys(date(2026, 8, 11))[-1] == "2026-08-11 24:00"


# ---------------------------------------------------------------- 드리프트 분류
def _fill(pid, errs, prs, start=date(2026, 8, 1)):
    from datetime import timedelta

    dd.records[pid].clear()
    for i, (e, p) in enumerate(zip(errs, prs)):
        dd.records[pid].append({"date": (start + timedelta(days=i)).isoformat(), "day_error": e, "pr": p, "label": "t"})


@pytest.fixture(autouse=True)
def _clean_records(monkeypatch):
    dd.records.clear()
    monkeypatch.setattr(dd, "error_threshold", lambda: 10.0)  # 검증 오차 x1.25 대신 고정 임계값
    yield
    dd.records.clear()


def test_normal_window_is_ok():
    _fill("a", [7.0] * 21, [1.0] * 21)
    assert dd.classify("a")["status"] == "ok"


def test_equipment_when_only_one_plant_and_pr_collapses():
    _fill("a", [7.0] * 7 + [16.0] * 14, [1.0] * 7 + [0.46] * 14)
    _fill("b", [7.0] * 21, [1.0] * 21)   # 같은 시기 다른 발전소는 정상 -> 동시성 없음
    c = dd.classify("a")
    assert c["status"] == "equipment" and c["action"] == "alert"


def test_weather_when_error_high_but_pr_normal():
    _fill("a", [12.0] * 21, [1.0] * 21)
    _fill("b", [7.0] * 21, [1.0] * 21)
    assert dd.classify("a")["status"] == "weather"


def test_model_drift_when_fleet_moves_together():
    for pid in ("a", "b", "c"):
        _fill(pid, [13.0] * 21, [1.2] * 21)
    c = dd.classify("c")
    assert c["status"] == "model_drift" and c["action"] == "retrain" and c["fleet_share"] >= 0.5


def test_model_drift_when_actual_exceeds_expectation_for_one_plant():
    _fill("a", [11.5] * 21, [1.2] * 21)   # 위로 틀림(PR>1.15) 은 한 곳이어도 모델 기준 문제
    _fill("b", [7.0] * 21, [1.0] * 21)
    assert dd.classify("a")["status"] == "model_drift"


def test_retrain_excludes_low_pr_plant_unless_majority():
    for pid in ("a", "b", "c"):
        _fill(pid, [13.0] * 21, [1.2] * 21)
    _fill("faulty", [13.0] * 21, [0.6] * 21)  # 같은 시기, PR 0.6 -> 동시성에 묶여도 재학습 데이터에서 제외
    affected, excluded = select_retrain_plants("c", dd.classify("c"))
    assert "faulty" not in affected and excluded and excluded[0].startswith("faulty")
    for pid in ("a", "b"):                      # 과반이 낮으면 공통 원인 -> 전부 포함
        _fill(pid, [13.0] * 21, [0.7] * 21)
    affected, excluded = select_retrain_plants("c", dd.classify("c"))
    assert set(affected) >= {"a", "b", "c", "faulty"} and not excluded


# ---------------------------------------------------------------- 저장
def test_recent_records_later_write_wins(tmp_path):
    d = str(tmp_path)
    append_recent("p", [{"time": "2026-08-11 12:00", "generation_kwh": 100.0}], recent_dir=d)
    append_recent("p", [{"time": "2026-08-11 12:00", "generation_kwh": 125.0}], recent_dir=d)
    assert load_recent(recent_dir=d)["p"]["2026-08-11 12:00"] == 125.0


# ---------------------------------------------------------------- 스키마
def test_hour_record_time_format():
    HourRecord(time="2026-08-11 24:00", generation_kwh=0)
    with pytest.raises(ValidationError):
        HourRecord(time="2026-08-11T12:00", generation_kwh=0)
    with pytest.raises(ValidationError):
        HourRecord(time="2026-08-11 12:00", generation_kwh=-1)


def test_batch_test_requires_history_plus_one_day():
    recs = [HourRecord(time=k, generation_kwh=0) for k in hour_keys(date(2026, 8, 11))]
    with pytest.raises(ValidationError):
        BatchTestRequest(plant_id="p", records=recs)
