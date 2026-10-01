"""
발전량 예측제도 기준 지표 - 게이트(train_and_register.py)와 드리프트 판정(drift_detector.py)이 같은 함수를 쓴다.

    시간별 오차율(%) = |예측 - 실제| / 설비용량 x 100        (발전량이 설비용량의 10% 이상인 시간만)
    일 오차율(%)     = 그날 평가 대상 시간의 오차율 평균       (평가 대상 시간이 없으면 None: 야간·정지일)
    통과율           = 일 오차율 <= 8%(또는 6%) 인 날의 비율
    성능비 PR        = 실제 일 발전량 / 기상으로 기대되는 일 발전량   (모델이 "그날 실제 기상"으로 낸 예측 = 기대치)
"""
from data.features import MIN_GEN_FRACTION

ERROR_THRESHOLD = 8.0  # % - 정산금 지급 상한 오차율
ERROR_THRESHOLD_GOOD = 6.0  # % - 상향 정산 구간


def day_error_rate(pred_kwh: list[float], actual_kwh: list[float], capacity_kw: float) -> float | None:
    errs = []
    for p, a in zip(pred_kwh, actual_kwh):
        if a >= MIN_GEN_FRACTION * capacity_kw:
            errs.append(abs(p - a) / capacity_kw * 100.0)
    if not errs:
        return None
    return sum(errs) / len(errs)


def performance_ratio(actual_kwh: list[float], expected_kwh: list[float], capacity_kw: float) -> float | None:
    e = sum(expected_kwh)
    if e < 0.5 * capacity_kw:  # 기대 발전량이 너무 작은 날(폭설·완전 흐림)은 PR 을 계산하지 않음
        return None
    return sum(actual_kwh) / e


def summarize(day_errors: list[float | None]) -> dict:
    vals = [e for e in day_errors if e is not None]
    if not vals:
        return {"n_days": 0, "mean_error": None, "pass_rate_8": None, "pass_rate_6": None}
    return {
        "n_days": len(vals),
        "mean_error": round(sum(vals) / len(vals), 2),
        "pass_rate_8": round(sum(1 for v in vals if v <= ERROR_THRESHOLD) / len(vals), 3),
        "pass_rate_6": round(sum(1 for v in vals if v <= ERROR_THRESHOLD_GOOD) / len(vals), 3),
    }
