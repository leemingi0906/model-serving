"""
[Day3] 드리프트 감지 - serving_app/monitoring/drift_detector.py (SolarCast)

"최근 모델이 설비용량 대비 몇 % 씩 틀리고 있는지(nRMSE)"를 계산해서,
8% 보다 많이 틀리면 "데이터가 달라졌다(드리프트)"고 판단합니다.

판단 기준
   최근 21일(WINDOW_SIZE)의 nRMSE > 8%  -> 드리프트
   · 8%  = 발전량 예측제도의 정산금 지급 기준 오차율. 이보다 틀리면 정산금이 0 이 되므로
           "서비스가 돈을 못 벌기 시작하는 선"을 그대로 알람 기준으로 씁니다.
   · 21일 = 기상 변동(며칠 흐림)은 걸러내고, 계절 전환·설비 변화는 3주 안에 잡는 길이.
"""
from data.features import nrmse

NRMSE_THRESHOLD = 8.0  # %  (설비용량 대비)
WINDOW_SIZE = 21       # 최근 21일


def compute_nrmse(recent_predictions: list[dict]) -> float:
    """[{"predicted": 3200.0, "actual": 3350.0}, ...] -> nRMSE(%). 빈 목록이면 0.0"""
    if not recent_predictions:
        return 0.0
    return nrmse([p["actual"] for p in recent_predictions], [p["predicted"] for p in recent_predictions])


def is_drift(recent_predictions: list[dict]) -> bool:
    """(데이터 충분한가?) -> 최근 21건만 -> nRMSE 계산 -> 기준(8%)보다 크면 드리프트"""
    if len(recent_predictions) < WINDOW_SIZE:
        return False  # 아직 판단할 만큼 데이터가 쌓이지 않음
    window = recent_predictions[-WINDOW_SIZE:]
    return compute_nrmse(window) > NRMSE_THRESHOLD
