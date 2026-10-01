"""
태양 고도 계산 (위경도 + 시각 -> sin(고도)).

일몰·계절에 따른 해 길이는 날씨가 아니라 천문학적으로 정해지는 값이므로, 모델이 데이터에서 배우게 두지 않고
피처로 직접 넣는다. (기획안 드리프트 분류: "해가 지는 것"은 드리프트가 아니라 설계에서 제거되는 항목)

NOAA 근사식. 정밀도 ±0.5° 수준이면 피처로 충분하다. 입력 시각은 KST(UTC+9) 기준 "시간 구간의 끝"이며,
구간 평균을 대표하도록 끝 시각 - 0.5h 를 쓴다.
"""
import math
from datetime import date, timedelta

KST_OFFSET_H = 9.0


def _declination_deg(doy: int) -> float:
    return 23.44 * math.sin(math.radians(360.0 / 365.0 * (doy - 81)))


def _equation_of_time_min(doy: int) -> float:
    b = math.radians(360.0 / 365.0 * (doy - 81))
    return 9.87 * math.sin(2 * b) - 7.53 * math.cos(b) - 1.5 * math.sin(b)


def sin_elevation(lat: float, lon: float, day: date, hour_end: int) -> float:
    """day 의 (hour_end-1 ~ hour_end) 구간을 대표하는 태양 고도의 sin. 지평선 아래면 0."""
    doy = day.timetuple().tm_yday
    local_h = hour_end - 0.5
    solar_h = local_h + (lon - 15.0 * KST_OFFSET_H) * 4.0 / 60.0 + _equation_of_time_min(doy) / 60.0
    hour_angle = math.radians(15.0 * (solar_h - 12.0))
    decl = math.radians(_declination_deg(doy))
    phi = math.radians(lat)
    s = math.sin(phi) * math.sin(decl) + math.cos(phi) * math.cos(decl) * math.cos(hour_angle)
    return max(0.0, s)


def day_profile(lat: float, lon: float, day: date) -> list[float]:
    """하루 24개 구간(01:00 ~ 24:00 끝 시각)의 sin(고도)."""
    return [sin_elevation(lat, lon, day, h) for h in range(1, 25)]


if __name__ == "__main__":
    for d in (date(2026, 6, 21), date(2026, 12, 21)):
        p = day_profile(34.934, 128.072, d)
        print(d, [round(x, 2) for x in p])
