# SolarCast v2 데이터 스키마 (시간별 통합 모델)

전략 v2(기획안 참조): 발전소 통합 · 시간별 24개 직접 예측 · 제도 오차율로 게이트/드리프트 통일.

## 1. 파일 구성

| 파일 | 상태 | 내용 |
|---|---|---|
| `plants.csv` | 템플릿 작성됨, **조원 확인 필요** | 발전소 메타: plant_id, 용량(추정/공식), 위경도, use 플래그 |
| `data_raw/monthly/*.csv` | 있음 (2023-01~2026-08) | 원본: 발전소별 1~24시 발전량 (kWh) |
| `data/all_plants_hourly.csv.gz` | `scripts/build_hourly_dataset.py` 로 생성 | 시간별 long 포맷 + 이용률 |
| `data_raw/weather/obs_{lat}_{lon}.csv` | **수집 필요** (`scripts/fetch_open_meteo.py`) | 과거 관측(재분석) 시간별 기상 |
| `data_raw/weather/fcst_{lat}_{lon}.csv` | 수집 필요 (같은 스크립트) | 과거 예보 아카이브 |
| `data_raw/weather/d1_{lat}_{lon}.csv` | 수집 필요 (같은 스크립트) | 하루 전 발행 예보 (D-1 리드타임) |

## 2. plants.csv

| 컬럼 | 설명 |
|---|---|
| plant_id | 코드에서 쓰는 영문 키 (예: samcheonpo_2) |
| plant, unit | 원본 CSV 의 발전구분, 호기 (매핑 키, 수정 금지) |
| capacity_kw_est | 기간 중 시간 발전량 최대치로 추정한 용량 (kW) |
| capacity_kw_official | **조원이 채울 것**: 한국남동발전 공개자료의 공식 설비용량. 비어 있으면 est 사용 |
| lat, lon | 발전소 위경도. 현재 값은 사업소 주소 기준 근사치, **확인 필요** |
| use | 1 = 학습·평가에 사용, 0 = 제외 (사유는 note) |

## 3. 시간 정렬 규칙

- 발전 데이터 "N시 발전량" = (N-1):00 ~ N:00 구간의 kWh. 데이터셋 time 은 구간 **끝** 시각 `N:00`.
- Open-Meteo 복사량(shortwave 등)은 "직전 1시간 평균"이라 `N:00` 행이 같은 구간을 가리킴. 그대로 조인.
- 기온·운량 등 순간값은 `N:00` 시점 값. 구간 평균이 필요하면 (N-1, N) 두 값 평균.
- 모두 KST(Asia/Seoul). 서머타임 없음.

## 4. 모델 입력/출력 (v2)

```
입력 (발전소 1곳, 예측일 D 에 대해)
  history : D-3 01:00 ~ D-1 24:00 이용률 72개
  forecast: D 01:00 ~ 24:00 기상 24 x k  (학습 = obs, 서빙 = d1/실시간 예보)
  solar   : D 01:00 ~ 24:00 태양고도 sin 24개 (위경도·시각으로 계산, 일몰·계절 흡수)
  calendar: day-of-year sin/cos
출력
  D 01:00 ~ 24:00 이용률 24개  -> x 용량 = hourly_kwh[24]
```

## 5. 평가 지표 (제도 기준)

```
hour_error_rate = |pred_kwh - actual_kwh| / capacity_kw * 100      (발전량 >= 용량 10% 인 시간만)
day_error_rate  = mean(hour_error_rate over qualifying hours)
pass_rate_8     = day_error_rate <= 8% 인 날의 비율   (6% 도 함께 기록)
```
배포 게이트: 테스트 기간 pass_rate_8 가 현재 Production 이상. 드리프트: 최근 21일 day_error_rate 평균 > 8%.

## 6. 성능비 (드리프트 원인 분류용)

```
expected_kwh = 모델이 "그날 실제 기상(obs)"으로 낸 예측      (날씨를 다 알았을 때의 기대치)
PR           = sum(actual_kwh) / sum(expected_kwh)   (일 단위, 발전소별)
```
PR 이 한 발전소만 급락 → 설비 이상, 완만 하락 → 오염, 전 발전소 동반 상승·하락 → 모델/계절 드리프트.
