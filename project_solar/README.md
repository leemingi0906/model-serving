# SolarCast v2 - 태양광 시간별 발전량 예측 B2B 서비스 (조별과제)

HAIC 실습 스켈레톤(`../project/`)의 서빙 → MLOps → AIOps 루프 위에, 발전소 통합 시간별 모델과
드리프트 **원인 분류** 기반 대응을 얹은 버전입니다. 기획안: https://claude.ai/code/artifact/99cef027-f8a6-4d2a-81ec-aefc85a12335
데이터 규칙: `../team_solar/DATA_SCHEMA.md`

## HAIC 실습 → v2 에서 바뀐 것

| 항목 | HAIC (`project/`) | SolarCast v2 | 파일 |
|---|---|---|---|
| 예측 대상 | 다음 날 종가 1개 | 다음 날 **24시간** 발전량 (제도 제출 포맷) | `schemas.py`, `lstm_model.py` |
| 입력 | 최근 20일 (close, volume) | 과거 72h 이용률 + 내일 24h 기상(일사·운량·기온) + 태양고도 + 날짜 | `data/features.py`, `data/solar.py` |
| 단위 | 달러 | 이용률 = kWh ÷ 설비용량 → 발전소 10곳을 한 모델로 | `data/plants.csv` |
| 데이터 | 가상 시세 756행 | 실제 실적 10곳 × 3.7년 시간별(32만 행) + Open-Meteo 기상 | `data/sample_solar_hourly.csv.gz`, `data/weather/` |
| 게이트 | RMSE ≤ $4 | **제도 오차율 8% 통과율**(일 오차율 ≤ 8%인 날의 비율): 첫 배포는 ≥ 0.45(기준선 0.30×1.5), 이후는 현 Production 이상(챔피언/챌린저). fine-tune은 같은 held-out 날짜에서 현 Production보다 평균 오차가 낮을 때만 | `train_and_register.py`, `data/metrics.py` |
| 드리프트 | 오차 크면 재학습 | 21일 오차율 > 임계값(max(8%, 검증 오차×1.25)) 이면 성능비 PR + 발전소 동시성으로 **원인 분류** → ok / weather / equipment / soiling / model_drift | `monitoring/drift_detector.py` |
| 대응 | 재학습 1종 | 날씨·설비·오염 = 알림(재학습 금지), 모델 드리프트만 최근 30일 fine-tuning → 게이트 → 재배포 | `monitoring/retrain_trigger.py` |
| 재학습 데이터 | 업로드 파일 마지막 41행 | 업로드 + 운영 중 수신한 실적(`data/recent/`) → 드리프트를 일으킨 데이터로 재학습 | `data/storage.py` |
| 시나리오 | 랜덤워크 변동성 3배 | 실제 실적 구간: 정상 / 장마철 / 설비 고장(×0.5) / 전 발전소 변화(×1.25) | `scripts/simulate_drift.py`, `/data/window` |

**지표 정의** (`data/metrics.py`): 시간별 오차율 = |예측−실제| ÷ 설비용량 × 100 (발전량이 용량 10% 이상인 시간만),
일 오차율 = 그 평균, PR = 실제 일 발전량 ÷ 모델이 "그날 관측 기상"으로 낸 기대 발전량.

## 실행 순서 (포트 8010)

```bash
cd project_solar
pip install -r requirements.txt
# (저장소에 data/ 가 포함돼 있음. 다시 만들려면 저장소 루트에서 python project_solar/scripts/build_data.py)

# --- Day1 ---
uvicorn serving_app.main:app --host 0.0.0.0 --port 8010        # http://localhost:8010/ 대시보드, /docs
# 대시보드에서 data/sample_solar_hourly.csv.gz 업로드
python scripts/train_baseline_v1.py                              # scaler.pkl + solarcast_v2.keras (CPU 수 분)

# --- Day2 ---
python serving_app/train_and_register.py                         # MLflow 기록 + 제도 오차율 게이트 → Production 승격
MODEL_SOURCE=mlflow uvicorn serving_app.main:app --host 0.0.0.0 --port 8010

# --- Day3 ---
python scripts/simulate_drift.py                                 # 정상 → 장마철 → 설비 고장 → 전 발전소 변화
cat logs/aiops.log ; curl localhost:8010/predict/drift-state
```

`/predict` 요청 예시 (D=2026-09-01, 삼천포 2호기):

```json
{"plant_id": "samcheonpo_2", "date": "2026-09-01",
 "history_kwh": [0, 0, ..., 312.5],            // D-3 01:00 ~ D-1 24:00, 72개
 "forecast": [{"ghi": 0, "cloud_cover": 20, "temperature": 23.1}, ... 24개]}
```

응답: `{"plant_id": "samcheonpo_2", "date": "2026-09-01", "hourly_kwh": [0, 0, ..., 0], "day_total_kwh": 3410.5, "model_version": "production-v1"}`

## 완료 기준

- [ ] `/data/upload` 로 시간별 실적을 올리면 `/data/status` 에 발전소 10곳·기간이 보이는가
- [ ] Day2 게이트: 테스트 1년 8% 통과율로 Production 승격되는가 (기준선 persistence 16.0% / GHI 선형 12.0% 대비 개선 수치)
- [ ] 장마철·설비 고장 배치는 재학습 없이 `[WARN]`/`[ALERT]` 만 남고, 전 발전소 변화 배치만 `[INFO] retrain triggered` → `[OK]` 로 이어지는가
- [ ] 재배포 후 `/predict` 의 `model_version` 이 `production-v2` 로 바뀌는가
