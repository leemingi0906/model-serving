# 해아림 v2 - 태양광 시간별 발전량 예측 B2B 서비스 (조별과제)

HAIC 실습 스켈레톤(`../project/`)의 서빙 → MLOps → AIOps 루프 위에, 발전소 통합 시간별 모델과
드리프트 **원인 분류** 기반 대응을 얹은 버전입니다. 기획안: https://claude.ai/code/artifact/99cef027-f8a6-4d2a-81ec-aefc85a12335
데이터 규칙: `../team_solar/DATA_SCHEMA.md`

## HAIC 실습 → v2 에서 바뀐 것

| 항목 | HAIC (`project/`) | 해아림 v2 | 파일 |
|---|---|---|---|
| 예측 대상 | 다음 날 종가 1개 | 다음 날 **24시간** 발전량 (제도 제출 포맷) | `schemas.py`, `lstm_model.py` |
| 입력 | 최근 20일 (close, volume) | 과거 72h 이용률 + 내일 24h 기상(일사·운량·기온) + 태양고도 + 날짜 | `data/features.py`, `data/solar.py` |
| 단위 | 달러 | 이용률 = kWh ÷ 설비용량 → 발전소 10곳을 한 모델로 | `data/plants.csv` |
| 데이터 | 가상 시세 756행 | 실제 실적 10곳 × 3.7년 시간별(32만 행) + Open-Meteo 기상 | `data/sample_solar_hourly.csv.gz`, `data/weather/` |
| 모델 레지스트리 | stage=Production | alias `@champion` (stage 는 폐기 예정). 동봉 모델은 `--register-local` 로 학습 없이 같은 게이트를 거쳐 등록 | `train_and_register.py`, `model_loader.py` |
| 감시 보조 지표 | 없음 | 같은 날을 **하루 전 예보**로 예측한 오차(`d1_error`, 서빙 조건)와 **예보 괴리**(낮 시간 관측-예보 일사량 차이)를 함께 기록 | `routers/predict.py` |
| 업로드 검증 | 없음 | 컬럼·time 형식·중복·미등록 발전소·범위(용량 105%) 검사 후 거부, 계약 테스트 13개 | `routers/data.py`, `tests/` |
| 게이트 | RMSE ≤ $4 | **제도 오차율 8% 통과율**(일 오차율 ≤ 8%인 날의 비율): 첫 배포는 ≥ 0.45(기준선 0.30×1.5), 이후는 현 Production 이상(챔피언/챌린저). fine-tune은 같은 held-out 날짜에서 현 Production보다 평균 오차가 낮을 때만 | `train_and_register.py`, `data/metrics.py` |
| 드리프트 | 오차 크면 재학습 | 21일 오차율 > 임계값(max(8%, 검증 오차×1.25)) 이면 성능비 PR + 발전소 동시성으로 **원인 분류** → ok / weather / equipment / soiling / model_drift | `monitoring/drift_detector.py` |
| 대응 | 재학습 1종, 요청 안에서 동기 실행 | 날씨·설비·오염 = 알림(재학습 금지), 모델 드리프트만 최근 30일 fine-tuning → 게이트 → 재배포. 재학습은 **백그라운드 작업**(`/jobs/current`, 중복 요청 합침)으로 돌아 학습 중에도 `/predict` 는 기존 버전으로 응답. 같은 시기에 PR < 0.75 인 발전소는 동시성에 묶여도 fine-tune 데이터에서 제외(고장을 정상으로 학습 방지) | `monitoring/retrain_trigger.py` |
| 재학습 데이터 | 업로드 파일 마지막 41행 | 업로드 + 운영 중 수신한 실적(`data/recent/`) → 드리프트를 일으킨 데이터로 재학습 | `data/storage.py` |
| 시나리오 | 랜덤워크 변동성 3배 | 실제 실적 구간: 정상 / 장마철 / 설비 고장(×0.5) / 전 발전소 변화(×1.25) + 실제 사례 3건 + 기후 변화 모의 2건(우기 패턴만 / 폭염화 관계 변화) | `scripts/simulate_drift.py`, `/data/window` |

**지표 정의** (`data/metrics.py`): 시간별 오차율 = |예측−실제| ÷ 설비용량 × 100 (발전량이 용량 10% 이상인 시간만),
일 오차율 = 그 평균, PR = 실제 일 발전량 ÷ 모델이 "그날 관측 기상"으로 낸 기대 발전량.

## 결과 (2026-10-01 클라우드 실행, 테스트 2025-09~2026-08 · 10곳 · 3,534 발전소-일)

| 모델 | 일 오차율 평균 | 8% 통과율 | 6% 통과율 |
|---|---|---|---|
| 전날 같은 시간 그대로 (persistence) | 16.0% | 0.30 | 0.21 |
| 일사량 선형 비례 (발전소별 계수) | 12.0% | 0.30 | 0.16 |
| LSTM + 기상 (mse) | 8.98% | 0.51 | 0.32 |
| + 낮 시간 가중 MAE | 8.33% | 0.57 | 0.38 |
| **+ 발전소 임베딩 (채택)** | **7.99% (실험) / 8.29% (Production v1)** | **0.60 / 0.58** | 0.42 / 0.39 |
| 같은 모델, 입력을 하루 전 예보로 (서빙 조건) | 9.85% | 0.49 | 0.31 |

v1 일 발전량 모델(기상 없음)은 일 단위라 직접 비교가 안 되지만, 일 총량 기준 persistence 대비 20% 개선에 그쳤던 것에 비해
시간별 기상 모델은 persistence 대비 오차 절반입니다. 드리프트 임계값 = 8.29% × 1.25 = 10.36%.

| 시나리오 (삼천포2 등, 21일) | 일 오차율 | PR | 판정 → 대응 |
|---|---|---|---|
| 정상 (2026-08-11~31) | 7.17% | 0.98 | ok |
| 장마철 (2026-06-20~07-10) | 9.19% | 1.05 | ok (임계값 이내, PR 정상) |
| 설비 고장 (8일째부터 ×0.5) | 16.29% | 0.66 → 최근 7일 0.46 | equipment → `[ALERT]` 재학습 차단 |
| 전 발전소 변화 (경남 3곳 ×1.25) | 12.2 / 13.6 / 14.2% (v1 기준) · 9개 순서로 돌리면 v2 기준 18.3~19.5%, PR 1.43~1.49 | 1.19~1.22 | model_drift → fine-tune(3곳, 30일) new_error 10.41% ≤ 현 12.0% → **v2 승격** (9개 순서에서는 11.03% → v3) |

| 실제 사례 (주입 없음, Production v1 로 평가) | 일 오차율 | PR 최근 7일 | 판정 → 대응 |
|---|---|---|---|
| 광양항 2025-10-05~25 (PR 0.3 지속, 10/6~11 정지) | 23.5% | 0.44 | equipment → `[ALERT]` 재학습 차단 |
| 삼천포2 2025-12-05~25 (12/16 급락, 1/9 복구) | 15.8% | 0.47 | equipment → `[ALERT]` |
| 영흥#5 2026-08-11~31 (3월부터 점진 하락 0.96→0.64) | 11.7% | 0.68 | equipment → `[ALERT]` (60일+ 기록이면 soiling) |

| 기후 변화 모의 (경남·경북 6곳, 21일) | 일 오차율 | PR | 판정 → 대응 |
|---|---|---|---|
| A. 우기 패턴만 (장마철 2026-06-20~, 변형 없음) | 6.9~9.2% | 0.98~1.11 | 6곳 모두 ok. 모델이 아는 날씨(입력 분포 변화)라 재학습 없음 |
| B. 폭염화 관계 변화 (2026-07-11~, 관측기온 28°C 초과 1°C당 −5%) | 경남 4곳 10.9~15.0%, 경북 2곳 8.4~8.6% | 0.70~0.75 / 0.87~0.90 | 경남 4곳 model_drift (동시성 0.67) → fine-tune(4곳, 30일) new_error 8.39% → **v2 승격** |

같은 20% 손실이라도 장마철 창에 넣으면 오차율이 8.5~9.2%에 그쳐 임계값(10.36%)을 넘지 않습니다. 제도 오차율은 설비용량 대비 절대 오차라서 흐린 날의 비례 손실은 작게 잡히기 때문입니다.
기후 변화 시나리오를 폭염(맑은 날) 창에 둔 이유입니다. `/data/window?heat_loss=0.05&heat_base=28` 이 관측 기온으로 시간별 손실을 만듭니다(업로드 원본 기준이라 반복 실행해도 중첩되지 않음).

지난 1년 실제 데이터에서 발전소 10곳이 동시에 틀린 달은 없었습니다(월별 PR 중앙값 0.93~1.03). 즉 재학습이 필요한 모델 드리프트는 없었고
이상은 전부 설비 문제였으며, 감지기는 세 건 모두 재학습을 막았습니다. 분석: `../team_solar/experiments/drift_analysis.py` (결과 JSON·로그 동봉).

발표용 드리프트 설명(분류·사례·판정 트리·예상 질문): `../team_solar/DRIFT_GUIDE.md`, 그림 `../team_solar/snapshots/41_v2_drift_cases_pr.png`.

재배포 후 `/predict`: production-v1 → production-v2 (9개 순서에서는 v3). 새 클론 전체 재현 로그: `../team_solar/snapshots/43_v2_run_all_9scenarios.log`. 로그·캡처: `../team_solar/snapshots/3*_v2_*`, `2*_v2_*.png`.

## 실행 순서 (포트 8010)

팀원 빠른 시작 (동봉 모델 등록, 학습 없음, 3~5분) - 자세한 안내는 `TEAM_GUIDE.md`:

```bash
cd project_solar
pip install -r requirements.txt
bash scripts/quick_start.sh          # 업로드 -> 동봉 모델 평가·게이트·MLflow 등록 -> 서버 기동
python scripts/simulate_drift.py     # 시나리오 9개 (재학습 2회는 백그라운드 작업, 자동 대기)
python -m pytest -q                  # 계약 테스트 13개 (TensorFlow 없이 0.5초)
```

처음부터 전부 학습하려면 (CPU 20~30분):

```bash
bash scripts/run_all.sh
```

단계별로 돌리려면:

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
python scripts/simulate_drift.py                                 # 정상 → 장마철 → 설비 고장 → 실제 사례 3건 → 기후 A(우기) → 기후 B(폭염화, 재학습 v2) → 전 발전소 변화(재학습 v3)
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

- [x] `/data/upload` 로 시간별 실적을 올리면 `/data/status` 에 발전소 10곳·기간이 보이는가
- [x] Day2 게이트: 테스트 1년 8% 통과율 0.579 ≥ 0.45 로 Production v1 승격 (persistence 16.0% / GHI 선형 12.0% → 8.29%)
- [x] 장마철·우기 패턴은 ok, 설비 고장·실제 사례 3건은 `[ALERT]` 만 남고 재학습 차단, 폭염화 관계 변화와 전 발전소 변화만 `[INFO] retrain triggered` → `[OK]`
- [x] 재배포 후 `/predict` 의 `model_version` 이 `production-v2` 로 바뀜
