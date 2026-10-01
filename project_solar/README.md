# SolarCast - 태양광 일 발전량 예측 B2B 서비스 (조별과제)

HAIC 실습 스켈레톤(`../project/`)의 서빙 → MLOps → AIOps 루프를 그대로 쓰고,
데이터·지표·드리프트 시나리오만 **삼천포 태양광 2호기 실제 일 발전량**으로 바꾼 복제본입니다.
기획안: https://claude.ai/code/artifact/99cef027-f8a6-4d2a-81ec-aefc85a12335

## HAIC 실습과 바뀐 점

| 항목 | HAIC (`project/`) | SolarCast (`project_solar/`) | 바뀐 파일 |
|---|---|---|---|
| 예측 대상 | 다음 날 종가 ($) | 다음 날 일 발전량 (kWh) | `schemas.py` |
| 입력 | 최근 20거래일 (close, volume) | 최근 14일 generation_kwh (피처 1개) | `data/features.py`, `lstm_model.py` |
| 데이터 | IBM 참조 가상 시세 756행 | 삼천포 2호기 2023-01-01~2026-08-31 실적 1,339행 | `data/sample_samcheonpo2_daily.csv` |
| 배포 게이트 | RMSE ≤ $4.00 | **nRMSE ≤ 8%** (설비용량 대비, 정산금 기준과 동일) | `train_and_register.py` |
| 드리프트 판정 | 최근 21건 RMSE > $4 | 최근 21일 nRMSE > 8% | `monitoring/drift_detector.py` |
| 재학습 | 최근 21거래일 fine-tuning | 최근 30일 fine-tuning (warm start, 10 epoch) | `monitoring/retrain_trigger.py` |
| 드리프트 시나리오 | 랜덤워크 변동성 3배 | 실제 구간 주입: 정상(최근 35일) / 장마철(2026-06-20~) / 신규 발전소 편입(×1.8) | `scripts/simulate_drift.py`, `routers/data.py` `/data/window` |
| 재배포 확인 | model_version="production" | `production-v{N}` 라벨 + 승격 시 캐시 무효화 | `model_loader.py` |

**nRMSE 정의**: RMSE(kWh) ÷ (설비용량 kW × 24h) × 100. 발전량 예측제도의 시간별 오차율
(|예측−실제| ÷ 설비용량)을 일 단위로 옮긴 것입니다. 설비용량은 공식 자료를 찾지 못해 시간 발전량
피크(847 kWh/h)로부터 **1 MW 로 가정**했습니다 (`data/features.py`의 `CAPACITY_KW` 하나만 바꾸면 됩니다).

## 사전 실험 (왜 피처가 발전량 하나인가)

테스트 구간 2025-09~2026-08, 3층 LSTM, 60 epoch:

| 입력 | RMSE (kWh) | nRMSE |
|---|---|---|
| 전날 값 그대로 (persistence) | 1,650 | 6.9% |
| 발전량 14일 (채택) | 1,331 | 5.5% |
| 발전량 + 계절(sin/cos day-of-year) | 1,399 | 5.8% |

계절 피처는 도움이 되지 않았습니다. 기상(일사량·운량) 피처는 2단계 개선 항목이며
`N_FEATURES` 와 `SolarScaler.transform_point` 만 늘리면 됩니다.

## 실행 순서 (포트 8010 - HAIC 서버 8000 과 동시에 띄울 수 있음)

```bash
cd project_solar
pip install -r requirements.txt

# --- Day1 ---
uvicorn serving_app.main:app --host 0.0.0.0 --port 8010     # http://localhost:8010/ 대시보드, /docs
# 대시보드에서 data/sample_samcheonpo2_daily.csv 업로드 (또는 curl -F file=@data/sample_samcheonpo2_daily.csv localhost:8010/data/upload)
python scripts/train_baseline_v1.py                           # scaler.pkl + solarcast_v1.keras

# --- Day2 ---
python serving_app/train_and_register.py                      # MLflow 기록 + nRMSE 8% 게이트 통과 시 Production 승격
MODEL_SOURCE=mlflow uvicorn serving_app.main:app --host 0.0.0.0 --port 8010

# --- Day3 ---
python scripts/simulate_drift.py                              # 정상 -> 장마철 -> 신규 발전소 편입 순서로 주입
```

`/predict` 요청 예시:

```json
{"sequence": [{"generation_kwh": 3320.8}, {"generation_kwh": 3918.0}, {"...": "12개 더"}]}
```

응답: `{"predicted_kwh": 3410.5, "model_version": "production-v1"}`

## 완료 기준

- [ ] `/data/upload` 로 실적 CSV 를 올리면 `/data/status` 에 기간·행수가 보이는가
- [ ] 정상 배치 nRMSE ≤ 8%, 신규 발전소 편입 배치 nRMSE > 8% 가 재현되는가
- [ ] `logs/aiops.log` 에 `[WARN] drift detected` → `[INFO] retrain triggered` → `[OK] new_nrmse=...` 순서로 기록되는가
- [ ] 재배포 후 `/predict` 의 `model_version` 이 `production-v2` 로 바뀌는가
