# 해아림 v2 - 팀원 실행 안내

이 폴더(`project_solar/`)만 있으면 됩니다. 실적 10곳 3.7년, Open-Meteo 기상(관측·하루 전 예보), 학습된 모델, 대시보드, Swagger 가 전부 들어 있습니다.

## 1. 준비 (한 번만)

- Python 3.10 ~ 3.12 (3.11 권장). TensorFlow 2.21 이 설치되므로 디스크 2GB, 메모리 4GB 이상.
- 터미널에서:

```bash
cd project_solar
python -m venv .venv
# Windows PowerShell:  .venv\Scripts\Activate.ps1      macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
```

## 2. 빠른 시작 (학습 없음, 3~5분)

macOS / Linux / WSL:

```bash
bash scripts/quick_start.sh
```

Windows PowerShell (같은 일을 손으로):

```powershell
$env:TF_CPP_MIN_LOG_LEVEL="2"
Remove-Item -Recurse -Force mlruns, mlflow.db, data\recent, data\uploads, logs -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force logs, data\uploads | Out-Null
# (1) 실적 업로드: 서버를 local 모드로 띄운 뒤 다른 창에서
Start-Process -NoNewWindow python -ArgumentList "-m uvicorn serving_app.main:app --host 0.0.0.0 --port 8010"
Start-Sleep 60
curl.exe -s -F "file=@data/sample_solar_hourly.csv.gz" http://localhost:8010/data/upload
Get-Process python | Stop-Process
# (2) 동봉 모델을 평가·게이트·MLflow 등록 (1~2분)
python serving_app/train_and_register.py --register-local
# (3) MLflow champion 으로 서빙
$env:MODEL_SOURCE="mlflow"
python -m uvicorn serving_app.main:app --host 0.0.0.0 --port 8010
```

브라우저: 대시보드 http://localhost:8010 · Swagger http://localhost:8010/docs

## 2-1. Docker 로 띄우기 (Day2 컨테이너 재현)

Docker Desktop 이 설치돼 있으면 Python 설치 없이 한 줄로 같은 화면이 뜹니다. 이미지 빌드 시점에 실적 시드 → 동봉 모델 평가·게이트·MLflow 등록 → 계약 테스트까지 끝납니다 (pip 설치 포함 5~10분, 이미지 1GB 대).

```bash
cd project_solar
docker compose -f serving_app/docker-compose.yml up --build
# 빌드 로그에 [GATE PASSED] 와 "14 passed" 가 보이면 정상. 뜨고 나면 http://localhost:8010 · /docs
```

- 컨테이너는 `MODEL_SOURCE=mlflow`, `LOADING_MODE=eager` 로 기동하므로 `/health` 가 처음부터 `model_loaded: true` 입니다. `docker ps` 의 STATUS 가 `healthy` 가 되면 트래픽을 받을 준비가 된 것입니다.
- 처음부터 학습한 이미지를 원하면 `TRAIN=full docker compose -f serving_app/docker-compose.yml up --build` (CPU 20분 안팎).
- 운영 로그는 `haearim-logs` 볼륨에 남습니다. 꺼내려면 `docker compose -f serving_app/docker-compose.yml exec serving-app cat logs/aiops.log`.
- 멈추기: `docker compose -f serving_app/docker-compose.yml down` (볼륨까지 지우려면 `-v`).

## 3. 화면에서 해볼 것 (대시보드 버튼 순서)

| 버튼 | 무엇이 보이나 | 기대 결과 |
|---|---|---|
| 정상 (삼천포2 최근 21일) | 오차 7.2%, PR 0.98 | 정상, 파이프라인 4단계까지 완료 |
| 장마철 | 오차 9.2%, PR 1.05 | 정상 (임계값 10.36% 이내) |
| 설비 고장 (8일째부터 ×0.5) | 오차 16%, PR 0.46 | 설비 이상 → `[ALERT]`, 재학습 트리거 대기 |
| 광양항 / 삼천포2 / 영흥#5 실제 | 실제 데이터 3건 | 모두 설비 이상 → 알림만 |
| 기후 A: 우기 패턴만 | 6곳 장마철 그대로 | 정상 (모델이 아는 날씨) |
| 기후 B: 폭염화 관계 변화 | 6곳, 경남 4곳만 임계 초과 | 모델 드리프트 → **백그라운드 재학습**(1~2분) → 게이트 → v2 승격. 학습 중에도 `/predict` 는 v1 로 응답 |
| 전 발전소 변화 (3곳 ×1.25) | 루프 동작 검증용 | 모델 드리프트 → 재학습 → v3 |

재학습이 돌 때 상단 배지가 "재학습 running" 으로 바뀌고, 끝나면 파이프라인 8단계가 모두 켜지며 `aiops.log` 에 `[OK] ... promoted` 가 남습니다. Swagger `/predict` 응답의 `model_version` 이 production-v1 → v2 로 바뀐 것을 확인하세요.

터미널에서 한 번에 돌리려면 `python scripts/simulate_drift.py` (재학습 작업은 자동으로 기다립니다).

## 4. 파일 안내

| 경로 | 내용 |
|---|---|
| `data/sample_solar_hourly.csv.gz` | 발전소 10곳 시간별 실적 (plant_id, time, generation_kwh). time 은 구간 끝 시각 `YYYY-MM-DD HH:00` (HH=01..24) |
| `data/weather/obs.csv.gz`, `d1.csv.gz` | Open-Meteo 관측 / 하루 전 예보 (일사량·운량·기온), 발전소 위치 7곳 |
| `data/plants.csv` | 발전소 레지스트리 (추정 설비용량, 좌표, 사용 여부). 공식 용량·좌표로 바꾸면 그대로 반영 |
| `serving_app/models/` | 동봉 모델 `solarcast_v2.keras`, 정규화 상수 `scaler.pkl` |
| `serving_app/routers/` | `/predict`, `/predict/batch-test`, `/data/upload·status·window`, `/jobs/current`, `/health`, `/logs` |
| `serving_app/monitoring/` | `drift_detector.py`(원인 분류), `retrain_trigger.py`(대응), `jobs.py`(백그라운드 작업) |
| `serving_app/train_and_register.py` | 학습·게이트·MLflow 등록 (`--register-local` 로 학습 생략) |
| `scripts/simulate_drift.py` | 시나리오 9개 자동 실행 · `scripts/run_all.sh` 처음부터 전체 학습(20~30분) |
| `tests/` | `python -m pytest -q` 계약 테스트 14개 |
| `logs/aiops.log` | 감지·알림·재학습·승격 기록 (발표 증빙) |

## 5. 자주 막히는 곳

- `pip install` 에서 TensorFlow 가 실패하면 Python 버전을 3.11 로 맞추세요 (3.13 은 미지원).
- 서버가 뜨는 데 TensorFlow 로딩 때문에 30~60초 걸립니다. `/health` 가 200 이면 준비된 것입니다.
- 포트 8010 이 쓰이고 있으면 `PORT=8020 bash scripts/quick_start.sh` 처럼 바꾸세요 (스크립트 안의 주소도 같이).
- 처음부터 다시 학습하고 싶으면 `bash scripts/run_all.sh` (CPU 20~30분). 결과 수치는 `README.md` 결과표와 같아야 합니다.
- MLflow UI 를 보고 싶으면 `mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000`.
