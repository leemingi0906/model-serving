#!/usr/bin/env bash
# 팀원용 빠른 시작: 10분 학습 없이 동봉된 모델을 등록해 Day2 상태(MLflow champion)로 만들고 서버를 띄운다. 3~5분.
#   cd project_solar && bash scripts/quick_start.sh
# 그 뒤 브라우저에서 http://localhost:8010 (대시보드) / http://localhost:8010/docs (Swagger)
# 시나리오 9개를 자동으로 돌리려면:  python scripts/simulate_drift.py
set -euo pipefail
cd "$(dirname "$0")/.."
export TF_CPP_MIN_LOG_LEVEL=2 MLFLOW_DISABLE_AGENT_HINT=1
PORT="${PORT:-8010}"
PY="${PYTHON:-python3}"
step() { echo; echo "===== $* ====="; }
wait_health() { for i in $(seq 1 240); do curl -sf "localhost:$PORT/health" >/dev/null && return 0; sleep 1; done; echo "서버가 240초 안에 뜨지 않았습니다 (logs/server_*.log 확인)"; exit 1; }
stop_server() { pkill -f "uvicorn serving_app.main:app.*--port $PORT" 2>/dev/null || true; sleep 1; }

[ -f serving_app/models/solarcast_v2.keras ] || { echo "serving_app/models/solarcast_v2.keras 가 없습니다. scripts/run_all.sh(전체 학습) 를 쓰세요."; exit 1; }

step "0. 초기화 (이전 MLflow·로그·운영 실적 제거, 동봉 모델은 유지)"
rm -rf mlruns mlflow.db data/recent data/uploads logs serving_app/models/production_metrics.json
mkdir -p logs data/uploads
stop_server

step "1. 실적 업로드 (서버 local 모드)"
nohup $PY -m uvicorn serving_app.main:app --host 0.0.0.0 --port "$PORT" > logs/server_local.log 2>&1 &
wait_health
curl -s -F "file=@data/sample_solar_hourly.csv.gz" "localhost:$PORT/data/upload"; echo
stop_server

step "2. 동봉 모델 평가 -> 게이트 -> MLflow 등록 (1~2분, 학습 없음)"
$PY serving_app/train_and_register.py --register-local 2>&1 | grep -v "^I0000\|^WARNING\|oneDNN\|cuda\|AVX\|━━━\|FutureWarning" | tee logs/register_local.log
grep -q "GATE PASSED" logs/register_local.log || { echo "게이트 미통과"; exit 1; }

step "3. MLflow champion 으로 서빙 시작"
MODEL_SOURCE=mlflow nohup $PY -m uvicorn serving_app.main:app --host 0.0.0.0 --port "$PORT" > logs/server.log 2>&1 &
wait_health
curl -s "localhost:$PORT/health"; echo
echo
echo "준비 끝. 대시보드 http://localhost:$PORT/  ·  Swagger http://localhost:$PORT/docs"
echo "시나리오 자동 실행:  python scripts/simulate_drift.py     종료: pkill -f 'uvicorn serving_app.main:app'"
