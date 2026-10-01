#!/usr/bin/env bash
# 해아림 v2 전체 루프를 한 번에 재현한다 (Day1 baseline -> Day2 MLflow 게이트 -> Day3 시나리오 7개).
#   cd project_solar && bash scripts/run_all.sh
# 산출물: logs/*.log, logs/aiops.log, serving_app/models/*, mlruns/, 마지막에 서버는 8010 포트에 떠 있는 상태로 둔다.
# CPU 기준 20~30분. 중간에 실패하면 그 단계에서 멈춘다 (set -e).
set -euo pipefail
cd "$(dirname "$0")/.."
export TF_CPP_MIN_LOG_LEVEL=2 MLFLOW_DISABLE_AGENT_HINT=1
PORT="${PORT:-8010}"
PY="${PYTHON:-python3}"

step() { echo; echo "===== $* ====="; }
wait_health() { for i in $(seq 1 240); do curl -sf "localhost:$PORT/health" >/dev/null && return 0; sleep 1; done; echo "서버가 240초 안에 뜨지 않았습니다 (logs/server_*.log 확인)"; exit 1; }
stop_server() { pkill -f "uvicorn serving_app.main:app.*--port $PORT" 2>/dev/null || true; sleep 1; }

step "0. 초기화 (이전 산출물 제거)"
rm -rf mlruns mlflow.db data/recent data/uploads logs serving_app/models/*.keras serving_app/models/*.pkl serving_app/models/production_metrics.json
mkdir -p logs data/uploads
stop_server

step "1. Day1 - 서버 기동(local) + 실적 업로드"
nohup $PY -m uvicorn serving_app.main:app --host 0.0.0.0 --port "$PORT" > logs/server_day1.log 2>&1 &
wait_health
curl -s -F "file=@data/sample_solar_hourly.csv.gz" "localhost:$PORT/data/upload"; echo
curl -s "localhost:$PORT/data/status" | head -c 300; echo

step "2. Day1 - baseline 학습 (수 분)"
$PY scripts/train_baseline_v1.py 2>&1 | grep -v "^I0000\|^WARNING\|oneDNN\|cuda\|AVX" | tee logs/train_baseline_v1.log

step "3. Day2 - MLflow 학습 + 제도 오차율 게이트 (수 분)"
$PY serving_app/train_and_register.py 2>&1 | grep -v "^I0000\|^WARNING\|oneDNN\|cuda\|AVX\|━━━\|FutureWarning\|transition_model" | tee logs/train_and_register_day2.log
grep -q "GATE PASSED" logs/train_and_register_day2.log || { echo "게이트 미통과: Production 없음"; exit 1; }

step "4. Day2 - MLflow Production 으로 서빙 전환"
stop_server
MODEL_SOURCE=mlflow nohup $PY -m uvicorn serving_app.main:app --host 0.0.0.0 --port "$PORT" > logs/server_day2_day3.log 2>&1 &
wait_health
curl -s "localhost:$PORT/health"; echo

step "5. Day3 - 시나리오 9개 (합성 4 + 실제 3 + 기후 2), 폭염화·전 발전소 변화에서 fine-tuning 각 1~2분"
$PY scripts/simulate_drift.py 2>&1 | grep -v "^I0000\|^WARNING\|oneDNN\|cuda\|AVX" | tee logs/simulate_drift_day3.log

step "6. 결과"
echo "--- logs/aiops.log"; cat logs/aiops.log
echo "--- production_metrics.json"; cat serving_app/models/production_metrics.json
echo
echo "서버가 http://localhost:$PORT/ 에 떠 있습니다 (대시보드 /, Swagger /docs). 종료: pkill -f 'uvicorn serving_app.main:app'"
