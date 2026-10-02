# Day2: 해아림 서빙 앱 컨테이너 이미지 (단일 컨테이너, 자기 완결형)
#   빌드/실행은 project_solar/ 에서:
#     docker compose -f serving_app/docker-compose.yml up --build
#   또는
#     docker build -f serving_app/Dockerfile -t haearim-serving:v2 .
#     docker run --rm -p 8010:8010 haearim-serving:v2
#
# 이미지 빌드 시점에 컨테이너 안에서 (1) 샘플 실적을 data/uploads/ 에 "업로드" 해 두고
# (2) 동봉 모델을 평가·게이트·MLflow 등록(--register-local, 학습 없음, 1~2분) 까지 끝낸다.
# MLflow 는 별도 서버 없이 컨테이너 안 sqlite(mlflow.db) 로 동작하므로 학습(등록)과 서빙이
# 같은 파일시스템(/app) 안에서 이뤄진다. 처음부터 학습하려면 아래 ARG 를 full 로 바꾼다 (CPU 20분 안팎).
# tensorflow 가 들어가 이미지가 1GB 대이고 pip 설치에 몇 분 걸린다.
FROM python:3.11-slim

WORKDIR /app
ENV TF_CPP_MIN_LOG_LEVEL=2 MLFLOW_DISABLE_AGENT_HINT=1 PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY serving_app/ serving_app/
COPY data/ data/
COPY scripts/ scripts/
COPY tests/ tests/

# TRAIN=local : 동봉 모델(serving_app/models/solarcast_v2.keras) 을 같은 게이트로 등록 (기본)
# TRAIN=full  : Day1 baseline 학습 + Day2 scratch 학습까지 컨테이너 안에서 수행
ARG TRAIN=local
RUN rm -rf data/uploads data/recent mlruns mlflow.db logs && mkdir -p data/uploads logs \
 && cp data/sample_solar_hourly.csv.gz data/uploads/build_seed.csv.gz \
 && if [ "$TRAIN" = "full" ]; then \
      python scripts/train_baseline_v1.py && python serving_app/train_and_register.py; \
    else \
      python serving_app/train_and_register.py --register-local; \
    fi \
 && python -m pytest -q tests

EXPOSE 8010

ENV MODEL_SOURCE=mlflow
ENV LOADING_MODE=eager
# 헬스체크: 프로세스가 살아 있고(status ok) 모델까지 올라왔는지(model_loaded) - eager 라 기동 직후 true
HEALTHCHECK --interval=15s --timeout=5s --start-period=90s --retries=5 \
  CMD python -c "import json,urllib.request;d=json.load(urllib.request.urlopen('http://127.0.0.1:8010/health'));exit(0 if d.get('model_loaded') else 1)"

CMD ["uvicorn", "serving_app.main:app", "--host", "0.0.0.0", "--port", "8010"]
