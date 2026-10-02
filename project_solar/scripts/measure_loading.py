"""
Day1 실습 포인트: Lazy vs Eager 로딩을 실제로 재서 표로 남긴다.
  cd project_solar && python scripts/measure_loading.py            # MODEL_SOURCE=mlflow (quick_start 뒤)
  MODEL_SOURCE=local python scripts/measure_loading.py             # 동봉 모델로 바로

각 모드마다 서버를 새로 띄우고 (1) /health 가 200 이 될 때까지 (2) 첫 /predict (3) 두 번째 /predict 시간을 잰다.
결과는 logs/loading_measure.json 과 표준출력(마크다운 표)에 남는다. 기존 8010 서버는 끝낸 뒤 다시 띄우지 않으므로
quick_start 상태로 돌아가려면 MODEL_SOURCE=mlflow uvicorn serving_app.main:app --port 8010 을 다시 실행한다.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
PORT = int(os.getenv("PORT", "8010"))
BASE = f"http://localhost:{PORT}"
REQ = ROOT / "tests" / "sample_predict_request.json"
RUNS = int(os.getenv("RUNS", "2"))


def sample_request() -> dict:
    if REQ.exists():
        return json.loads(REQ.read_text())
    # 업로드된 실적에서 72h+24h 구간을 받아 /predict 입력으로 만든다
    w = requests.get(f"{BASE}/data/window", params={"plant_id": "samcheonpo_2", "n_days": 4}, timeout=30).json()
    recs = w["records"]
    hist = [r["generation_kwh"] for r in recs[:72]]
    date = recs[72]["time"][:10]
    return {"plant_id": "samcheonpo_2", "date": date, "history_kwh": hist,
            "forecast": [{"ghi": max(0.0, 600.0 * __import__("math").sin(__import__("math").pi * (h - 6) / 12)) if 6 <= h <= 18 else 0.0,
                          "cloud_cover": 30.0, "temperature": 24.0} for h in range(1, 25)]}


def stop():
    subprocess.run(["pkill", "-f", f"uvicorn serving_app.main:app.*--port {PORT}"], check=False)
    time.sleep(1.5)


def measure(mode: str, req: dict) -> dict:
    stop()
    env = {**os.environ, "LOADING_MODE": mode, "TF_CPP_MIN_LOG_LEVEL": "2"}
    env.setdefault("MODEL_SOURCE", "mlflow")
    log = open(ROOT / "logs" / f"server_measure_{mode}.log", "w")
    t0 = time.perf_counter()
    p = subprocess.Popen([sys.executable, "-m", "uvicorn", "serving_app.main:app", "--host", "0.0.0.0", "--port", str(PORT)],
                         cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    health = None
    while time.perf_counter() - t0 < 300:
        try:
            r = requests.get(f"{BASE}/health", timeout=2)
            if r.status_code == 200:
                health = r.json()
                break
        except requests.RequestException:
            pass
        time.sleep(0.2)
    startup = time.perf_counter() - t0
    if health is None:
        p.kill()
        raise SystemExit(f"{mode}: 서버가 뜨지 않았습니다")
    lat = []
    for _ in range(2):
        t1 = time.perf_counter()
        r = requests.post(f"{BASE}/predict", json=req, timeout=120)
        lat.append(time.perf_counter() - t1)
        r.raise_for_status()
    ver = r.json().get("model_version")
    out = {"mode": mode, "startup_s": round(startup, 2), "health_model_loaded_at_start": health.get("model_loaded"),
           "first_predict_s": round(lat[0], 3), "second_predict_s": round(lat[1], 3), "model_version": ver}
    stop()
    return out


def main():
    (ROOT / "logs").mkdir(exist_ok=True)
    # 입력은 현재 떠 있는 서버(quick_start)에서 받는다. 없으면 샘플 파일이 있어야 한다.
    try:
        req = sample_request()
    except requests.RequestException:
        raise SystemExit("8010 서버가 없어 /predict 입력을 만들 수 없습니다. quick_start.sh 를 먼저 실행하세요.")
    results = []
    for mode in ["lazy", "eager"]:
        for i in range(RUNS):
            r = measure(mode, req)
            r["run"] = i + 1
            results.append(r)
            print(json.dumps(r, ensure_ascii=False))
    (ROOT / "logs" / "loading_measure.json").write_text(json.dumps(results, ensure_ascii=False, indent=1))
    print("\n| 모드 | 회차 | 서버 시작 → /health 200 | 시작 직후 model_loaded | 첫 /predict | 두 번째 /predict |")
    print("|---|---|---|---|---|---|")
    for r in results:
        print(f"| {r['mode']} | {r['run']} | {r['startup_s']} s | {r['health_model_loaded_at_start']} | {r['first_predict_s']*1000:.0f} ms | {r['second_predict_s']*1000:.0f} ms |")


if __name__ == "__main__":
    main()
