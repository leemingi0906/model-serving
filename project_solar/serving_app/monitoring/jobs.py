"""
[Day3] 재학습 백그라운드 작업 - serving_app/monitoring/jobs.py (SolarCast v2)

재학습(fine-tuning 1~2분)을 요청 스레드에서 돌리면 그동안 /predict 가 멈춘다. 그래서
  - 작업은 워커 1개짜리 스레드 풀에서 돌리고, 요청은 바로 {"job_id", "status": "queued"} 를 받는다.
  - 이미 queued/running 인 작업이 있으면 새 요청은 같은 작업으로 합친다 (중복 학습 방지).
  - 학습 중에도 서빙은 기존 Production 모델로 계속 응답하고, 승격되면 그때 캐시를 비워 교체한다.
  - 상태는 GET /jobs/current 로 본다. 마지막 결과는 logs/last_job.json 에도 남긴다.
"""
import json
import logging
import os
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

logger = logging.getLogger("aiops")

_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="solarcast-train")
_guard = threading.Lock()
_job: dict = {"status": "idle"}
LAST_JOB_PATH = os.path.join("logs", "last_job.json")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def job_status() -> dict:
    with _guard:
        return dict(_job)


def submit(fn, **meta) -> dict:
    """fn() 을 백그라운드에서 실행. 진행 중인 작업이 있으면 그 작업 정보를 deduplicated=True 로 돌려준다."""
    global _job
    with _guard:
        if _job.get("status") in ("queued", "running"):
            return {**_job, "deduplicated": True}
        job_id = uuid.uuid4().hex[:12]
        _job = {"job_id": job_id, "status": "queued", "started_at": _now(), **meta}
        snapshot = dict(_job)
    _executor.submit(_run, fn, job_id)
    return snapshot


def _run(fn, job_id: str) -> None:
    global _job
    with _guard:
        _job["status"] = "running"
    try:
        result = fn()
        with _guard:
            _job.update(status="completed", result=result, finished_at=_now())
    except Exception as exc:  # 학습 실패 -> 기존 모델 유지, 상태만 남긴다
        logger.exception("[FAILED] retrain job failed; existing Production retained")
        with _guard:
            _job.update(status="failed", error=f"{type(exc).__name__}: {exc}", finished_at=_now())
    finally:
        try:
            os.makedirs(os.path.dirname(LAST_JOB_PATH), exist_ok=True)
            with open(LAST_JOB_PATH, "w", encoding="utf-8") as f:
                json.dump(job_status(), f, ensure_ascii=False, indent=1, default=str)
        except OSError:
            pass


def run_sync(fn, **meta) -> dict:
    """테스트·스크립트용: 같은 상태 기록을 남기되 현재 스레드에서 바로 실행한다."""
    global _job
    job_id = uuid.uuid4().hex[:12]
    with _guard:
        _job = {"job_id": job_id, "status": "queued", "started_at": _now(), **meta}
    _run(fn, job_id)
    return job_status()
