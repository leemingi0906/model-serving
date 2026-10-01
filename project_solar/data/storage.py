"""
업로드된 발전 실적 파일 관리 + 운영 중 들어온 최근 실적(배치 테스트로 주입된 값) 보관.

    data/uploads/   대시보드 /data/upload 로 올린 CSV(.csv 또는 .csv.gz). 학습은 항상 가장 최근 파일.
    data/recent/    /predict/batch-test 가 "실적 도착"으로 받아들인 시간별 값. plant_id 별 CSV 에 덧붙인다.
                    재학습(fine_tune)은 업로드 데이터 위에 이 값을 덮어써서(최근이 우선) 최근 30일을 만든다.
                    -> 드리프트를 일으킨 바로 그 데이터로 재학습이 이뤄진다 (HAIC 실습의 단순화와 다른 점).
"""
import csv
import glob
import os

UPLOAD_DIR = "data/uploads"
RECENT_DIR = "data/recent"


def latest_upload(upload_dir: str = UPLOAD_DIR) -> str:
    files = sorted(glob.glob(os.path.join(upload_dir, "*.csv")) + glob.glob(os.path.join(upload_dir, "*.csv.gz")),
                   key=os.path.getmtime)
    if not files:
        raise FileNotFoundError(
            "업로드된 발전 실적이 없습니다. 대시보드에서 CSV 를 먼저 업로드하세요 "
            f"(data/sample_solar_hourly.csv.gz 를 예시로 올릴 수 있습니다 -> {upload_dir}/)."
        )
    return files[-1]


def append_recent(plant_id: str, records: list[dict], recent_dir: str = RECENT_DIR) -> str:
    """records = [{'time': 'YYYY-MM-DD HH:00', 'generation_kwh': float}, ...]"""
    os.makedirs(recent_dir, exist_ok=True)
    path = os.path.join(recent_dir, f"{plant_id}.csv")
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "generation_kwh"])
        for r in records:
            w.writerow([r["time"], r["generation_kwh"]])
    return path


def load_recent(recent_dir: str = RECENT_DIR) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for path in glob.glob(os.path.join(recent_dir, "*.csv")):
        pid = os.path.basename(path)[:-4]
        with open(path, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                out.setdefault(pid, {})[r["time"]] = float(r["generation_kwh"])  # 뒤에 쓴 값이 이김
    return out


def clear_recent(recent_dir: str = RECENT_DIR) -> None:
    for path in glob.glob(os.path.join(recent_dir, "*.csv")):
        os.remove(path)
