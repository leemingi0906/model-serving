"""
team_solar/ 의 원천 데이터에서 project_solar/data/ 가 쓰는 파일을 만든다 (저장소 루트에서 1회 실행).

    python project_solar/scripts/build_data.py

생성
    data/plants.csv                 발전소 레지스트리 (team_solar/plants.csv 복사)
    data/sample_solar_hourly.csv.gz 시간별 발전 실적, hourly_ok=1 인 발전소만 (대시보드 업로드용 샘플)
    data/weather/obs.csv.gz         과거 관측 기상 (학습·사후 평가용), 사용 위치만, long 포맷
    data/weather/d1.csv.gz          하루 전 발행 예보 (서빙 시뮬레이션용), 같은 포맷
"""
import csv
import gzip
import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
TEAM = os.path.join(os.path.dirname(PROJ), "team_solar")
VARS = ["shortwave_radiation", "cloud_cover", "temperature_2m"]


def main():
    os.makedirs(os.path.join(PROJ, "data", "weather"), exist_ok=True)
    shutil.copy(os.path.join(TEAM, "plants.csv"), os.path.join(PROJ, "data", "plants.csv"))
    plants = list(csv.DictReader(open(os.path.join(PROJ, "data", "plants.csv"), encoding="utf-8")))
    use = {p["plant_id"]: p for p in plants if p["use"] == "1" and p["hourly_ok"] == "1"}
    locs = sorted({(str(round(float(p["lat"]), 3)), str(round(float(p["lon"]), 3))) for p in use.values()})
    print(f"발전소 {len(use)}곳, 위치 {len(locs)}곳")

    n = 0
    with gzip.open(os.path.join(TEAM, "data", "all_plants_hourly.csv.gz"), "rt", encoding="utf-8") as src, \
         gzip.open(os.path.join(PROJ, "data", "sample_solar_hourly.csv.gz"), "wt", newline="", encoding="utf-8") as dst:
        rd = csv.DictReader(src)
        w = csv.writer(dst)
        w.writerow(["plant_id", "time", "generation_kwh"])
        for r in rd:
            if r["plant_id"] in use:
                w.writerow([r["plant_id"], r["time"], r["generation_kwh"]])
                n += 1
    print(f"sample_solar_hourly.csv.gz: {n}행")

    for kind in ("obs", "d1"):
        out = os.path.join(PROJ, "data", "weather", f"{kind}.csv.gz")
        m = 0
        with gzip.open(out, "wt", newline="", encoding="utf-8") as dst:
            w = csv.writer(dst)
            w.writerow(["loc", "time"] + VARS)
            for lat, lon in locs:
                path = os.path.join(TEAM, "data_raw", "weather", f"{kind}_{lat}_{lon}.csv")
                for r in csv.DictReader(open(path, encoding="utf-8")):
                    w.writerow([f"{lat}_{lon}", r["time"]] + [r[v] for v in VARS])
                    m += 1
        print(f"weather/{kind}.csv.gz: {m}행")


if __name__ == "__main__":
    main()
