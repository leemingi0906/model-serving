"""
원본 월별 CSV(발전소별 1~24시 발전량) -> 시간별 long 포맷 데이터셋.

    python team_solar/scripts/build_hourly_dataset.py
    -> team_solar/data/all_plants_hourly.csv.gz
       컬럼: plant_id, time(KST 'YYYY-MM-DD HH:00', HH = 구간 끝 시각: 01:00 = 00~01시 = 원본 "1시 발전량"),
             generation_kwh, cf(이용률 = generation_kwh / capacity_kw)

plants.csv 의 plant_id 매핑과 설비용량(capacity_kw_official 이 있으면 그 값, 없으면 capacity_kw_est)을 쓴다.
use=0 발전소도 포함해 저장한다 (실제 정지 구간을 드리프트 사례로 쓰기 위해). 모델 학습 시 use=1 로 거른다.
"""
import csv
import glob
import gzip
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    with open(os.path.join(ROOT, "plants.csv"), encoding="utf-8") as f:
        plants = {(r["plant"], r["unit"]): r for r in csv.DictReader(f)}

    out_path = os.path.join(ROOT, "data", "all_plants_hourly.csv.gz")
    n = 0
    unknown = set()
    with gzip.open(out_path, "wt", newline="", encoding="utf-8") as out:
        w = csv.writer(out)
        w.writerow(["plant_id", "time", "generation_kwh", "cf"])
        for fpath in sorted(glob.glob(os.path.join(ROOT, "data_raw", "monthly", "*.csv"))):
            with open(fpath, encoding="utf-8-sig") as fh:
                rd = csv.reader(fh)
                next(rd)
                for r in rd:
                    r = [c.strip() for c in r]
                    if len(r) < 27:
                        continue
                    meta = plants.get((r[0], r[1]))
                    if meta is None:
                        unknown.add((r[0], r[1]))
                        continue
                    cap = float(meta["capacity_kw_official"] or meta["capacity_kw_est"])
                    day = r[2][:10]
                    for h in range(24):
                        kwh = float(r[3 + h] or 0)
                        w.writerow([meta["plant_id"], f"{day} {h + 1:02d}:00", round(kwh, 3), round(kwh / cap, 5) if cap else ""])
                        n += 1
    print(f"{n}행 -> {os.path.relpath(out_path, ROOT)}")
    if unknown:
        print("plants.csv 에 없는 발전소 (건너뜀):", sorted(unknown))


if __name__ == "__main__":
    main()
