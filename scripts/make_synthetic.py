"""가상 데이터를 data/synthetic/ 에 만든다. 사용: python scripts/make_synthetic.py [--seed N] [--out DIR]"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jangsa.synthetic import SEED, write_all  # noqa: E402
from jangsa.config import DATA_SYNTHETIC  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", default=str(DATA_SYNTHETIC))
    a = ap.parse_args()
    for p in write_all(a.out, a.seed):
        print(f"{p.name}  {p.stat().st_size // 1024} KB")
    print("전부 가상 데이터입니다(제주 실제 통계 아님).")
