"""
make_dataset.py - (re)generate the simulated plant year.

    python scripts/make_dataset.py                 # config defaults: 330 batches, seed 42 -> data/raw/
    python scripts/make_dataset.py --n 500 --seed 7 --out data/raw_seed7

Deterministic for a given seed. SIMULATED data - not plant performance.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tsd_twin.config import load_config  # noqa: E402
from tsd_twin.simulate import generate_dataset  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate the simulated TSD dataset")
    ap.add_argument("--n", type=int, default=None, help="number of batches (default: config)")
    ap.add_argument("--seed", type=int, default=None, help="random seed (default: config)")
    ap.add_argument("--out", type=str, default="data/raw", help="output folder relative to the project")
    args = ap.parse_args()
    cfg = load_config()
    log = generate_dataset(cfg, ROOT / args.out, n_batches=args.n, seed=args.seed)
    print(f"written {log['n_batches']} batches / {log['rows']:,} rows to {args.out}  (seed {log['seed']})")


if __name__ == "__main__":
    main()
