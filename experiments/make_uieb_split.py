"""UIEB 890 페어를 train/val/test 로 분할하는 재현용 CSV 생성기.

raw-890 / reference-890 의 공통 stem 을 seed 고정 셔플 후 780/55/55 로 나눠
``configs/uieb_split.csv`` (열: stem,split) 로 저장한다.

사용:  python experiments/make_uieb_split.py
"""
from __future__ import annotations

import csv
import random
import sys
from pathlib import Path

import yaml

_MARINE_ROOT = Path(__file__).resolve().parents[1]
_IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}

SEED = 42
N_VAL = 55
N_TEST = 55


def main() -> int:
    paths_yaml = _MARINE_ROOT / "configs" / "paths_marine.yaml"
    cfg = yaml.safe_load(open(paths_yaml, encoding="utf-8"))
    root = Path(cfg["marine_root"]).resolve()
    uieb = root / cfg["datasets"]["uieb"]
    out_csv = root / cfg["uieb_split_csv"]

    raw = uieb / "raw-890"
    ref = uieb / "reference-890"
    raw_stems = {p.stem for p in raw.iterdir() if p.suffix.lower() in _IMG_EXTS}
    ref_stems = {p.stem for p in ref.iterdir() if p.suffix.lower() in _IMG_EXTS}
    stems = sorted(raw_stems & ref_stems)
    print(f"UIEB 공통 페어 stem: {len(stems)} (raw={len(raw_stems)}, ref={len(ref_stems)})")
    if len(stems) < N_VAL + N_TEST + 1:
        print("[error] 페어 수가 너무 적습니다.", file=sys.stderr)
        return 1

    rng = random.Random(SEED)
    rng.shuffle(stems)
    test = set(stems[:N_TEST])
    val = set(stems[N_TEST:N_TEST + N_VAL])
    # 나머지 = train

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    n = {"train": 0, "val": 0, "test": 0}
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["stem", "split"])
        for s in sorted(stems):
            split = "test" if s in test else ("val" if s in val else "train")
            n[split] += 1
            w.writerow([s, split])

    print(f"저장: {out_csv}")
    print(f"  train={n['train']}  val={n['val']}  test={n['test']}  (seed={SEED})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
