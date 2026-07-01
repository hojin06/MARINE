"""DUO(COCO) → YOLO 형식 변환 + duo.yaml 생성 (M4.2 준비).

DUO/annotations/instances_{train,test}.json 을 읽어 이미지별 YOLO 라벨
(labels/{train,test}/<stem>.txt, 각 줄: cls cx cy w h [0~1])을 만들고,
ultralytics 학습용 configs/duo.yaml 을 작성한다.

카테고리(1..4) → YOLO 클래스(0..3): holothurian, echinus, scallop, starfish.

사용: python experiments/prep_duo_yolo.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

_MARINE_ROOT = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore
    except Exception:
        pass

NAMES = ["holothurian", "echinus", "scallop", "starfish"]  # cat_id 1..4 → 0..3


def convert_split(duo_root: Path, split: str) -> int:
    ann = duo_root / "annotations" / f"instances_{split}.json"
    img_dir = duo_root / "images" / split
    lbl_dir = duo_root / "labels" / split
    lbl_dir.mkdir(parents=True, exist_ok=True)
    d = json.load(open(ann, encoding="utf-8"))
    imgs = {im["id"]: im for im in d["images"]}
    by_img: dict = {i: [] for i in imgs}
    for a in d["annotations"]:
        by_img.setdefault(a["image_id"], []).append(a)
    n = 0
    for iid, im in imgs.items():
        W, H = im["width"], im["height"]
        stem = Path(im["file_name"]).stem
        lines = []
        for a in by_img.get(iid, []):
            x, y, w, h = a["bbox"]
            cls = a["category_id"] - 1  # 1..4 → 0..3
            cx = (x + w / 2) / W
            cy = (y + h / 2) / H
            nw, nh = w / W, h / H
            if nw <= 0 or nh <= 0:
                continue
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
        (lbl_dir / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")
        n += 1
    print(f"  {split}: {n} 라벨 파일 → {lbl_dir}  (이미지 {len(list(img_dir.glob('*.jpg')))}장)")
    return n


def main() -> int:
    paths = yaml.safe_load(open(_MARINE_ROOT / "configs" / "paths_marine.yaml", encoding="utf-8"))
    duo_root = Path(paths["marine_root"]) / paths["datasets"].get("duo", "datasets/DUO")
    if not (duo_root / "annotations").is_dir():
        # paths_marine 에 duo 키 없으면 기본 경로
        duo_root = _MARINE_ROOT / "datasets" / "DUO"
    print(f"DUO root: {duo_root}")
    for sp in ("train", "test"):
        convert_split(duo_root, sp)

    duo_yaml = {
        "path": str(duo_root).replace("\\", "/"),
        "train": "images/train",
        "val": "images/test",
        "names": {i: n for i, n in enumerate(NAMES)},
    }
    out = _MARINE_ROOT / "configs" / "duo.yaml"
    with open(out, "w", encoding="utf-8") as f:
        yaml.safe_dump(duo_yaml, f, allow_unicode=True, sort_keys=False)
    print(f"duo.yaml → {out}")
    print(yaml.safe_dump(duo_yaml, allow_unicode=True, sort_keys=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
