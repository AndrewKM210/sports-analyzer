"""Build a YOLO dataset from annotated frames.

Every frame in the folder is used. Frames without a label file have no logos (annotation tools
like makesense.ai only export files for frames with boxes) and are kept as background images,
which teach the model what isn't a logo. The split is by time (the last frames go to validation)
so that near-identical neighbouring frames don't end up in both train and val.
"""

import argparse
import shutil
from collections import Counter
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frames", type=Path, help="folder of .jpg frames")
    parser.add_argument("labels", type=Path, help="folder of YOLO .txt labels, one per annotated frame")
    parser.add_argument("--sport", required=True)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--out", type=Path, help="dataset folder (default: data/datasets/<sport>)")
    args = parser.parse_args()

    brands = list(yaml.safe_load((Path("configs") / f"{args.sport}.yaml").read_text())["brands"])
    out = args.out or Path("data/datasets") / args.sport
    pairs = [(img, args.labels / f"{img.stem}.txt") for img in sorted(args.frames.glob("*.jpg"))]
    if len(pairs) < 2:
        raise SystemExit(f"Need at least 2 frames, found {len(pairs)}")
    n_val = max(1, round(len(pairs) * args.val_fraction))

    for sub in ("images", "labels"):
        shutil.rmtree(out / sub, ignore_errors=True)
    for split, items in [("train", pairs[:-n_val]), ("val", pairs[-n_val:])]:
        boxes: Counter[str] = Counter()
        for sub in ("images", "labels"):
            (out / sub / split).mkdir(parents=True, exist_ok=True)
        for img, label in items:
            shutil.copy(img, out / "images" / split / img.name)
            text = label.read_text() if label.exists() else ""
            (out / "labels" / split / label.name).write_text(text)
            for line in text.splitlines():
                if line.strip():
                    cls = int(line.split()[0])
                    boxes[brands[cls] if cls < len(brands) else f"unknown class {cls}"] += 1
        empty = sum(not label.exists() or not label.read_text().strip() for _, label in items)
        print(f"{split}: {len(items)} frames ({empty} without logos), boxes {dict(boxes)}")

    data = {"path": str(out.resolve()), "train": "images/train", "val": "images/val", "names": dict(enumerate(brands))}
    (out / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
    print(f"Dataset config: {out / 'data.yaml'}")


if __name__ == "__main__":
    main()
