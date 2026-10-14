"""Build a YOLO dataset from annotated frames.

Every frame in the folder is used. Frames without a label file have no logos (annotation tools
like makesense.ai only export files for frames with boxes) and are kept as background images,
which teach the model what isn't a logo.

The video is cut into blocks of a few seconds and every Nth block goes to validation. Whole blocks
stay together, so near-identical neighbouring frames don't end up in both train and val, while
validation still samples the whole video. Brands that end up in only one split are reported,
since their validation scores mean nothing.

Datasets are versioned (data/datasets/<sport>_v01, _v02, ...) and never overwritten, so a training
run can always be traced back to the exact data it used. data.yaml records where the data came from.
"""

import argparse
import shutil
from collections import Counter
from datetime import date
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frames", type=Path, help="folder of .jpg frames named <video>_<frame number>.jpg")
    parser.add_argument("labels", type=Path, help="folder of YOLO .txt labels, one per annotated frame")
    parser.add_argument("--sport", required=True)
    parser.add_argument("--block-seconds", type=float, default=2.0, help="length of the blocks the video is cut into")
    parser.add_argument("--val-every", type=int, default=5, help="every Nth block goes to validation")
    parser.add_argument("--fps", type=float, default=60.0, help="frame rate of the source video")
    parser.add_argument("--out", type=Path, help="dataset folder (default: the next data/datasets/<sport>_vNN)")
    args = parser.parse_args()

    brands = list(yaml.safe_load((Path("configs") / f"{args.sport}.yaml").read_text())["brands"])
    out = args.out or next_version(Path("data/datasets"), args.sport)
    if out.exists():
        raise SystemExit(f"{out} already exists; datasets aren't overwritten, so pick a new --out")
    pairs = [(img, args.labels / f"{img.stem}.txt") for img in sorted(args.frames.glob("*.jpg"))]

    def is_val(img: Path) -> bool:
        block = int(int(img.stem.rsplit("_", 1)[1]) / args.fps // args.block_seconds)
        return block % args.val_every == args.val_every // 2

    splits = {"train": [p for p in pairs if not is_val(p[0])], "val": [p for p in pairs if is_val(p[0])]}
    if not splits["train"] or not splits["val"]:
        raise SystemExit("Train or val is empty; use shorter --block-seconds or a smaller --val-every")

    boxes: dict[str, Counter[str]] = {}
    for split, items in splits.items():
        boxes[split] = Counter()
        for sub in ("images", "labels"):
            (out / sub / split).mkdir(parents=True, exist_ok=True)
        for img, label in items:
            shutil.copy(img, out / "images" / split / img.name)
            text = label.read_text() if label.exists() else ""
            (out / "labels" / split / label.name).write_text(text)
            boxes[split].update(brands[int(line.split()[0])] for line in text.splitlines() if line.strip())
        empty = sum(not label.exists() or not label.read_text().strip() for _, label in items)
        print(f"{split}: {len(items)} frames ({empty} without logos), boxes {dict(boxes[split])}")

    for brand in brands:
        n_train, n_val = boxes["train"][brand], boxes["val"][brand]
        if n_train and not n_val:
            print(f"Warning: {brand} is only in train ({n_train} boxes), so it isn't validated")
        elif n_val and not n_train:
            print(f"Warning: {brand} is only in val ({n_val} boxes); the model can't learn it, so it will score 0")

    # No "path" key: Ultralytics then resolves folders relative to data.yaml, so the dataset can be moved (e.g. to Colab)
    data = {
        "train": "images/train",
        "val": "images/val",
        "names": dict(enumerate(brands)),
        "source": {  # ignored by Ultralytics; records how this version was made
            "frames": str(args.frames),
            "labels": str(args.labels),
            "split": f"{args.block_seconds:g} s blocks at {args.fps:g} fps, 1 in {args.val_every} to val",
            "created": date.today().isoformat(),
        },
    }
    (out / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
    print(f"Dataset config: {out / 'data.yaml'}")


def next_version(root: Path, sport: str) -> Path:
    """The first unused data/datasets/<sport>_vNN folder."""
    n = 1
    while (root / f"{sport}_v{n:02d}").exists() or (root / f"{sport}_v{n:02d}.zip").exists():
        n += 1
    return root / f"{sport}_v{n:02d}"


if __name__ == "__main__":
    main()
