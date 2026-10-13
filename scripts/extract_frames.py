"""Save evenly spaced frames from a video for annotation.

Also writes labels.txt (the sport's brands, one per line) so the annotation tool's
class order matches the config.
"""

import argparse
from pathlib import Path

import cv2
import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--sport", required=True)
    parser.add_argument("--start", type=float, default=0, help="start time in seconds")
    parser.add_argument("--end", type=float, help="end time in seconds (default: end of video)")
    parser.add_argument("--count", type=int, default=60, help="number of frames to save")
    parser.add_argument("--out", type=Path, help="output folder (default: data/frames/<video name>)")
    args = parser.parse_args()

    out = args.out or Path("data/frames") / args.video.stem
    out.mkdir(parents=True, exist_ok=True)
    brands = yaml.safe_load((Path("configs") / f"{args.sport}.yaml").read_text())["brands"]
    (out / "labels.txt").write_text("\n".join(brands) + "\n")

    cap = cv2.VideoCapture(str(args.video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    last = int(args.end * fps) if args.end else int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    first = int(args.start * fps)
    step = max(1, (last - first) // args.count)
    for idx in range(first, last, step)[: args.count]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok:
            cv2.imwrite(str(out / f"{args.video.stem}_{idx:06d}.jpg"), frame)
    print(f"Saved frames and labels.txt to {out}")


if __name__ == "__main__":
    main()
