"""Command-line entry point: `sports-analyzer analyze <video> --detector ...`."""

import argparse
from pathlib import Path

import yaml

from sports_analyzer.detectors import Detector, FinetunedDetector, ZeroShotDetector
from sports_analyzer.pipeline import analyze_video

CONFIG_DIR = Path("configs")


def main() -> None:
    """Parse arguments and run the requested command."""
    parser = argparse.ArgumentParser(
        prog="sports-analyzer",
        description="Detect sponsor logos in sports footage and measure their exposure.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    analyze = commands.add_parser("analyze", help="detect logos in a video and save an annotated copy")
    analyze.add_argument("video", type=Path)
    analyze.add_argument("--detector", choices=["zeroshot", "finetuned"], default="zeroshot")
    analyze.add_argument(
        "--sport",
        required=True,
        choices=sorted(p.stem for p in CONFIG_DIR.glob("*.yaml")),
        help="which configs/<sport>.yaml to use for brands and detection settings",
    )
    analyze.add_argument("--weights", help="model weights (overrides config)")
    analyze.add_argument("--every", type=int, help="run detection on every Nth frame (overrides config)")
    analyze.add_argument("--conf", type=float, help="confidence threshold (overrides config)")
    analyze.add_argument("--device", help="cpu or mps (default: mps when available)")
    analyze.add_argument("--output", type=Path, help="annotated video path (default: outputs/<video>_<detector>.mp4)")

    args = parser.parse_args()
    config = yaml.safe_load((CONFIG_DIR / f"{args.sport}.yaml").read_text())
    every = args.every or config["sample_every"]
    output = args.output or Path("outputs") / f"{args.video.stem}_{args.detector}.mp4"

    settings = config[args.detector]
    weights = args.weights or settings["weights"]
    conf = args.conf if args.conf is not None else settings["conf"]
    detector: Detector
    if args.detector == "zeroshot":
        detector = ZeroShotDetector(config["brands"], weights, conf, settings["imgsz"], args.device)
    else:
        detector = FinetunedDetector(weights, conf, settings["imgsz"], args.device)
    stats = analyze_video(args.video, detector, output, every)

    print(
        f"Processed {stats.frames_processed}/{stats.frames_total} frames (every {every}) "
        f"in {stats.wall_seconds:.1f} s: {stats.frames_processed / stats.wall_seconds:.1f} frames/s, "
        f"{stats.video_seconds / stats.wall_seconds:.2f}x real time on {detector.device}"
    )
    print("Detections per brand:")
    for brand in detector.brands:
        print(f"  {brand:<12} {stats.detections[brand]}")
    print(f"Annotated video: {output}")


if __name__ == "__main__":
    main()
