"""Command-line entry point: `sports-analyzer analyze | train | evaluate ...`."""

import argparse
from pathlib import Path

import yaml

from sports_analyzer.detectors import Detector, FinetunedDetector, ZeroShotDetector
from sports_analyzer.evaluate import evaluate
from sports_analyzer.pipeline import analyze_video
from sports_analyzer.train import train

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
    add_detector_args(analyze)
    analyze.add_argument("--conf", type=float, help="confidence threshold (overrides config)")
    analyze.add_argument("--every", type=int, help="run detection on every Nth frame (overrides config)")
    analyze.add_argument("--output", type=Path, help="annotated video path (default: outputs/<video>_<detector>.mp4)")

    fit = commands.add_parser("train", help="fine-tune a detector, resuming the run if it was interrupted")
    fit.add_argument("data", help="data.yaml of a dataset made by scripts/prepare_dataset.py")
    fit.add_argument("--model", default="yolo11n.pt", help="pretrained model to start from")
    fit.add_argument("--epochs", type=int, default=100)
    fit.add_argument("--imgsz", type=int, default=1280)
    fit.add_argument("--batch", type=int, default=8, help="-1 picks the largest batch that fits (CUDA only)")
    fit.add_argument("--name", help="run name (default: <dataset>_<model>_<imgsz>); an unfinished run is resumed")
    fit.add_argument("--project", default="runs/train", help="folder for training runs")
    fit.add_argument("--device", help="cuda, mps or cpu (default: the best available)")

    score = commands.add_parser("evaluate", help="precision, recall and mAP per brand on a dataset's val split")
    score.add_argument("data", help="data.yaml of a dataset made by scripts/prepare_dataset.py")
    add_detector_args(score)
    score.add_argument("--project", default="runs/val", help="folder for evaluation runs")

    args = parser.parse_args()
    if args.command == "analyze":
        run_analyze(args)
    elif args.command == "train":
        run_dir = train(args.data, args.model, args.epochs, args.imgsz, args.batch, args.project, args.name, args.device)
        print(f"Weights: {run_dir / 'weights' / 'best.pt'}")
    else:
        detector = load_detector(args)
        print(f"Results: {evaluate(detector, args.data, eval_name(args, detector), args.project)}")


def add_detector_args(parser: argparse.ArgumentParser) -> None:
    """Options for choosing a detector and its settings from configs/<sport>.yaml."""
    parser.add_argument(
        "--sport",
        required=True,
        choices=sorted(p.stem for p in CONFIG_DIR.glob("*.yaml")),
        help="which configs/<sport>.yaml to use for brands and detection settings",
    )
    parser.add_argument("--detector", choices=["zeroshot", "finetuned"], default="zeroshot")
    parser.add_argument("--weights", help="model weights (overrides config)")
    parser.add_argument("--device", help="cuda, mps or cpu (default: the best available)")


def load_detector(args: argparse.Namespace) -> Detector:
    """Build the detector chosen on the command line, with settings from the sport's config."""
    config = yaml.safe_load((CONFIG_DIR / f"{args.sport}.yaml").read_text())
    settings = config[args.detector]
    weights = args.weights or settings["weights"]
    conf = getattr(args, "conf", None)
    conf = conf if conf is not None else settings["conf"]
    if args.detector == "zeroshot":
        return ZeroShotDetector(config["brands"], weights, conf, settings["imgsz"], args.device)
    return FinetunedDetector(weights, conf, settings["imgsz"], args.device)


def eval_name(args: argparse.Namespace, detector: Detector) -> str:
    """Name an evaluation after the model: its training run for fine-tuned weights, e.g. f1_v01_yolo11s_1280."""
    config = yaml.safe_load((CONFIG_DIR / f"{args.sport}.yaml").read_text())
    weights = Path(args.weights or config[args.detector]["weights"])
    if args.detector == "zeroshot":
        return f"zeroshot_{weights.stem}_{detector.imgsz}"
    return weights.parent.parent.name if weights.parent.name == "weights" else weights.stem


def run_analyze(args: argparse.Namespace) -> None:
    """Run a detector over a video and print throughput and detections per brand."""
    config = yaml.safe_load((CONFIG_DIR / f"{args.sport}.yaml").read_text())
    every = args.every or config["sample_every"]
    output = args.output or Path("outputs") / f"{args.video.stem}_{args.detector}.mp4"
    detector = load_detector(args)
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
