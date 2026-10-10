"""Run a detector over a video and write an annotated copy."""

import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import cv2

from sports_analyzer.annotate import draw_detections
from sports_analyzer.detect import ZeroShotDetector


@dataclass
class RunStats:
    """Summary of one run, used to report throughput."""

    frames_total: int
    frames_processed: int
    video_seconds: float
    wall_seconds: float
    detections: Counter[str]


def analyze_video(video: Path, detector: ZeroShotDetector, output: Path, sample_every: int = 3) -> RunStats:
    """Detect logos on every `sample_every`-th frame and save those frames, annotated, to `output`.

    The output video keeps the original duration at a frame rate of fps / sample_every.
    """
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"avc1"), fps / sample_every, size)

    counts: Counter[str] = Counter()
    frame_idx = processed = 0
    start = time.perf_counter()
    # grab() skips decoding; frames are only decoded (retrieve) when they're sampled
    while cap.grab():
        if frame_idx % sample_every == 0:
            _, frame = cap.retrieve()
            detections = detector.detect(frame)
            counts.update(d.brand for d in detections)
            writer.write(draw_detections(frame, detections, detector.brands))
            processed += 1
            print(f"\rFrame {frame_idx}/{total}", end="", flush=True)
        frame_idx += 1
    print()

    cap.release()
    writer.release()
    return RunStats(frame_idx, processed, frame_idx / fps, time.perf_counter() - start, counts)
