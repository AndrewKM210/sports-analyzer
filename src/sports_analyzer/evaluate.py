"""Scoring a detector on a dataset's validation split."""

from pathlib import Path

import yaml

from sports_analyzer.detectors import Detector


def evaluate(detector: Detector, data: str, name: str, project: str = "runs/val") -> Path:
    """Print precision, recall and mAP per brand; plots go to <project>/<dataset>/<name>. Returns that folder.

    Ultralytics matches boxes by class index, so the detector's classes must be the dataset's
    brands in the same order.
    """
    names = list(yaml.safe_load(Path(data).read_text())["names"].values())
    classes = getattr(detector, "class_brands", detector.brands)  # zero-shot: one class per prompt
    if classes != names:
        raise ValueError(f"Detector classes {classes} don't match the dataset's {names}")

    dataset_dir = Path(project).resolve() / Path(data).parent.name
    detector.model.val(
        data=data,
        imgsz=detector.imgsz,
        device=detector.device,
        project=str(dataset_dir),
        name=name,
        exist_ok=True,  # scoring the same model on the same data gives the same result
    )
    return dataset_dir / name
