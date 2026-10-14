"""Fine-tuning that survives interruptions.

Ultralytics saves a full checkpoint (`weights/last.pt`: weights, optimizer, epoch, training
settings) after every epoch. `train` decides what to do with an existing run: start it, resume it
from that checkpoint, or leave it alone if it already finished. Rerunning the same command after a
crash or a Colab disconnect therefore picks up from the last completed epoch.

Runs are named after what went into them (dataset version, model, image size), so changing the
dataset starts a new run while the same settings resume the existing one.
"""

from pathlib import Path

import torch
from ultralytics import YOLO

from sports_analyzer.detectors import default_device


def checkpoint_epoch(checkpoint: Path) -> int:
    """Last completed epoch (0-based) in a checkpoint; Ultralytics sets it to -1 once training is done."""
    return torch.load(checkpoint, map_location="cpu", weights_only=False).get("epoch", -1)


def run_name(data: str, model: str, imgsz: int) -> str:
    """Default run name, e.g. data/datasets/f1_v01/data.yaml + yolo11s.pt + 1280 -> f1_v01_yolo11s_1280."""
    return f"{Path(data).parent.name}_{Path(model).stem}_{imgsz}"


def train(
    data: str,
    model: str = "yolo11n.pt",
    epochs: int = 100,
    imgsz: int = 1280,
    batch: int = 8,
    project: str = "runs/train",
    name: str | None = None,
    device: str | None = None,
) -> Path:
    """Train a detector, or resume the run if it was interrupted. Returns the run folder.

    Resuming restores the original settings (including `epochs`), so the other arguments only
    apply to a new run.
    """
    name = name or run_name(data, model, imgsz)
    run_dir = Path(project).resolve() / name
    last = run_dir / "weights" / "last.pt"
    if not last.exists():
        YOLO(model).train(
            data=data,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            cache="ram",
            device=device or default_device(),
            project=str(run_dir.parent),
            name=name,
            exist_ok=True,  # the folder can exist without last.pt if a run died during its first epoch
        )
    elif (epoch := checkpoint_epoch(last)) == -1:
        print(f"{name} already finished training ({run_dir}); change the settings or --name to train again")
    else:
        # Calling resume on a finished run would silently start a new one, hence the check above
        print(f"Resuming {name} after epoch {epoch + 1}")
        YOLO(last).train(resume=True, device=device or default_device())
    return run_dir
