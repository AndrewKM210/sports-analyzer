"""Fine-tuning that survives interruptions.

Ultralytics saves a full checkpoint (`weights/last.pt`: weights, optimizer, epoch, training
settings) after every epoch. `train` decides what to do with an existing run: start it, resume it
from that checkpoint, or leave it alone if it already finished. Rerunning the same command after a
crash or a Colab disconnect therefore picks up from the last completed epoch.
"""

from pathlib import Path

import torch
from ultralytics import YOLO

from sports_analyzer.detectors import default_device


def checkpoint_epoch(checkpoint: Path) -> int:
    """Last completed epoch (0-based) in a checkpoint; Ultralytics sets it to -1 once training is done."""
    return torch.load(checkpoint, map_location="cpu", weights_only=False).get("epoch", -1)


def train(
    data: str,
    model: str = "yolo11n.pt",
    epochs: int = 100,
    imgsz: int = 1280,
    batch: int = 8,
    project: str = "runs/detect",
    name: str = "train",
    device: str | None = None,
) -> Path:
    """Train a detector, or resume the run called `name` if it was interrupted. Returns the run folder.

    Resuming restores the original settings (including `epochs`), so the other arguments only
    apply to a new run.
    """
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
        print(f"{name} already finished training ({run_dir}); use a new --name to train again")
    else:
        # Calling resume on a finished run would silently start a new one, hence the check above
        print(f"Resuming {name} after epoch {epoch + 1}")
        YOLO(last).train(resume=True, device=device or default_device())
    return run_dir
