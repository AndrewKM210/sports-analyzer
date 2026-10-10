"""Logo detectors."""

from dataclasses import dataclass

import numpy as np
import torch
from ultralytics import YOLOWorld


@dataclass
class Detection:
    """One detected logo, with box corners in pixels."""

    brand: str
    conf: float
    x1: float
    y1: float
    x2: float
    y2: float


def default_device() -> str:
    # TODO: extend to CUDA as well
    """Use the Apple Silicon GPU when available, otherwise the CPU."""
    return "mps" if torch.backends.mps.is_available() else "cpu"


class ZeroShotDetector:
    """YOLO-World detector driven by text prompts such as "Nike logo".

    A brand can have several prompts; their detections are all reported under the brand name.
    """

    def __init__(
        self,
        brands: dict[str, list[str]],
        weights: str = "yolov8s-worldv2.pt",
        conf: float = 0.1,
        imgsz: int = 640,
        device: str | None = None,
    ) -> None:
        self.brands = list(brands)
        self.class_brands = [brand for brand, prompts in brands.items() for _ in prompts]
        self.model = YOLOWorld(weights)
        self.model.set_classes([prompt for prompts in brands.values() for prompt in prompts])
        self.conf = conf
        self.imgsz = imgsz
        self.device = device or default_device()

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Detect logos in a BGR frame."""
        result = self.model.predict(
            frame,
            conf=self.conf,
            imgsz=self.imgsz,
            device=self.device,
            agnostic_nms=True,  # one box per logo, even when several prompts match it
            verbose=False,
        )[0]
        boxes = result.boxes
        return [
            Detection(self.class_brands[int(cls)], conf, *xyxy)
            for xyxy, conf, cls in zip(boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist())
        ]
