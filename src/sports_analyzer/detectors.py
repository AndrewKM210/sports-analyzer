"""Logo detectors."""

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch
from ultralytics import YOLO, YOLOWorld
from ultralytics.engine.model import Model
from ultralytics.engine.results import Results


@dataclass
class Detection:
    """One detected logo, with box corners in pixels."""

    brand: str
    conf: float
    x1: float
    y1: float
    x2: float
    y2: float


class Detector(Protocol):
    """Interface shared by all detectors."""

    brands: list[str]
    device: str
    imgsz: int
    model: Model

    def detect(self, frame: np.ndarray) -> list[Detection]: ...


def default_device() -> str:
    """Use an NVIDIA GPU (e.g. on Colab) or the Apple Silicon GPU when available, otherwise the CPU."""
    if torch.cuda.is_available():
        return "cuda"
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
        return to_detections(result, self.class_brands)


class FinetunedDetector:
    """YOLO model fine-tuned on logo images. Brand names come from the model's classes."""

    def __init__(self, weights: str, conf: float = 0.25, imgsz: int = 1280, device: str | None = None) -> None:
        self.model = YOLO(weights)
        self.brands = list(self.model.names.values())
        self.conf = conf
        self.imgsz = imgsz
        self.device = device or default_device()

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Detect logos in a BGR frame."""
        result = self.model.predict(frame, conf=self.conf, imgsz=self.imgsz, device=self.device, verbose=False)[0]
        return to_detections(result, self.brands)


def to_detections(result: Results, class_brands: list[str]) -> list[Detection]:
    """Convert an Ultralytics result to detections, mapping class indices to brand names."""
    boxes = result.boxes
    return [
        Detection(class_brands[int(cls)], conf, *xyxy)
        for xyxy, conf, cls in zip(boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist())
    ]
