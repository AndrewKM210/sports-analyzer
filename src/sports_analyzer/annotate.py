"""Drawing detections on frames."""

import numpy as np
from ultralytics.utils.plotting import Annotator, colors

from sports_analyzer.detect import Detection


def draw_detections(frame: np.ndarray, detections: list[Detection], brands: list[str]) -> np.ndarray:
    """Draw labelled boxes on a frame. Each brand gets a fixed colour based on its position in `brands`."""
    annotator = Annotator(frame)
    for d in detections:
        color = colors(brands.index(d.brand), bgr=True)
        annotator.box_label((d.x1, d.y1, d.x2, d.y2), f"{d.brand} {d.conf:.2f}", color=color)
    return annotator.result()
