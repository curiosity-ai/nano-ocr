"""The page-type model's classes and input, shared by training, evaluation and the layer dumps.

The C# side is NanoOcr.Pipeline.PageTypeModel.Input; the two must stay identical. OpenCV's
INTER_AREA is what NanoOCR's CvResize reproduces to the byte, and Python's round() is Py.Round.
"""

import cv2
import numpy as np

CLASSES = ["printed", "table", "handwritten", "photo"]
SIDE = 512


def letterbox(gray: np.ndarray, side: int = SIDE) -> np.ndarray:
    """The grayscale page, long side scaled to `side`, top-left on a white square (uint8)."""
    h, w = gray.shape
    s = side / max(h, w)
    sw, sh = max(1, round(w * s)), max(1, round(h * s))
    small = cv2.resize(gray, (sw, sh), interpolation=cv2.INTER_AREA)
    out = np.full((side, side), 255, np.uint8)
    out[: min(sh, side), : min(sw, side)] = small[:side, :side]
    return out


def model_input(gray: np.ndarray, side: int = SIDE) -> np.ndarray:
    """The model's `x`: `[1, 1, side, side]` float32, gray / 255."""
    return (letterbox(gray, side).astype(np.float32) / 255.0)[None, None]
