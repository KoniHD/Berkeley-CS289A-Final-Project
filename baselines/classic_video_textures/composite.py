"""
Alpha compositing utilities.

Composites a PNG object (RGBA) onto a BGR frame at a given position and size.
"""

import cv2
import numpy as np


def load_object(path: str) -> np.ndarray:
    """
    Load a PNG with alpha channel.

    Returns RGBA uint8 array [H, W, 4].
    If the image has no alpha channel, a fully opaque one is added.
    """
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not load object image: {path}")

    if img.ndim == 2:
        # Grayscale — convert to BGRA
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    elif img.shape[2] == 3:
        alpha = np.full((*img.shape[:2], 1), 255, dtype=np.uint8)
        img = np.concatenate([img, alpha], axis=2)
    # img is now BGRA (OpenCV convention)
    return img


def composite_onto_frame(
    frame_bgr: np.ndarray,
    obj_bgra: np.ndarray,
    box: tuple,
) -> np.ndarray:
    """
    Alpha-composite obj_bgra onto frame_bgr at the given bounding box.

    Args:
        frame_bgr: [H, W, 3] uint8 BGR frame
        obj_bgra:  [Oh, Ow, 4] uint8 BGRA object (OpenCV channel order)
        box:       (x, y, w, h) — where to place the object on the frame

    Returns:
        Composited frame [H, W, 3] uint8 BGR.
    """
    x, y, w, h = box
    if w <= 0 or h <= 0:
        return frame_bgr.copy()

    # Resize object to fit the tracked bounding box
    obj_resized = cv2.resize(obj_bgra, (w, h), interpolation=cv2.INTER_AREA)

    out = frame_bgr.copy()
    fh, fw = frame_bgr.shape[:2]

    # Clamp to frame bounds
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, fw), min(y + h, fh)
    ox0, oy0 = x0 - x, y0 - y
    ox1, oy1 = ox0 + (x1 - x0), oy0 + (y1 - y0)

    if x1 <= x0 or y1 <= y0:
        return out

    alpha = obj_resized[oy0:oy1, ox0:ox1, 3:4].astype(np.float32) / 255.0
    obj_patch = obj_resized[oy0:oy1, ox0:ox1, :3].astype(np.float32)
    bg_patch = out[y0:y1, x0:x1].astype(np.float32)

    blended = alpha * obj_patch + (1.0 - alpha) * bg_patch
    out[y0:y1, x0:x1] = blended.astype(np.uint8)
    return out


def composite_sequence(
    frames_bgr: list,
    obj_bgra: np.ndarray,
    boxes: list,
) -> list:
    """
    Composite the object onto every frame using the corresponding tracked box.

    Args:
        frames_bgr: list of [H, W, 3] uint8 BGR frames
        obj_bgra:   [Oh, Ow, 4] uint8 BGRA object
        boxes:      list of (x, y, w, h), one per frame

    Returns:
        List of composited BGR frames.
    """
    result = []
    for frame, box in zip(frames_bgr, boxes):
        result.append(composite_onto_frame(frame, obj_bgra, box))
    return result
