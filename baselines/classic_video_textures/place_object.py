"""
Interactive keyframe placement tool.

Opens an OpenCV window showing the chosen keyframe with the PNG object
overlaid at the cursor position. The user:
  - Moves the mouse to position the object
  - Scrolls the mouse wheel (or presses +/-) to resize
  - Left-clicks to confirm placement
  - Presses Q or Escape to cancel

Returns (x, y, w, h) — top-left corner and size of the placed object
in the original full-resolution frame.
"""

import cv2
import numpy as np


def _overlay(background: np.ndarray, obj_rgba: np.ndarray, x: int, y: int) -> np.ndarray:
    """Alpha-composite obj_rgba onto a copy of background at position (x, y)."""
    out = background.copy()
    oh, ow = obj_rgba.shape[:2]
    bh, bw = background.shape[:2]

    # Clip to frame bounds
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + ow, bw), min(y + oh, bh)
    ox0, oy0 = x0 - x, y0 - y
    ox1, oy1 = ox0 + (x1 - x0), oy0 + (y1 - y0)

    if x1 <= x0 or y1 <= y0:
        return out

    alpha = obj_rgba[oy0:oy1, ox0:ox1, 3:4].astype(np.float32) / 255.0
    obj_rgb = obj_rgba[oy0:oy1, ox0:ox1, :3].astype(np.float32)
    bg_patch = out[y0:y1, x0:x1].astype(np.float32)

    blended = alpha * obj_rgb + (1.0 - alpha) * bg_patch
    out[y0:y1, x0:x1] = blended.astype(np.uint8)
    return out


def place_object_interactive(
    frame_bgr: np.ndarray,
    obj_rgba: np.ndarray,
    window_title: str = "Place object — scroll to resize, click to confirm, Q to cancel",
) -> tuple:
    """
    Show an interactive window and let the user place the object.

    Args:
        frame_bgr: the keyframe in BGR (OpenCV format), shape [H, W, 3]
        obj_rgba:  the object image with alpha channel, shape [H, W, 4]

    Returns:
        (x, y, w, h) top-left and size in frame pixel coordinates,
        or None if the user cancelled.
    """
    state = {
        "mx": frame_bgr.shape[1] // 2,
        "my": frame_bgr.shape[0] // 2,
        "scale": 1.0,
        "confirmed": False,
        "cancelled": False,
    }

    orig_h, orig_w = obj_rgba.shape[:2]

    def get_scaled_obj():
        w = max(4, int(orig_w * state["scale"]))
        h = max(4, int(orig_h * state["scale"]))
        return cv2.resize(obj_rgba, (w, h), interpolation=cv2.INTER_AREA)

    def redraw():
        obj = get_scaled_obj()
        oh, ow = obj.shape[:2]
        x = state["mx"] - ow // 2
        y = state["my"] - oh // 2
        preview = _overlay(frame_bgr, obj, x, y)

        # Draw crosshair at object center
        cx, cy = state["mx"], state["my"]
        cv2.drawMarker(preview, (cx, cy), (0, 255, 0),
                       cv2.MARKER_CROSS, 20, 2, cv2.LINE_AA)

        # HUD
        info = f"Size: {ow}x{oh}  |  Pos: ({cx},{cy})  |  Scroll to resize  |  Click to place"
        cv2.putText(preview, info, (10, preview.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
        cv2.imshow(window_title, preview)

    def on_mouse(event, x, y, flags, param):
        state["mx"] = x
        state["my"] = y

        if event == cv2.EVENT_LBUTTONDOWN:
            state["confirmed"] = True

        elif event == cv2.EVENT_MOUSEWHEEL:
            if flags > 0:
                state["scale"] = min(state["scale"] * 1.1, 10.0)
            else:
                state["scale"] = max(state["scale"] / 1.1, 0.05)

        redraw()

    cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window_title, on_mouse)
    redraw()

    print("\nPlacement window open.")
    print("  Move mouse to position the object.")
    print("  Scroll wheel to resize.")
    print("  Left-click to confirm.")
    print("  Press Q or Escape to cancel.\n")

    while True:
        key = cv2.waitKey(20) & 0xFF

        if state["confirmed"]:
            break
        if state["cancelled"] or key in (ord("q"), ord("Q"), 27):  # 27 = Escape
            state["confirmed"] = False
            break
        if key in (ord("+"), ord("=")):
            state["scale"] = min(state["scale"] * 1.1, 10.0)
            redraw()
        if key in (ord("-"), ord("_")):
            state["scale"] = max(state["scale"] / 1.1, 0.05)
            redraw()

    cv2.destroyAllWindows()

    if not state["confirmed"]:
        print("Placement cancelled.")
        return None

    obj = get_scaled_obj()
    ow, oh = obj.shape[1], obj.shape[0]
    x = state["mx"] - ow // 2
    y = state["my"] - oh // 2
    print(f"Object placed at ({x}, {y}), size {ow}x{oh}")
    return x, y, ow, oh


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python place_object.py <video.mp4> <object.png> [keyframe_index]")
        sys.exit(1)

    video_path = sys.argv[1]
    obj_path = sys.argv[2]
    keyframe_idx = int(sys.argv[3]) if len(sys.argv) > 3 else 0

    import imageio
    reader = imageio.get_reader(video_path)
    frame = None
    for i, f in enumerate(reader):
        if i == keyframe_idx:
            frame = cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
            break
    reader.close()

    if frame is None:
        print(f"Could not read frame {keyframe_idx} from {video_path}")
        sys.exit(1)

    obj = cv2.imread(obj_path, cv2.IMREAD_UNCHANGED)
    if obj is None:
        print(f"Could not load object image: {obj_path}")
        sys.exit(1)
    if obj.shape[2] == 3:
        # No alpha channel — add fully opaque alpha
        alpha = np.full((*obj.shape[:2], 1), 255, dtype=np.uint8)
        obj = np.concatenate([obj, alpha], axis=2)

    result = place_object_interactive(frame, obj)
    if result:
        print(f"Result: x={result[0]}, y={result[1]}, w={result[2]}, h={result[3]}")
