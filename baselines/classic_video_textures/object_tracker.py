"""
Lucas-Kanade sparse optical flow object tracker.

Given a keyframe index and the object's bounding box on that frame,
tracks the object's position and scale across all other frames using
cv2.calcOpticalFlowPyrLK.

Strategy:
  - Sample a grid of feature points inside the object bounding box on the keyframe
  - Track those points forward (keyframe → last frame) and backward (keyframe → first frame)
  - At each frame, compute the median displacement to get a robust (dx, dy) estimate
  - Estimate scale change from the spread of tracked points
  - Return a per-frame list of (x, y, w, h) bounding boxes

If LK loses track of too many points, the last known position is held.
"""

import cv2
import numpy as np


LK_PARAMS = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)


def _sample_points(x: int, y: int, w: int, h: int, n: int = 50) -> np.ndarray:
    """Sample a grid of points inside the bounding box."""
    cols = int(np.sqrt(n * w / max(h, 1)))
    rows = max(1, n // cols)
    xs = np.linspace(x + 2, x + w - 2, cols)
    ys = np.linspace(y + 2, y + h - 2, rows)
    gx, gy = np.meshgrid(xs, ys)
    pts = np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float32)
    return pts.reshape(-1, 1, 2)


def _gray(frame_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)


def _track_direction(
    frames_bgr: list,
    start_pts: np.ndarray,
    start_idx: int,
    direction: int,          # +1 = forward, -1 = backward
    init_box: tuple,
) -> dict:
    """
    Track points from start_idx in one direction.

    Returns a dict mapping frame_index -> (x, y, w, h).
    """
    x0, y0, w0, h0 = init_box
    results = {start_idx: (x0, y0, w0, h0)}

    prev_gray = _gray(frames_bgr[start_idx])
    prev_pts = start_pts.copy()
    cur_x, cur_y, cur_w, cur_h = x0, y0, w0, h0

    indices = range(start_idx + direction, len(frames_bgr) if direction > 0 else -1, direction)

    for i in indices:
        curr_gray = _gray(frames_bgr[i])
        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, prev_pts, None, **LK_PARAMS
        )

        good_next = next_pts[status.ravel() == 1].reshape(-1, 2)
        good_prev = prev_pts[status.ravel() == 1].reshape(-1, 2)

        if len(good_next) < 4:
            # Lost track — hold last position
            results[i] = (cur_x, cur_y, cur_w, cur_h)
            prev_gray = curr_gray
            continue

        # Median displacement → robust to outlier points
        disp = good_next - good_prev            # [N, 2]
        dx = float(np.median(disp[:, 0]))
        dy = float(np.median(disp[:, 1]))

        # Scale estimate from spread of tracked points
        prev_spread = good_prev.std(axis=0).mean() + 1e-6
        next_spread = good_next.std(axis=0).mean() + 1e-6
        scale = float(np.clip(next_spread / prev_spread, 0.5, 2.0))

        cur_x = int(cur_x + dx)
        cur_y = int(cur_y + dy)
        cur_w = int(cur_w * scale)
        cur_h = int(cur_h * scale)

        results[i] = (cur_x, cur_y, cur_w, cur_h)

        # Re-sample points inside the new box for next iteration
        prev_gray = curr_gray
        prev_pts = good_next.reshape(-1, 1, 2)

        # If too few points remain, re-sample inside current box
        if len(prev_pts) < 10:
            prev_pts = _sample_points(cur_x, cur_y, cur_w, cur_h, n=50)

    return results


def track_object(
    frames_bgr: list,
    keyframe_idx: int,
    box: tuple,
    n_points: int = 50,
) -> list:
    """
    Track an object bounding box across all frames using LK optical flow.

    Args:
        frames_bgr:   list of BGR frames (uint8 numpy arrays)
        keyframe_idx: the frame where the object was manually placed
        box:          (x, y, w, h) in the keyframe
        n_points:     number of LK feature points to sample

    Returns:
        boxes: list of (x, y, w, h) for every frame index 0..N-1
               Values are integer pixel coordinates in the original frame size.
    """
    x, y, w, h = box
    init_pts = _sample_points(x, y, w, h, n=n_points)

    print(f"Tracking object from keyframe {keyframe_idx} forward...")
    fwd = _track_direction(frames_bgr, init_pts, keyframe_idx, +1, box)

    print(f"Tracking object from keyframe {keyframe_idx} backward...")
    bwd = _track_direction(frames_bgr, init_pts, keyframe_idx, -1, box)

    # Merge both directions
    merged = {**bwd, **fwd}
    merged[keyframe_idx] = box  # keyframe is exact

    # Fill any missing frames (shouldn't happen, but be safe)
    all_boxes = []
    last = box
    for i in range(len(frames_bgr)):
        if i in merged:
            last = merged[i]
        all_boxes.append(last)

    print(f"  Tracked {len(all_boxes)} frames.")
    return all_boxes


if __name__ == "__main__":
    import sys

    # Quick visual test: track a manually specified box in a video
    if len(sys.argv) < 2:
        print("Usage: python object_tracker.py <video.mp4>")
        sys.exit(1)

    cap = cv2.VideoCapture(sys.argv[1])
    frames = []
    while True:
        ret, f = cap.read()
        if not ret:
            break
        frames.append(f)
    cap.release()

    # Use the first frame and a central box as a test
    h, w = frames[0].shape[:2]
    test_box = (w // 4, h // 4, w // 2, h // 2)
    keyframe = len(frames) // 2

    boxes = track_object(frames, keyframe, test_box)

    # Visualize
    for i, (frame, box) in enumerate(zip(frames, boxes)):
        x, y, bw, bh = box
        vis = frame.copy()
        cv2.rectangle(vis, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
        cv2.imshow("Tracking", vis)
        if cv2.waitKey(30) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()
