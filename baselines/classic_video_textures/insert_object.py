"""
Loop-aware object insertion into looping video textures.

Full pipeline:
  1. Load video frames
  2. Interactive keyframe placement (OpenCV window)
  3. LK optical flow tracking across all source frames
  4. Classic video texture synthesis (D1 → D2 → Q-learning → P)
  5. Composite object at tracked position for every synthesized frame
  6. Save two outputs for comparison:
       *_tracked.mp4  — object tracked per-frame (our method)
       *_naive.mp4    — object pinned at keyframe position (baseline)

Usage:
    python insert_object.py --video ../../videos/vtfishtk.mpg --object hat.png
    python insert_object.py --video vid.mp4 --object logo.png --keyframe 30 --out_length 8
"""

import argparse
import os
import sys

import cv2
import numpy as np
import torch
import imageio

from place_object import place_object_interactive
from object_tracker import track_object
from composite import load_object, composite_sequence
from computeD1 import compute_D1
from computeD2 import compute_D2
from q_learning import q_learning


# ---------------------------------------------------------------------------
# Video I/O
# ---------------------------------------------------------------------------

def load_frames(path: str, stride: int = 1) -> tuple:
    """
    Load all frames from a video using imageio+ffmpeg (handles .mpg/.avi/.mp4).

    Returns:
        frames_bgr: list of [H, W, 3] uint8 BGR numpy arrays (full resolution)
        fps:        float
    """
    import imageio
    reader = imageio.get_reader(path)
    meta = reader.get_meta_data()
    fps = float(meta.get("fps", 30.0))

    frames = []
    for idx, frame_rgb in enumerate(reader):
        if idx % stride == 0:
            # imageio gives RGB; convert to BGR for OpenCV compatibility
            frames.append(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
    reader.close()

    if len(frames) == 0:
        raise IOError(f"No frames read from {path}. Check the file path and format.")

    print(f"Loaded {len(frames)} frames @ {fps:.1f} fps  (stride={stride})")
    return frames, fps


def frames_to_tensor(frames_bgr: list, max_size: int = 128) -> torch.Tensor:
    """
    Resize frames and convert to float tensor [N, C, H, W] in [0,1]
    for distance matrix computation. Resize first to minimise peak memory.
    """
    h0, w0 = frames_bgr[0].shape[:2]
    scale = max_size / max(h0, w0)
    sh, sw = max(1, int(h0 * scale)), max(1, int(w0 * scale))
    small = []
    for f in frames_bgr:
        # Resize while still BGR (tiny allocation), then convert colour
        s = cv2.resize(f, (sw, sh), interpolation=cv2.INTER_AREA)
        s = cv2.cvtColor(s, cv2.COLOR_BGR2RGB)
        small.append(s)
    arr = np.stack(small).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(0, 3, 1, 2)


def save_video(frames_bgr: list, path: str, fps: float):
    """Save list of BGR frames as mp4."""
    writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=7)
    for f in frames_bgr:
        writer.append_data(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    writer.close()
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def temporal_smoothness(frames_bgr: list) -> float:
    """Mean squared pixel difference between consecutive frames, normalized to [0,1]."""
    diffs = []
    for a, b in zip(frames_bgr, frames_bgr[1:]):
        fa = a.astype(np.float32) / 255.0
        fb = b.astype(np.float32) / 255.0
        diffs.append(np.mean((fa - fb) ** 2))
    return float(np.mean(diffs)) if diffs else 0.0


def alignment_error(synth_boxes: list, tracked_boxes_all: list, naive_box: tuple) -> dict:
    """
    Primary metric: how far is the placed hat from where the tracker says
    the character actually is at each synthesized frame.

    For the tracked method this is always 0 (it uses the tracked positions).
    For the naive method it grows as the character moves away from the keyframe.

    Args:
        synth_boxes:       list of (x,y,w,h) — where tracked method places the hat
                           (one per synthesized frame, already in source-frame order)
        tracked_boxes_all: all_boxes[i] — ground-truth tracked position for source frame i
        naive_box:         the fixed keyframe box used by the naive method

    Returns dict with:
        naive_mean_err:    mean pixel displacement of naive hat from tracked position
        naive_at_jumps:    mean error only at non-sequential transitions
        naive_sequential:  mean error at sequential (non-jump) transitions
    """
    cx_naive, cy_naive = naive_box[0] + naive_box[2]//2, naive_box[1] + naive_box[3]//2

    errors = []
    for box in synth_boxes:
        cx_t = box[0] + box[2] // 2
        cy_t = box[1] + box[3] // 2
        errors.append(np.sqrt((cx_naive - cx_t)**2 + (cy_naive - cy_t)**2))

    return {
        "naive_mean_err_px": float(np.mean(errors)),
        "naive_max_err_px":  float(np.max(errors)),
    }


def jump_displacement(seq: list, synth_boxes: list) -> dict:
    """
    At non-sequential transitions (jumps), measure the hat position change
    for tracked vs. naive.

    Tracked method: hat jumps to the correct position for the destination frame.
    Naive method:   hat stays fixed — displacement = 0, but alignment is lost.

    The key insight: at a jump from frame A to frame B, the character has moved.
    Tracked correctly repositions the hat; naive does not.

    Returns displacements in pixels for jump frames and sequential frames separately.
    """
    jump_disps_tracked = []
    seq_disps_tracked  = []

    for t in range(len(seq) - 1):
        bx, by, bw, bh = synth_boxes[t]
        nx, ny, nw, nh = synth_boxes[t + 1]
        disp = np.sqrt((bx + bw//2 - nx - nw//2)**2 + (by + bh//2 - ny - nh//2)**2)

        if seq[t + 1] != seq[t] + 1:
            jump_disps_tracked.append(disp)
        else:
            seq_disps_tracked.append(disp)

    return {
        "jump_disp_tracked_px": float(np.mean(jump_disps_tracked)) if jump_disps_tracked else 0.0,
        "seq_disp_tracked_px":  float(np.mean(seq_disps_tracked))  if seq_disps_tracked  else 0.0,
        "n_jumps": len(jump_disps_tracked),
    }


# ---------------------------------------------------------------------------
# Synthesis
# ---------------------------------------------------------------------------

def build_transition_matrix(
    frames_tensor: torch.Tensor,
    sigma: float,
    filter_size: int,
    threshold: float,
    max_iters: int,
    batch_size: int,
) -> torch.Tensor:
    """Run D1 → D2 → Q-learning and return the thresholded transition matrix P."""
    print("\n[1/3] Computing visual distance matrix...")
    D1, _, _ = compute_D1(frames_tensor, sigma_factor=sigma,
                           feats="RGB", slow=True, batch_size=batch_size)

    print("[2/3] Binomial smoothing (D2)...")
    D2, _, _, _ = compute_D2(D1, sigma_factor=sigma, filter_size=filter_size)

    print("[3/3] Q-learning...")
    _, _, P, _ = q_learning(D2, sigma_factor=sigma,
                             thresholding=threshold, max_iters=max_iters)
    return P


def sample_sequence(P: torch.Tensor, n_out: int) -> list:
    """Sample a frame index sequence from the transition probability matrix."""
    N = P.shape[0]
    start = N // 4
    seq = [start]
    current = start
    jumps = 0
    while len(seq) < n_out:
        row = P[current]
        nonzero = row.nonzero(as_tuple=False).squeeze(1)
        if len(nonzero) == 0:
            current = (current + 1) % N
        else:
            probs = row[nonzero].numpy()
            probs /= probs.sum()
            current = int(np.random.choice(nonzero.numpy(), p=probs))
        if current != seq[-1] + 1:
            jumps += 1
        seq.append(current)
    print(f"  Sequence: {len(seq)} frames, {jumps} jumps ({100*jumps/len(seq):.1f}%)")
    return seq


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    # ── Load video ────────────────────────────────────────────────────────
    frames_bgr, fps = load_frames(args.video, stride=args.stride)
    N = len(frames_bgr)

    # ── Load object PNG ───────────────────────────────────────────────────
    obj_bgra = load_object(args.object)
    print(f"Object: {obj_bgra.shape[1]}x{obj_bgra.shape[0]} px  ({args.object})")

    # ── Choose keyframe ───────────────────────────────────────────────────
    keyframe_idx = args.keyframe if args.keyframe is not None else N // 4
    keyframe_idx = min(keyframe_idx, N - 1)
    print(f"Keyframe: {keyframe_idx}")

    # ── Interactive placement ─────────────────────────────────────────────
    print("\nOpening placement window...")
    result = place_object_interactive(frames_bgr[keyframe_idx], obj_bgra)
    if result is None:
        print("Placement cancelled. Exiting.")
        sys.exit(0)
    x, y, w, h = result
    keyframe_box = (x, y, w, h)

    # ── Track object across all frames ────────────────────────────────────
    print("\nTracking object across all frames...")
    all_boxes = track_object(frames_bgr, keyframe_idx, keyframe_box)

    # ── Build transition matrix (on small frames) ─────────────────────────
    print("\nBuilding transition matrix...")
    frames_tensor = frames_to_tensor(frames_bgr, max_size=args.max_size)
    P = build_transition_matrix(
        frames_tensor,
        sigma=args.sigma,
        filter_size=args.filter_size,
        threshold=args.threshold,
        max_iters=args.max_iters,
        batch_size=args.batch_size,
    )

    # ── Sample synthesized frame sequence ─────────────────────────────────
    n_out = int(args.out_length * fps)
    print(f"\nSampling {n_out} frames ({args.out_length}s)...")
    seq = sample_sequence(P, n_out)

    # ── Reindex frames and boxes using the synthesized sequence ───────────
    synth_frames  = [frames_bgr[i] for i in seq]
    synth_boxes   = [all_boxes[i]  for i in seq]
    naive_box     = keyframe_box   # pinned at keyframe position for baseline

    # ── Composite: tracked method ─────────────────────────────────────────
    print("\nCompositing (tracked)...")
    tracked_frames = composite_sequence(synth_frames, obj_bgra, synth_boxes)

    # ── Composite: naive baseline (same box every frame) ──────────────────
    print("Compositing (naive)...")
    naive_frames = composite_sequence(synth_frames, obj_bgra,
                                      [naive_box] * len(synth_frames))

    # ── Save outputs ──────────────────────────────────────────────────────
    os.makedirs(args.out_dir, exist_ok=True)
    video_name = os.path.splitext(os.path.basename(args.video))[0]
    obj_name   = os.path.splitext(os.path.basename(args.object))[0]
    base       = os.path.join(args.out_dir, f"{video_name}_{obj_name}")

    save_video(tracked_frames, base + "_tracked.mp4", fps)
    save_video(naive_frames,   base + "_naive.mp4",   fps)

    # Also save the raw synthesized video (no object) for reference
    save_video(synth_frames, base + "_synthesized.mp4", fps)

    # ── Metrics ───────────────────────────────────────────────────────────
    ts_tracked = temporal_smoothness(tracked_frames)
    ts_naive   = temporal_smoothness(naive_frames)
    ts_synth   = temporal_smoothness(synth_frames)

    jump_count = sum(1 for a, b in zip(seq[:-1], seq[1:]) if b != a + 1)
    jump_pct   = 100.0 * jump_count / max(len(seq) - 1, 1)

    # Alignment error: how far naive hat is from tracked (correct) position
    align = alignment_error(synth_boxes, all_boxes, naive_box)

    # Per-transition displacement for tracked hat at jump vs. sequential frames
    jd = jump_displacement(seq, synth_boxes)

    print(f"\n{'='*50}")
    print(f"Results saved to: {args.out_dir}/")
    print(f"  {video_name}_{obj_name}_tracked.mp4     ← our method")
    print(f"  {video_name}_{obj_name}_naive.mp4       ← baseline")
    print(f"  {video_name}_{obj_name}_synthesized.mp4 ← no object")
    print(f"\nSynthesis: {len(seq)} frames, {jump_count} jumps ({jump_pct:.1f}%)")
    print(f"\nWhole-frame temporal smoothness (MSE ↓):")
    print(f"  No object (reference) : {ts_synth:.6f}")
    print(f"  Tracked insertion     : {ts_tracked:.6f}")
    print(f"  Naive insertion       : {ts_naive:.6f}")
    print(f"\nObject alignment error — naive hat vs. tracked (correct) position (px ↓):")
    print(f"  Mean error across all frames : {align['naive_mean_err_px']:.1f} px")
    print(f"  Max  error across all frames : {align['naive_max_err_px']:.1f} px")
    print(f"  (Tracked method error        : 0.0 px by construction)")
    if jd["n_jumps"] > 0:
        print(f"\nHat displacement at transitions (tracked method) (px):")
        print(f"  At jumps       ({jd['n_jumps']} events) : {jd['jump_disp_tracked_px']:.1f} px")
        print(f"  Sequential frames            : {jd['seq_disp_tracked_px']:.1f} px")
    print(f"{'='*50}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Loop-aware object insertion via classic video textures"
    )
    parser.add_argument("--video",   required=True,  help="Path to input video")
    parser.add_argument("--object",  required=True,  help="Path to PNG object (with alpha)")
    parser.add_argument("--keyframe", type=int, default=None,
                        help="Keyframe index for manual placement (default: N//4)")
    parser.add_argument("--out_length", type=float, default=8.0,
                        help="Output video length in seconds")
    parser.add_argument("--sigma",      type=float, default=4.5)
    parser.add_argument("--filter_size", type=int,  default=8)
    parser.add_argument("--threshold",  type=float, default=0.75)
    parser.add_argument("--max_iters",  type=int,   default=15)
    parser.add_argument("--max_size",   type=int,   default=128,
                        help="Resize dimension for distance matrix computation")
    parser.add_argument("--stride",     type=int,   default=2,
                        help="Keep every nth frame (reduces N for long videos)")
    parser.add_argument("--batch_size", type=int,   default=64)
    parser.add_argument("--out_dir",    default="results")

    args = parser.parse_args()
    main(args)
