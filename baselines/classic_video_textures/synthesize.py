"""
Audio-conditioned video texture synthesis (CPU-compatible).

Pipeline:
    1. Load video frames (and optional audio)
    2. Compute visual distance matrix D1 (RGB pixel distance)
    3. If audio provided, compute joint D1 blending visual + MFCC audio distance
    4. Smooth with binomial filter -> D2
    5. Refine with Q-learning -> D3 / P3
    6. Sample a new frame sequence from P3
    7. Save output video and evaluation metrics

Usage:
    python synthesize.py --video path/to/video.mp4 --alpha 0.5 --out_length 10
"""

import argparse
import os
import sys

import numpy as np
import torch
import cv2
import imageio
import librosa

from computeD1 import compute_D1
from computeD2 import compute_D2
from q_learning import q_learning
from compute_joint_D1 import compute_joint_D1


# ---------------------------------------------------------------------------
# Video I/O
# ---------------------------------------------------------------------------

def load_video(path: str, max_size: int = 128, stride: int = 1) -> tuple:
    """
    Load video frames with OpenCV and audio with librosa.

    Args:
        max_size: resize frames so the larger dimension is at most this many pixels
        stride:   keep every nth frame (1 = all frames, 2 = every other, etc.)

    Returns:
        frames:       float32 tensor [N, C, H, W] in [0, 1]
        frames_full:  uint8 numpy [N, H_orig, W_orig, C] for output rendering
        audio:        float32 numpy [T] (mono), or None
        fps:          frame rate (float)
        sr:           audio sample rate (int), or 0
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Compute resize dimensions
    scale = max_size / max(orig_h, orig_w)
    small_h = int(orig_h * scale)
    small_w = int(orig_w * scale)

    frames_small = []
    frames_full  = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % stride == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames_full.append(rgb)
            small = cv2.resize(rgb, (small_w, small_h), interpolation=cv2.INTER_AREA)
            frames_small.append(small)
        idx += 1
    cap.release()

    frames_np = np.stack(frames_small, axis=0).astype(np.float32) / 255.0
    frames_t  = torch.from_numpy(frames_np).permute(0, 3, 1, 2)  # [N, C, H, W]
    frames_full_np = np.stack(frames_full, axis=0)                # [N, H, W, C] uint8

    # Try to load audio with librosa
    audio, sr = None, 0
    try:
        audio, sr = librosa.load(path, sr=None, mono=True)
        if stride > 1:
            # Subsample audio to match frame stride
            total_frames = idx
            samples_per_frame = len(audio) / total_frames
            kept = np.arange(0, total_frames, stride)
            audio_chunks = [
                audio[int(i * samples_per_frame): int((i + 1) * samples_per_frame)]
                for i in kept
            ]
            audio = np.concatenate(audio_chunks).astype(np.float32)
    except Exception:
        pass  # video has no audio track or format not supported

    return frames_t, frames_full_np, audio, fps, sr


def save_video(frames_uint8: np.ndarray, path: str, fps: float):
    """Save [N, H, W, C] uint8 array as mp4."""
    writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=7)
    for frame in frames_uint8:
        writer.append_data(frame)
    writer.close()
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def temporal_smoothness(frames: np.ndarray) -> float:
    """
    Mean squared pixel difference between consecutive frames, normalized to [0, 1].
    Lower = smoother output.
    frames: [N, H, W, C] uint8
    """
    f = frames.astype(np.float32) / 255.0
    diffs = np.mean((f[1:] - f[:-1]) ** 2)
    return float(diffs)


# ---------------------------------------------------------------------------
# Synthesis
# ---------------------------------------------------------------------------

def sample_sequence(P: torch.Tensor, n_frames: int, start: int = None) -> list:
    """
    Sample a frame index sequence by following the transition probability matrix.

    Args:
        P: transition probability matrix [N, N], rows sum to 1
        n_frames: desired output length
        start: starting frame index (random if None)

    Returns:
        list of frame indices
    """
    N = P.shape[0]
    if start is None:
        start = N // 4  # avoid edge frames
    seq = [start]
    current = start
    jump_count = 0

    while len(seq) < n_frames:
        row = P[current]
        nonzero = row.nonzero(as_tuple=False).squeeze(1)
        if len(nonzero) == 0:
            # fallback: sequential step
            current = (current + 1) % N
        else:
            probs = row[nonzero].numpy()
            probs = probs / probs.sum()
            current = int(np.random.choice(nonzero.numpy(), p=probs))

        if len(seq) > 0 and current != seq[-1] + 1:
            jump_count += 1
        seq.append(current)

    print(f"  Sampled {len(seq)} frames, {jump_count} non-sequential jumps "
          f"({100*jump_count/len(seq):.1f}%)")
    return seq


def synthesize(
    frames_full: np.ndarray,
    P: torch.Tensor,
    out_length_sec: float,
    fps: float,
) -> np.ndarray:
    """
    Build output video by reindexing frames according to the sampled sequence.

    Args:
        frames_full: [N, H, W, C] uint8 — full-resolution frames for output
        P: transition probability matrix [N, N]

    Returns:
        output: [N_out, H, W, C] uint8 numpy array
    """
    n_out = int(out_length_sec * fps)
    seq = sample_sequence(P, n_out)
    output = np.stack([frames_full[i] for i in seq])
    return output


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    print(f"\n=== Loading video: {args.video} ===")
    frames, frames_full, audio, fps, sr = load_video(
        args.video, max_size=args.max_size, stride=args.stride
    )
    N = len(frames)
    print(f"  {N} frames @ {fps:.1f} fps  "
          f"(small: {frames.shape[-2]}x{frames.shape[-1]}, "
          f"output: {frames_full.shape[1]}x{frames_full.shape[2]})")
    print(f"  audio={'yes, sr=' + str(sr) if audio is not None else 'none'}")

    print(f"\n=== Computing visual distance matrix (feats={args.feats}) ===")
    D1_visual, P1, sigma = compute_D1(
        frames, sigma_factor=args.sigma, feats=args.feats,
        slow=True, batch_size=args.batch_size
    )

    if audio is not None and args.alpha < 1.0:
        print(f"\n=== Computing joint audio-visual D1 (alpha={args.alpha}) ===")
        D1_joint, D1_audio, mfcc = compute_joint_D1(
            D1_visual, audio, sr, alpha=args.alpha, n_mfcc=args.n_mfcc
        )
    else:
        if args.alpha < 1.0 and audio is None:
            print("Warning: no audio found, falling back to visual-only (alpha=1.0)")
        D1_joint = D1_visual

    print(f"\n=== Computing D2 (filter_size={args.filter_size}) ===")
    D2, P2, sigma2, _ = compute_D2(D1_joint, sigma_factor=args.sigma,
                                   filter_size=args.filter_size)

    print(f"\n=== Q-learning (threshold={args.threshold}) ===")
    D3, P3, P3_thresh, sigma3 = q_learning(
        D2, sigma_factor=args.sigma, thresholding=args.threshold,
        max_iters=args.max_iters
    )

    print(f"\n=== Synthesizing {args.out_length}s video ===")
    output_frames = synthesize(frames_full, P3_thresh, args.out_length, fps)

    # Metrics
    smooth = temporal_smoothness(output_frames)
    print(f"\n  Temporal smoothness (MSE): {smooth:.6f}")

    # Save
    os.makedirs(args.out_dir, exist_ok=True)
    video_name = os.path.splitext(os.path.basename(args.video))[0]
    out_path = os.path.join(
        args.out_dir,
        f"{video_name}_alpha{args.alpha:.2f}_sigma{args.sigma}_fs{args.filter_size}.mp4"
    )
    save_video(output_frames, out_path, fps)
    print(f"\nDone. Temporal smoothness={smooth:.6f}")
    return smooth


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audio-conditioned video texture synthesis")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Visual weight (1.0=visual only, 0.0=audio only)")
    parser.add_argument("--feats", default="RGB", choices=["RGB", "ResNet"],
                        help="Visual feature type")
    parser.add_argument("--sigma", type=float, default=4.5,
                        help="Gaussian bandwidth scale factor")
    parser.add_argument("--filter_size", type=int, default=16,
                        help="Binomial smoothing filter size")
    parser.add_argument("--threshold", type=float, default=0.75,
                        help="Q-learning transition probability threshold")
    parser.add_argument("--out_length", type=float, default=10.0,
                        help="Output video length in seconds")
    parser.add_argument("--n_mfcc", type=int, default=20,
                        help="Number of MFCC coefficients for audio features")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for distance matrix computation")
    parser.add_argument("--max_iters", type=int, default=30,
                        help="Max Q-learning iterations")
    parser.add_argument("--max_size", type=int, default=128,
                        help="Resize frames to at most this dimension for distance computation")
    parser.add_argument("--stride", type=int, default=2,
                        help="Keep every nth frame (reduces N for large videos)")
    parser.add_argument("--out_dir", default="results", help="Output directory")

    args = parser.parse_args()
    main(args)
