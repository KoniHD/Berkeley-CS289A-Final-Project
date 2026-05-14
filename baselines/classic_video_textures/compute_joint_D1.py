"""
Joint audio-visual distance matrix for audio-conditioned video textures.

Computes per-frame MFCC features from audio, then blends the visual
distance matrix D1_visual with an audio distance matrix D1_audio:

    D1_joint = alpha * D1_visual_norm + (1 - alpha) * D1_audio_norm

alpha=1.0 reproduces the purely visual baseline.
alpha=0.0 selects transitions purely by audio similarity.
"""

import numpy as np
import torch
import torch.nn.functional as F
import librosa


def extract_mfcc_per_frame(
    audio: np.ndarray,
    sr: int,
    n_frames: int,
    n_mfcc: int = 20,
) -> torch.Tensor:
    """
    Extract one MFCC vector per video frame.

    Splits the audio signal into n_frames equal-length segments and
    computes the mean MFCC across each segment.

    Args:
        audio: 1-D audio waveform, shape [T]
        sr: sample rate
        n_frames: number of video frames (determines segment length)
        n_mfcc: number of MFCC coefficients

    Returns:
        mfcc_per_frame: tensor of shape [n_frames, n_mfcc]
    """
    samples_per_frame = len(audio) // n_frames
    mfcc_list = []

    for i in range(n_frames):
        start = i * samples_per_frame
        end = start + samples_per_frame
        segment = audio[start:end].astype(np.float32)

        if len(segment) < 512:
            # pad short segments
            segment = np.pad(segment, (0, 512 - len(segment)))

        mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=n_mfcc)
        mfcc_list.append(mfcc.mean(axis=1))  # [n_mfcc]

    return torch.tensor(np.stack(mfcc_list), dtype=torch.float32)  # [N, n_mfcc]


def compute_audio_D1(mfcc: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise L2 distance between per-frame MFCC vectors.

    Args:
        mfcc: [N, n_mfcc]

    Returns:
        D1_audio: [N, N]
    """
    mfcc_norm = F.normalize(mfcc, dim=1)
    # cosine distance = 1 - cosine_similarity; use L2 on normalized vecs
    diff = mfcc_norm.unsqueeze(0) - mfcc_norm.unsqueeze(1)  # [N, N, n_mfcc]
    return torch.norm(diff, dim=2)                           # [N, N]


def _normalize_matrix(D: torch.Tensor) -> torch.Tensor:
    """Min-max normalize a distance matrix to [0, 1]."""
    d_min = D.min()
    d_max = D.max()
    if d_max - d_min < 1e-8:
        return torch.zeros_like(D)
    return (D - d_min) / (d_max - d_min)


def compute_joint_D1(
    D1_visual: torch.Tensor,
    audio: np.ndarray,
    sr: int,
    alpha: float = 0.5,
    n_mfcc: int = 20,
) -> tuple:
    """
    Blend visual and audio distance matrices into a joint transition cost.

    Args:
        D1_visual: visual pairwise distance matrix [N, N]
        audio: raw audio waveform [T]
        sr: sample rate
        alpha: weight for visual component (1.0 = visual only, 0.0 = audio only)
        n_mfcc: number of MFCC coefficients

    Returns:
        D1_joint: blended distance matrix [N, N]
        D1_audio: audio-only distance matrix [N, N]
        mfcc: per-frame MFCC features [N, n_mfcc]
    """
    n_frames = D1_visual.shape[0]

    print(f"Extracting MFCC features for {n_frames} frames...")
    mfcc = extract_mfcc_per_frame(audio, sr, n_frames, n_mfcc=n_mfcc)

    print("Computing audio distance matrix...")
    D1_audio = compute_audio_D1(mfcc)

    # Normalize both matrices to [0, 1] before blending so alpha is meaningful
    D1_visual_norm = _normalize_matrix(D1_visual.float())
    D1_audio_norm = _normalize_matrix(D1_audio.float())

    D1_joint = alpha * D1_visual_norm + (1.0 - alpha) * D1_audio_norm

    print(f"Joint D1 built — alpha={alpha:.2f} "
          f"(visual weight={alpha:.2f}, audio weight={1-alpha:.2f})")

    return D1_joint, D1_audio, mfcc
