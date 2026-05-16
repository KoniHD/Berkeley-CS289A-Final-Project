"""
Evaluation metrics aligned with underwater / schooling-fish experiments:

- **Fréchet Video Distance (FVD):** Fréchet distance between Gaussian fits to video
  features (same form as FID; typically features come from a Frozen I3D or similar).
- **Diversity Score (DS):** Mean pairwise distance between *transition* embeddings
  (e.g. per-clip differences of frame-wise features), measuring novelty of transitions.
"""

from __future__ import annotations

import numpy as np
from scipy import linalg
from scipy.spatial.distance import pdist


def _feature_covariance(features: np.ndarray) -> np.ndarray:
    """Unbiased covariance; zero matrix if fewer than two samples."""
    x = np.asarray(features, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    n, d = x.shape
    if n <= 1:
        return np.zeros((d, d), dtype=np.float64)
    cov = np.cov(x, rowvar=False, bias=False)
    return np.atleast_2d(cov)


def _frechet_gaussians(
    mu1: np.ndarray,
    sigma1: np.ndarray,
    mu2: np.ndarray,
    sigma2: np.ndarray,
    eps: float = 1e-6,
) -> float:
    """Fréchet distance between N(mu1, sigma1) and N(mu2, sigma2)."""
    mu1 = np.asarray(mu1, dtype=np.float64).reshape(-1)
    mu2 = np.asarray(mu2, dtype=np.float64).reshape(-1)
    sigma1 = np.asarray(sigma1, dtype=np.float64)
    sigma2 = np.asarray(sigma2, dtype=np.float64)

    assert sigma1.shape == sigma2.shape == (mu1.shape[0], mu1.shape[0])

    sigma1 = sigma1 + np.eye(sigma1.shape[0]) * eps
    sigma2 = sigma2 + np.eye(sigma2.shape[0]) * eps

    diff = mu1 - mu2
    covmean = linalg.sqrtm(sigma1 @ sigma2)

    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset) @ (sigma2 + offset))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    fid = float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2.0 * tr_covmean)
    return max(fid, 0.0)


def frechet_video_distance(
    real_features: np.ndarray,
    fake_features: np.ndarray,
    eps: float = 1e-6,
) -> float:
    """
    Fréchet Video Distance from two pools of per-clip (or per-video) feature vectors.

    Parameters
    ----------
    real_features:
        Array of shape (N, D) from real underwater footage (or any reference distribution).
    fake_features:
        Array of shape (M, D) from synthesized clips.
    """
    real_features = np.asarray(real_features, dtype=np.float64)
    fake_features = np.asarray(fake_features, dtype=np.float64)
    if real_features.ndim != 2 or fake_features.ndim != 2:
        raise ValueError("Features must be 2-D arrays shaped (num_samples, dim).")
    if real_features.shape[1] != fake_features.shape[1]:
        raise ValueError("Feature dimensions must match between real and fake.")

    mu1 = real_features.mean(axis=0)
    mu2 = fake_features.mean(axis=0)
    sigma1 = _feature_covariance(real_features)
    sigma2 = _feature_covariance(fake_features)
    return _frechet_gaussians(mu1, sigma1, mu2, sigma2, eps=eps)


def diversity_score(transition_features: np.ndarray, *, metric: str = "euclidean") -> float:
    """
    Diversity score: mean pairwise distance between transition embeddings.

    ``transition_features`` should have shape (T, D), where each row is a vector
    capturing a transition (e.g. f(x_{t+1}) - f(x_t) aggregated over a clip).

    Parameters
    ----------
    transition_features:
        (T, D) transition embeddings.
    metric:
        Passed to ``scipy.spatial.distance.pdist`` (default ``\"euclidean\"``).
    """
    x = np.asarray(transition_features, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("transition_features must be a 2-D array shaped (num_transitions, dim).")
    if x.shape[0] < 2:
        return 0.0
    dists = pdist(x, metric=metric)
    return float(np.mean(dists))
