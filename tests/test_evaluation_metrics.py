"""Tests for experimental metrics (FVD realism, diversity of transitions)."""

import numpy as np
import pytest

from evaluation.metrics import diversity_score, frechet_video_distance


def test_frechet_video_distance_identical_pools_near_zero():
    rng = np.random.default_rng(0)
    feats = rng.standard_normal((64, 16))
    fvd = frechet_video_distance(feats, feats)
    assert fvd >= 0.0
    assert fvd < 1e-4


def test_frechet_video_distance_separated_pools_is_positive():
    rng = np.random.default_rng(1)
    real = rng.standard_normal((128, 8))
    fake = rng.standard_normal((128, 8)) + 5.0
    fvd = frechet_video_distance(real, fake)
    assert fvd > 10.0


def test_frechet_video_distance_rejects_mismatched_dims():
    real = np.zeros((10, 3))
    fake = np.zeros((10, 4))
    with pytest.raises(ValueError, match="Feature dimensions"):
        frechet_video_distance(real, fake)


def test_frechet_video_distance_rejects_non_2d():
    with pytest.raises(ValueError, match="2-D"):
        frechet_video_distance(np.zeros(3), np.zeros((2, 3)))


def test_diversity_score_identical_transitions_is_zero():
    t = np.repeat(np.array([[1.0, 0.0, -1.0]]), repeats=5, axis=0)
    assert diversity_score(t) == 0.0


def test_diversity_score_increases_with_spread():
    spread = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
            [0.0, -1.0],
        ]
    )
    tight = spread * 0.01
    assert diversity_score(spread) > diversity_score(tight)


def test_diversity_score_fewer_than_two_transitions():
    assert diversity_score(np.array([[1.0, 2.0]])) == 0.0


def test_diversity_score_rejects_non_2d():
    with pytest.raises(ValueError, match="2-D"):
        diversity_score(np.array([1.0, 2.0]))
