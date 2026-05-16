"""
Print evaluation table: FVD (unconditional), FVD (conditional), diversity score.

Conditional FVD is computed only when paired feature files exist (see
``--cond_real_features`` / ``--cond_fake_features``), e.g. from matched
audio-conditioned clips.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from evaluation.metrics import diversity_score, frechet_video_distance


def _fmt(x: float) -> str:
    if np.isnan(x):
        return "N/A (needs paired conditional features)"
    return f"{x:.4f}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compute FVD and diversity scores.")
    parser.add_argument(
        "--feat_dir",
        type=Path,
        default=Path("../results/fvd_features"),
        help="Directory with real_features.npy, fake_features.npy, etc.",
    )
    parser.add_argument(
        "--cond_real_features",
        type=Path,
        default=None,
        help="Optional (N, D) features for conditional real pool.",
    )
    parser.add_argument(
        "--cond_fake_features",
        type=Path,
        default=None,
        help="Optional (M, D) features for conditional fake pool (matched conditions).",
    )
    args = parser.parse_args(argv)

    feat_dir = args.feat_dir
    real_path = feat_dir / "real_features.npy"
    fake_path = feat_dir / "fake_features.npy"
    fake_trans_path = feat_dir / "fake_transitions.npy"

    for p in (real_path, fake_path, fake_trans_path):
        if not p.is_file():
            raise FileNotFoundError(
                f"Missing {p}. Run step 3a (evaluation.extract_features) first."
            )

    real_features = np.load(real_path)
    fake_features = np.load(fake_path)
    fake_trans = np.load(fake_trans_path)

    fvd_uncond = frechet_video_distance(real_features, fake_features)
    ds = diversity_score(fake_trans) if fake_trans.size else float("nan")

    fvd_cond = float("nan")
    cond_real = args.cond_real_features or feat_dir / "cond_real_features.npy"
    cond_fake = args.cond_fake_features or feat_dir / "cond_fake_features.npy"
    if cond_real.is_file() and cond_fake.is_file():
        fvd_cond = frechet_video_distance(np.load(cond_real), np.load(cond_fake))

    print()
    print(f"{'Metric':<28} {'Value':>12}")
    print("-" * 42)
    print(f"{'FVD (UNCONDITIONAL)':<28} {_fmt(fvd_uncond):>12}")
    print(f"{'FVD (CONDITIONAL)':<28} {_fmt(fvd_cond):>12}")
    print(f"{'DIVERSITY SCORE':<28} {_fmt(ds):>12}")
    print()
    print(f"  real_features {real_features.shape}  fake_features {fake_features.shape}")
    if np.isnan(fvd_cond):
        print(
            "  Conditional FVD: add matched pools as "
            f"{cond_real.name} and {cond_fake.name}, then re-run."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
