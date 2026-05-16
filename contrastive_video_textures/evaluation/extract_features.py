"""
Extract frozen clip features from .mp4 files for FVD / diversity metrics.

Default encoder: **2D ResNet-18** (same family as ``--enc_arch resnet18`` training).
Uses **ImageNet** weights — **not** your contrastive checkpoint (that would be
circular). Optional: ``--encoder r3d18`` (3D R3D, unrelated to ResNet training).

Example (from ``contrastive_video_textures/``)::

    python -m evaluation.extract_features \\
        --real_glob "../data/*.mp4" "../results/Clown-Fish_original.mp4" \\
        --fake_glob "../results/video_Clown-Fish_SF_5.mp4" \\
        --out_dir ../results/fvd_features
"""

from __future__ import annotations

import argparse
import glob
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.io as io
import torchvision.transforms.functional as TF
from torchvision.models.video import R3D_18_Weights, r3d_18

from models.models import ModelBuilder

# Same normalization as validate.py / training on resnet18.
IMAGENET_MEAN = [0.4345, 0.4051, 0.3775]
IMAGENET_STD = [0.2768, 0.2713, 0.2737]


@dataclass(frozen=True)
class ClipSettings:
    num_frames: int = 16
    sampling_rate: int = 2
    spatial_size: int = 224


def _collect_paths(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if not matches and Path(pattern).is_file():
            matches = [pattern]
        if not matches:
            raise FileNotFoundError(f"No videos matched pattern: {pattern!r}")
        paths.extend(Path(m) for m in matches)
    seen: set[str] = set()
    unique: list[Path] = []
    for p in paths:
        key = str(p.resolve())
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


def _clip_frame_indices(
    num_video_frames: int,
    settings: ClipSettings,
    clip_stride: int,
    max_clips: int | None,
) -> list[list[int]]:
    num_frames = settings.num_frames
    sampling_rate = settings.sampling_rate
    span = (num_frames - 1) * sampling_rate + 1
    if num_video_frames < span:
        idx = list(range(num_video_frames))
        while len(idx) < span:
            idx.append(num_video_frames - 1)
        idx = idx[:span]
        return [[idx[i * sampling_rate] for i in range(num_frames)]]

    starts = list(range(0, num_video_frames - span + 1, clip_stride))
    if max_clips is not None and len(starts) > max_clips:
        picks = np.linspace(0, len(starts) - 1, max_clips, dtype=int)
        starts = [starts[i] for i in picks]

    return [
        [start + i * sampling_rate for i in range(num_frames)] for start in starts
    ]


class ResNet18ClipEncoder(nn.Module):
    """Frozen 2D ResNet-18 trunk — same architecture family as texture training."""

    def __init__(self, device: torch.device, settings: ClipSettings):
        super().__init__()
        self.settings = settings
        trunk, self.fc_dim = ModelBuilder.build_network(arch="resnet18", pretrained=True)
        self.trunk = trunk.eval()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.to(device)
        self.device = device

    @torch.no_grad()
    def encode_clip(self, frames_rgb: torch.Tensor, frame_ids: list[int]) -> np.ndarray:
        selected = frames_rgb[frame_ids]
        size = self.settings.spatial_size
        frame_vecs: list[torch.Tensor] = []
        for i in range(selected.shape[0]):
            frame = selected[i].permute(2, 0, 1).float() / 255.0
            frame = TF.resize(frame, [size, size], antialias=True)
            frame = TF.normalize(frame, mean=IMAGENET_MEAN, std=IMAGENET_STD)
            frame = frame.unsqueeze(0).to(self.device)
            feat = self.pool(self.trunk(frame)).reshape(-1)
            frame_vecs.append(feat)
        clip_vec = torch.stack(frame_vecs, dim=0).mean(dim=0)
        return clip_vec.cpu().numpy()


class R3D18ClipEncoder(nn.Module):
    """Optional 3D R3D-18 (Kinetics) — not the same as 2D ResNet training."""

    def __init__(self, device: torch.device, settings: ClipSettings):
        super().__init__()
        self.settings = settings
        weights = R3D_18_Weights.KINETICS400_V1
        backbone = r3d_18(weights=weights)
        backbone.fc = nn.Identity()
        self.backbone = backbone.eval().to(device)
        meta = weights.meta
        self.mean = list(meta["mean"])
        self.std = list(meta["std"])
        self.device = device
        self.settings = ClipSettings(
            num_frames=settings.num_frames,
            sampling_rate=settings.sampling_rate,
            spatial_size=112,
        )

    @torch.no_grad()
    def encode_clip(self, frames_rgb: torch.Tensor, frame_ids: list[int]) -> np.ndarray:
        selected = frames_rgb[frame_ids]
        size = self.settings.spatial_size
        frames: list[torch.Tensor] = []
        for i in range(selected.shape[0]):
            frame = selected[i].permute(2, 0, 1).float() / 255.0
            frame = TF.resize(frame, [size, size], antialias=True)
            frame = TF.normalize(frame, mean=self.mean, std=self.std)
            frames.append(frame)
        clip = torch.stack(frames, dim=1).unsqueeze(0).to(self.device)
        return self.backbone(clip).detach().float().reshape(-1).cpu().numpy()


def extract_video_features(
    encoder: nn.Module,
    video_path: Path,
    settings: ClipSettings,
    *,
    clip_stride: int | None = None,
    max_clips: int | None = 64,
) -> np.ndarray:
    video, _, _ = io.read_video(str(video_path), pts_unit="sec")
    if video.numel() == 0:
        raise ValueError(f"Empty video: {video_path}")

    span = (settings.num_frames - 1) * settings.sampling_rate + 1
    stride = clip_stride if clip_stride is not None else max(span // 2, 1)
    clips = _clip_frame_indices(
        num_video_frames=video.shape[0],
        settings=settings,
        clip_stride=stride,
        max_clips=max_clips,
    )
    feats = [encoder.encode_clip(video, ids) for ids in clips]
    return np.stack(feats, axis=0)


def extract_paths(
    video_paths: list[Path],
    encoder: nn.Module,
    settings: ClipSettings,
    *,
    clip_stride: int | None = None,
    max_clips: int | None,
) -> np.ndarray:
    all_feats: list[np.ndarray] = []
    for path in video_paths:
        print(f"  encoding {path} ...")
        vf = extract_video_features(
            encoder,
            path,
            settings,
            clip_stride=clip_stride,
            max_clips=max_clips,
        )
        all_feats.append(vf)
    return np.concatenate(all_feats, axis=0)


def transition_embeddings(clip_features: np.ndarray) -> np.ndarray:
    if clip_features.shape[0] < 2:
        return np.zeros((0, clip_features.shape[1]), dtype=np.float64)
    return clip_features[1:] - clip_features[:-1]


def build_encoder(name: str, device: torch.device, settings: ClipSettings) -> nn.Module:
    if name == "resnet18":
        return ResNet18ClipEncoder(device, settings)
    if name == "r3d18":
        return R3D18ClipEncoder(device, settings)
    raise ValueError(f"Unknown encoder {name!r}; use resnet18 or r3d18.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Extract clip features from MP4s for FVD evaluation."
    )
    parser.add_argument(
        "--real_glob",
        nargs="+",
        required=True,
        help="Glob(s) or file path(s) for real / reference .mp4 files.",
    )
    parser.add_argument(
        "--fake_glob",
        nargs="+",
        required=True,
        help="Glob(s) or file path(s) for generated .mp4 files.",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("../results/fvd_features"),
        help="Directory for real_features.npy and fake_features.npy.",
    )
    parser.add_argument(
        "--encoder",
        choices=("resnet18", "r3d18"),
        default="resnet18",
        help="Frozen backbone (default resnet18 matches --enc_arch resnet18 training).",
    )
    parser.add_argument(
        "--max_clips_per_video",
        type=int,
        default=64,
        help="Cap clips per file (uniform subsample). Use 0 for no cap.",
    )
    parser.add_argument("--clip_stride", type=int, default=None)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args(argv)

    real_paths = _collect_paths(args.real_glob)
    fake_paths = _collect_paths(args.fake_glob)
    max_clips = None if args.max_clips_per_video == 0 else args.max_clips_per_video

    device = torch.device(args.device)
    settings = ClipSettings()
    print(
        f"Loading encoder={args.encoder} on {device} "
        "(frozen ImageNet/Kinetics weights, not your texture checkpoint) ..."
    )
    encoder = build_encoder(args.encoder, device, settings)

    print(f"Real videos ({len(real_paths)}):")
    real_features = extract_paths(
        real_paths,
        encoder,
        settings,
        clip_stride=args.clip_stride,
        max_clips=max_clips,
    )
    print(f"Fake videos ({len(fake_paths)}):")
    fake_features = extract_paths(
        fake_paths,
        encoder,
        settings,
        clip_stride=args.clip_stride,
        max_clips=max_clips,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    real_out = args.out_dir / "real_features.npy"
    fake_out = args.out_dir / "fake_features.npy"
    np.save(real_out, real_features)
    np.save(fake_out, fake_features)

    real_trans = transition_embeddings(real_features)
    fake_trans = transition_embeddings(fake_features)
    np.save(args.out_dir / "real_transitions.npy", real_trans)
    np.save(args.out_dir / "fake_transitions.npy", fake_trans)

    print(f"Saved {real_out}  shape={real_features.shape}")
    print(f"Saved {fake_out}  shape={fake_features.shape}")
    print(f"Saved transition arrays: {real_trans.shape}, {fake_trans.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
