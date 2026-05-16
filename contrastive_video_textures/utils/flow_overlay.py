"""Flow-guided PNG overlay for native video frames (inference-only)."""

import os
from typing import Optional

import cv2
import numpy as np
from PIL import Image


def load_and_scale_sprite(path: str, max_side: int) -> Image.Image:
    im = Image.open(path).convert("RGBA")
    w, h = im.size
    if w <= 0 or h <= 0:
        raise ValueError("Invalid sprite size for {}".format(path))
    scale = max_side / float(max(w, h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return im.resize((new_w, new_h), Image.Resampling.LANCZOS)


def sample_placements(
    num: int,
    H: int,
    W: int,
    sprite_w: int,
    sprite_h: int,
    rng: np.random.Generator,
    xy_sigma_frac: float,
    max_tries: int = 100000,
) -> np.ndarray:
    if sprite_w > W or sprite_h > H:
        raise ValueError(
            "Sprite ({}, {}) does not fit in frame ({}, {})".format(
                sprite_w, sprite_h, W, H
            )
        )
    placements = np.zeros((num, 2), dtype=np.int64)
    sigma = float(xy_sigma_frac) * float(min(H, W))
    cx = W / 2.0
    cy = H / 2.0
    mean_x = cx - sprite_w / 2.0
    mean_y = cy - sprite_h / 2.0

    for i in range(num):
        placed = False
        for _ in range(max_tries):
            x = int(round(mean_x + rng.normal(0.0, sigma)))
            y = int(round(mean_y + rng.normal(0.0, sigma)))
            if x >= 0 and y >= 0 and x + sprite_w <= W and y + sprite_h <= H:
                placements[i, 0] = x
                placements[i, 1] = y
                placed = True
                break
        if not placed:
            raise RuntimeError(
                "Could not sample in-bounds placement {} after {} tries".format(
                    i, max_tries
                )
            )
    return placements


def sprite_to_numpy_rgba(sprite: Image.Image) -> np.ndarray:
    return np.asarray(sprite, dtype=np.uint8)


def render_placement_layer(
    sprite_rgba: np.ndarray, placement_xy: np.ndarray, H: int, W: int
) -> np.ndarray:
    """Full-frame RGB composite of the sprite on a black canvas."""
    layer = np.zeros((H, W, 3), dtype=np.uint8)
    x, y = int(placement_xy[0]), int(placement_xy[1])
    sh, sw = sprite_rgba.shape[0], sprite_rgba.shape[1]
    rgb = sprite_rgba[..., :3].astype(np.float32)
    alpha = sprite_rgba[..., 3:4].astype(np.float32) / 255.0
    blended = (rgb * alpha).astype(np.uint8)
    layer[y : y + sh, x : x + sw] = blended
    return layer


def build_distance_matrix(rgb_layers: list) -> np.ndarray:
    """Pairwise Euclidean (L2) distance between flattened RGB uint8 layers."""
    flat = np.stack([x.reshape(-1).astype(np.float64) for x in rgb_layers])
    diff = flat[:, None, :] - flat[None, :, :]
    return np.sqrt((diff * diff).sum(axis=2))


def compute_flow(prev_gray: np.ndarray, curr_gray: np.ndarray) -> np.ndarray:
    return cv2.calcOpticalFlowFarneback(
        prev_gray,
        curr_gray,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )


def bilinear_sample(flow: np.ndarray, pt: np.ndarray) -> np.ndarray:
    """Sample optical flow at (x, y); flow[h, w] = (dx, dy). pt = (x, y)."""
    x = float(pt[0])
    y = float(pt[1])
    h, w = flow.shape[:2]
    eps = 1e-4
    x = min(max(x, 0.0), w - 1.0 - eps)
    y = min(max(y, 0.0), h - 1.0 - eps)
    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    x1 = min(x0 + 1, w - 1)
    y1 = min(y0 + 1, h - 1)
    dx = x - x0
    dy = y - y0
    f00 = flow[y0, x0]
    f01 = flow[y1, x0]
    f10 = flow[y0, x1]
    f11 = flow[y1, x1]
    f0 = f00 * (1.0 - dy) + f01 * dy
    f1 = f10 * (1.0 - dy) + f11 * dy
    return f0 * (1.0 - dx) + f1 * dx


def transition_probs(
    prev_z: int,
    centers: np.ndarray,
    D: np.ndarray,
    p_pred: np.ndarray,
    sigma_d: float,
    sigma_f: float,
) -> np.ndarray:
    sigma_d = max(float(sigma_d), 1e-12)
    sigma_f = max(float(sigma_f), 1e-12)
    d_row = D[prev_z].astype(np.float64)
    diff = centers.astype(np.float64) - p_pred.reshape(1, 2)
    flow_term = (diff * diff).sum(axis=1)
    E = d_row / sigma_d + flow_term / sigma_f
    E = E - np.max(E)
    P = np.exp(-E)
    s = P.sum()
    if s <= 0 or not np.isfinite(s):
        P = np.ones_like(P) / len(P)
    else:
        P = P / s
    return P


def composite_on_frame(
    frame_rgb_uint8: np.ndarray, sprite_rgba: np.ndarray, xy: np.ndarray
) -> np.ndarray:
    x, y = int(xy[0]), int(xy[1])
    sh, sw = sprite_rgba.shape[0], sprite_rgba.shape[1]
    out = frame_rgb_uint8.copy()
    region = out[y : y + sh, x : x + sw].astype(np.float32)
    rgb = sprite_rgba[..., :3].astype(np.float32)
    alpha = sprite_rgba[..., 3:4].astype(np.float32) / 255.0
    comp = rgb * alpha + region * (1.0 - alpha)
    out[y : y + sh, x : x + sw] = np.clip(comp, 0, 255).astype(np.uint8)
    return out


def rgb_to_gray_u8(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


class FlowOverlayRunner:
    """Advances a sprite placement using Farnebäck flow on native frames."""

    def __init__(
        self,
        png_path: str,
        H: int,
        W: int,
        num_placements: int = 100,
        sigma_d: float = 1e6,
        sigma_f: float = 500.0,
        xy_sigma_frac: float = 0.15,
        seed: Optional[int] = None,
    ):
        if not os.path.isfile(png_path):
            raise FileNotFoundError("overlay PNG not found: {}".format(png_path))
        self.H = int(H)
        self.W = int(W)
        self.sigma_d = float(sigma_d)
        self.sigma_f = float(sigma_f)
        self.num_placements = int(num_placements)
        max_side = int(round(0.05 * min(self.H, self.W)))
        if max_side < 1:
            max_side = 1

        sprite_pil = load_and_scale_sprite(png_path, max_side)
        self.sprite_rgba = sprite_to_numpy_rgba(sprite_pil)
        self.sprite_h, self.sprite_w = self.sprite_rgba.shape[0], self.sprite_rgba.shape[1]

        self.rng = np.random.default_rng(seed)
        self.placements = sample_placements(
            self.num_placements,
            self.H,
            self.W,
            self.sprite_w,
            self.sprite_h,
            self.rng,
            xy_sigma_frac,
        )
        self.centers = np.zeros((self.num_placements, 2), dtype=np.float64)
        self.centers[:, 0] = self.placements[:, 0] + self.sprite_w / 2.0
        self.centers[:, 1] = self.placements[:, 1] + self.sprite_h / 2.0

        layers = []
        for i in range(self.num_placements):
            layers.append(
                render_placement_layer(
                    self.sprite_rgba, self.placements[i], self.H, self.W
                )
            )
        self.D = build_distance_matrix(layers)

        self._prev_native_gray: Optional[np.ndarray] = None
        self._prev_z: Optional[int] = None
        self._p: Optional[np.ndarray] = None

    def process_frame(self, frame_rgb_uint8: np.ndarray) -> np.ndarray:
        if frame_rgb_uint8.dtype != np.uint8:
            frame_rgb_uint8 = frame_rgb_uint8.astype(np.uint8)
        if frame_rgb_uint8.shape[2] != 3:
            raise ValueError("Expected HxWx3 RGB uint8 frame")

        curr_gray = rgb_to_gray_u8(frame_rgb_uint8)

        if self._prev_native_gray is None:
            z = int(self.rng.integers(0, high=self.num_placements))
        else:
            assert self._prev_z is not None and self._p is not None
            flow = compute_flow(self._prev_native_gray, curr_gray)
            v = bilinear_sample(flow, self._p)
            p_pred = self._p + v
            probs = transition_probs(
                self._prev_z, self.centers, self.D, p_pred, self.sigma_d, self.sigma_f
            )
            z = int(self.rng.choice(self.num_placements, p=probs))

        xy = self.placements[z]
        self._p = self.centers[z].copy()
        self._prev_z = z
        self._prev_native_gray = curr_gray

        return composite_on_frame(frame_rgb_uint8, self.sprite_rgba, xy)
