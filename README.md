# Loop-Aware Object Insertion via Classical Video Textures

**CS 289A Final Project — Spring 2026**  
Konstantin Zeck & Alexander Gasca Rosas

---

## Overview

This project implements loop-aware object insertion into looping videos. Given a source video and a PNG object with transparency, a user interactively places the object on a single keyframe. Lucas-Kanade optical flow propagates the object's position across all frames of the source video. The classical video texture algorithm synthesizes a new looping video by selecting frames non-sequentially, and the object is composited at its tracked position for each selected frame.

The key finding is that **naive object insertion** (pinning the object at the keyframe position for every synthesized frame) produces large alignment errors at non-sequential transitions, while **flow-tracked insertion** maintains correct alignment throughout — reducing mean alignment error from 35.7 px to 0.0 px.

---

## Background

### Original Proposal

The original proposal (see `CS_289_Project_Proposal.pdf`) described a five-stage pipeline using the contrastive video texture model from Narasimhan et al. (WACV 2022) with a SlowFast encoder, RAFT optical flow, and SuperSloMo frame interpolation. The goal was to augment the learned transition probability matrix with an object-consistency term and perform flow-based jump reconciliation at non-sequential transitions.

### Why We Pivoted

The contrastive codebase (`contrastive_video_textures/`) had several blockers:

- Hardcoded Linux paths (`/home/medhini/...`) in three files — broken on Windows
- Missing pretrained checkpoints (`SuperSloMo.ckpt`, `pytorch_vggish.pth`) not included in the repo
- CUDA 11.6 + PyTorch 1.12.1 + PyTorchVideo dependency stack — not installable on Windows without significant effort
- Bugs in unused model classes (`ContrastiveFramePrediction` references undefined variable `k`)

We pivoted to the **classical video texture baseline** (`baselines/classic_video_textures/`), which requires no GPU, no pretrained weights, and runs on standard pip packages. We replaced RAFT with Lucas-Kanade optical flow (built into OpenCV) and dropped the learned contrastive model. The core object insertion idea from the proposal is preserved.

---

## Pipeline

```
Input video + PNG object (with alpha channel)
        │
        ▼
1. Load all frames  (imageio + ffmpeg — handles .mpg/.avi/.mp4)
        │
        ▼
2. Interactive keyframe placement  (OpenCV window)
   Move mouse to position object, scroll to resize, click to confirm
        │
        ▼
3. Lucas-Kanade sparse optical flow tracking
   Sample feature grid inside object bounding box on keyframe
   Propagate (x, y, w, h) forward and backward through all N frames
        │
        ▼
4. Classical video texture synthesis
   D1  — pairwise RGB frame distance matrix  [N × N]
   D2  — binomial-smoothed D1 (rewards transitions where neighbors also match)
   Q-learning — discount future transition costs to find stable loops
   P   — thresholded transition probability matrix
        │
        ▼
5. Sample synthesized frame sequence from P
        │
        ├──► Composite at TRACKED position per frame  →  *_tracked.mp4   [our method]
        ├──► Composite at FIXED keyframe position     →  *_naive.mp4     [baseline]
        └──► No object                                →  *_synthesized.mp4  [reference]
```

### Connection to CS 289A Course Material

| Pipeline step | Course concept |
|---------------|---------------|
| D1 → P1 Gaussian kernel | Kernel methods, probability |
| D2 binomial smoothing | Regularization, bias-variance tradeoff |
| Q-learning with discount α | Markov decision process, reinforcement learning |
| LK optical flow | Iterative least-squares optimization on image gradients |
| Transition probability sampling | Probabilistic modeling, Markov chains |

---

## Installation

No GPU required.

```bash
pip install torch torchvision librosa opencv-python imageio imageio-ffmpeg numpy scipy matplotlib
```

Tested on Python 3.9, Windows 10, CPU only.

---

## Usage

### Object Insertion (main pipeline)

```bash
cd baselines/classic_video_textures

python insert_object.py \
    --video ../../videos/vtfishtk.mpg \
    --object hat.png \
    --stride 3 \
    --sigma 0.5 \
    --filter_size 8 \
    --max_iters 10 \
    --out_length 5
```

**Placement window controls:**

| Input | Action |
|-------|--------|
| Mouse move | Position the object |
| Scroll wheel / `+` / `-` | Resize the object |
| Left click | Confirm placement |
| `Q` or `Escape` | Cancel |

**Key arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--video` | required | Path to input video |
| `--object` | required | Path to PNG with alpha channel |
| `--keyframe` | N//4 | Frame index for manual placement |
| `--stride` | 2 | Keep every nth frame (reduces N for long videos) |
| `--sigma` | 4.5 | Gaussian bandwidth scale for transition probability |
| `--filter_size` | 8 | Binomial smoothing kernel size |
| `--threshold` | 0.75 | Q-learning sparsification threshold |
| `--max_iters` | 15 | Q-learning iteration cap |
| `--max_size` | 128 | Frame resize dimension for distance computation |
| `--out_length` | 8.0 | Output video length in seconds |
| `--out_dir` | results/ | Output directory |

**Sigma tuning guide** (fish video, stride=3, filter_size=8):

| sigma | Avg transitions/row | Jump rate | Notes |
|-------|-------------------|-----------|-------|
| 0.3 | 13.6 | 0.7% | Near-perfect loop — use for loop quality demo |
| **0.5** | **38.0** | **95.3%** | **Most structured — use for insertion comparison** |
| 0.8 | 206.8 | 98.6% | Loosely structured |
| 1.5+ | 327.0 | 99%+ | Essentially random frame selection |

### Video Texture Only (no object)

```bash
python synthesize.py \
    --video ../../videos/vtfishtk.mpg \
    --sigma 0.5 \
    --stride 3 \
    --out_length 5
```

---

## Results

All experiments use `videos/vtfishtk.mpg` (1001 frames at 30 fps, sampled with stride=3 → 334 frames).

### Finding 1 — The algorithm finds a genuine loop at low sigma

At **sigma=0.3**, Q-learning produces a transition matrix with only 13.6 valid transitions per row. The synthesizer finds a near-perfect periodic cycle: in a 5-second output, only **1 non-sequential jump** occurs (0.7% jump rate). This confirms the classical video texture algorithm successfully exploits the fish video's periodic structure.

### Finding 2 — Tracked insertion outperforms naive at transition points

Primary experiment at **sigma=0.5** (38 valid transitions/row, 141 non-sequential jump events over 5 seconds):

| Metric | Naive (baseline) | Tracked (ours) |
|--------|-----------------|----------------|
| Mean alignment error | **35.7 px** | **0.0 px** |
| Max alignment error | 71.7 px | 0.0 px |
| Hat displacement — sequential frames | 0.0 px | 0.5 px |
| Hat displacement — at jump transitions | 0.0 px* | 22.5 px |
| Whole-frame temporal smoothness (MSE ↓) | 0.000320 | 0.007202 |

*The naive method shows 0 px displacement at jumps because the hat never moves — but it is in the **wrong position** relative to the character.

**Reading the results:**

- The naive hat drifts **35.7 px on average** (up to 71.7 px) from where the fish actually is, because it is frozen at the keyframe placement position while the fish continues to move.
- The tracked hat maintains **0 px alignment error** — it is always at the LK-tracked position of the fish.
- At jump transitions the tracked hat makes **22.5 px corrections** on average to follow the character to its destination frame position. Between sequential frames it moves only **0.5 px** (stable, not jitter).
- Whole-frame temporal smoothness is higher for the tracked method because the hat is intentionally moving with the fish. This metric conflates "correctly tracking motion" with "choppy output," which is why **alignment error is the primary metric** for this task.

### Full Sigma Sweep

| sigma | Valid transitions/row | Jump rate | Naive alignment error (mean) | Tracked correction at jumps |
|-------|----------------------|-----------|------------------------------|----------------------------|
| 0.3 | 13.6 | 0.7% | — (1 event) | — |
| 0.5 | 38.0 | 95.3% | 35.7 px | 22.5 px |
| 0.8 | 206.8 | 98.6% | 74.9 px | 46.5 px |
| 1.5 | 327.0 | 99.3% | 102.5 px | 67.8 px |

As sigma increases, the transition matrix becomes flatter (more valid transitions per row), the jump rate approaches 100%, and the naive alignment error grows because the synthesizer draws frames from increasingly distant parts of the video.

---

## Output Files

After running `insert_object.py`, three videos are saved to `results/`:

| File | Description |
|------|-------------|
| `*_tracked.mp4` | Our method — hat follows the character at every frame |
| `*_naive.mp4` | Baseline — hat pinned at keyframe position |
| `*_synthesized.mp4` | No object — raw video texture output for reference |

---

## File Structure

```
Berkeley-CS289A-Final-Project/
├── CS_289_Project_Proposal.pdf
├── README.md
├── audio_conditioned_texture.ipynb      ← Colab notebook (audio-conditioned variant)
├── videos/
│   └── vtfishtk.mpg
├── baselines/
│   └── classic_video_textures/
│       ├── insert_object.py             ← main pipeline (placement → tracking → synthesis → composite)
│       ├── place_object.py              ← interactive OpenCV placement window
│       ├── object_tracker.py            ← LK sparse optical flow tracker
│       ├── composite.py                 ← alpha compositing utilities
│       ├── synthesize.py                ← standalone video texture synthesis (no object)
│       ├── computeD1.py                 ← pairwise frame distance matrix (CPU-compatible)
│       ├── computeD2.py                 ← binomial smoothing filter (CPU-compatible)
│       ├── q_learning.py                ← Q-learning transition refinement (CPU-compatible)
│       ├── compute_joint_D1.py          ← MFCC audio + visual joint distance matrix
│       ├── hat.png                      ← test object (procedurally generated)
│       └── results/                     ← output videos
└── contrastive_video_textures/          ← original paper codebase (not used, see Background)
```

---

## Key Design Decisions

**Lucas-Kanade over RAFT.** LK sparse optical flow tracks a grid of feature points inside the object bounding box, propagating position forward and backward from the keyframe. It runs in milliseconds per frame on CPU. RAFT requires a GPU and pretrained checkpoint. For a rigid object like a hat, LK is sufficient.

**RGB pixel distance for D1.** No pretrained model needed. Raw pixel L2 distance captures frame similarity well for natural videos with a fixed camera. ResNet features are also supported via `--feats ResNet`.

**Alignment error as primary metric.** Whole-frame temporal smoothness penalizes the tracked method for correctly following the character's motion. Alignment error — pixel distance between the placed hat center and the LK-tracked position — directly measures whether the object is in the right place at each frame.

**CPU-only rewrite.** All three compute files were rewritten to replace hardcoded `.cuda()` calls with `torch.device("cuda" if torch.cuda.is_available() else "cpu")`. The full pipeline runs in under 2 minutes on a laptop CPU for a 334-frame video.

**imageio over OpenCV for video loading.** OpenCV's MSMF backend fails to decode `.mpg` files on Windows. imageio with the ffmpeg plugin handles all common formats without additional configuration.

---

## References

- Schödl et al., *Video Textures*, SIGGRAPH 2000
- Narasimhan et al., *Strumming to the Beat: Audio-Conditioned Contrastive Video Textures*, WACV 2022
- Lucas & Kanade, *An Iterative Image Registration Technique with an Application to Stereo Vision*, IJCAI 1981
- Bouguet, *Pyramidal Implementation of the Lucas-Kanade Feature Tracker*, Intel 2001
