# Audio-Conditioned Video Texture Generation

This is the official Pytorch implementation for the paper, "Strumming to the Beat: Audio-Coniditoned Contrastive Video Texture Synthesis", WACV 2022. We provide the datasets and code for training and testing the contrastive video texture synthesis model and the baselines as described in the paper.  

If you find our repo useful in your research, please use the following BibTeX entry for citation.

```BibTeX
@InProceedings{Narasimhan_2022_WACV,
    author    = {Narasimhan, Medhini and Ginosar, Shiry and Owens, Andrew and Efros, Alexei A. and Darrell, Trevor},
    title     = {Strumming to the Beat: Audio-Conditioned Contrastive Video Textures},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {January},
    year      = {2022},
    pages     = {3761-3770}
}
```

<<<<<<< Updated upstream
## Environment Setup

Create the conda environment from the yaml file and activate the environment,
=======
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
>>>>>>> Stashed changes

```
conda env create -f avgan.yml
conda activate avgan
```

## Dataset

Coming soon!

## Contrastive Video Textures

```cd contrastive_video_textures```

Train model for a single video:

```
python main.py -vdata <path to video folder> -m 1 -w 20 -stride 4 -temp 0.1 -th 0.0 -bs 8 -negs 14 -vl <list of video names> -ea slowfast -lr 1e-4
```

Synthesize texture for the same video using the above model:

```
python main.py -vdata <path to video folder> -m 1 -w 20 -stride 4 -temp 0.1 -th 0.3 -bs 24 -vl <list of video names> -e -mbs 100
```

## Audio-Conditioned Contrastive Video Texture Synthesis

First, train a contrastive model for the video using the command above. Ensure that the audio for the same video is in the audio folder as a wav file with the same name. Next, to synthesize a new video conditioned on an audio, 

```
python main.py -vdata <path to video folder> -adata <path to audio folder> -m 2 -w 20 -stride 4 -temp 0.1 -th 0.0 -bs 24 -negs 20 -e -vl <list of video names> -da <list of coniditioning audios> -alpha 0.5 
```

## Baselines

```cd baselines```

### Video Textures Baslines

```cd classic_video_textures```

1. Classic: 

```
python video_textures.py -m 1 -vdata <source video folder> -vl <list of video names> -s -bs 48
```

2. Classic+:

```
python video_textures.py -m 2 -vdata <source video folder> -vl <list of video names> -s -bs 48
```

3. Classic++: 

```
python video_textures.py -m 3 -vdata <source video folder> -vl <list of video names> -s -bs 48
```

### Audio-Conditioned Video Textures Baselines

```cd audio_baselines```

1. Random Clip: ```python random_segment_baseline.py -vl <original_video_list> -tl <target_audio_list>```
2. Random Baseline: ```python random_baseline.py -vl <original_video_list> -tl <target_audio_list>```
3. Random Shift: ```python random_shift.py -vl <original_video_list> -tl <target_audio_list>```
4. Audio Nearest Neighbour: ```python audio_nearestneighbour.py -vl <original_video_list> -dl <target_audio_list>```
