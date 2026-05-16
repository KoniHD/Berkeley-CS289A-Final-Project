# Audio-Conditioned Video Texture Generation

Berkeley **CS 289A** course project based on the WACV 2022 system *Strumming to the Beat: Audio-Conditioned Contrastive Video Textures*. This repository keeps the original training and baseline code, modernizes local setup with **uv** / `pyproject.toml`, and adds a small **clownfish** driving-video experiment (`data/`, `videos/`) plus optional **flow-guided PNG overlay** for compositing a sprite during synthesis.

**Interactive demo:** open **[`demo.ipynb`](demo.ipynb)** for a Google Colab-oriented walkthrough (install, checkpoints, sample training command).

## Example outputs (side by side)

<table>
<tr>
<td align="center" width="33%"><b>Original clip</b><br/>
<video src="https://raw.githubusercontent.com/KoniHD/Berkeley-CS289A-Final-Project/main/results/Clown-Fish_original.mp4" width="100%" controls muted playsinline preload="metadata"></video>
</td>
<td align="center" width="33%"><b>Synthesized texture (no fish overlay)</b><br/>
<video src="https://raw.githubusercontent.com/KoniHD/Berkeley-CS289A-Final-Project/main/results/video_Clown-Fish_SF_5.mp4" width="100%" controls muted playsinline preload="metadata"></video>
</td>
<td align="center" width="33%"><b>With flow-guided fish overlay</b><br/>
<video src="https://raw.githubusercontent.com/KoniHD/Berkeley-CS289A-Final-Project/main/results/video_Clown-Fish_ADD_SF_5.mp4" width="100%" controls muted playsinline preload="metadata"></video>
</td>
</tr>
</table>

If the embedded players do not render in your viewer, use the files under [`results/`](results/): [original](results/Clown-Fish_original.mp4) · [synthesized](results/video_Clown-Fish_SF_5.mp4) · [with overlay](results/video_Clown-Fish_ADD_SF_5.mp4).

## Environment setup (uv, recommended)

```bash
uv sync   # installs the project and locked deps from pyproject.toml / uv.lock
```

The original upstream instructions used Conda (`avtexture.yml`). For reproducible installs, prefer **uv** as above.

## Checkpoints the code expects

| Artifact | Why | Where the code loads it |
| --- | --- | --- |
| **SlowFast** Kinetics config + `.pkl` | Video encoder backbone when `--enc_arch slowfast` | Hardcoded under `/home/medhini/audio_video_gan/contrastive_video_textures/...` (see `contrastive_video_textures/models/models.py`, `dataset/dataset.py`). **`demo.ipynb`** creates those directories and downloads the files. On your own machine, mirror that layout or edit those paths. |
| **`pytorch_vggish.pth`** | Audio encoder | Repo root: `torch.load("pytorch_vggish.pth")` in `main.py` / `validate.py`. |
| **`ckpt/SuperSloMo.ckpt`** | Frame interpolation when `--evaluate` and interpolation is on | `validate.py` loads `ckpt/SuperSloMo.ckpt` (SuperSloMo / “slomo” weights). **`demo.ipynb`** downloads this into `ckpt/`. |

**Note:** `--SF` is the *slow-motion interpolation factor* used during **evaluation**, not a knob for SlowFast/ImageNet pretraining. It does not change training in a meaningful way.

## Repository layout (high level)

- `contrastive_video_textures/` — main model, training, validation, overlay options.
- `baselines/` — classic and audio baselines from the paper.
- `data/`, `videos/` — example assets for the fish demo.
- `results/` — example rendered clips for the README.
- `demo.ipynb` — Colab-oriented setup and sample command.

## Citation (original paper)

If you find the upstream method useful in your research, please cite:

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

## Contrastive Video Textures

`cd contrastive_video_textures`

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

`cd baselines`

### Video Textures Baslines

`cd classic_video_textures`

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

`cd audio_baselines`

1. Random Clip: `python random_segment_baseline.py -vl <original_video_list> -tl <target_audio_list>`
2. Random Baseline: `python random_baseline.py -vl <original_video_list> -tl <target_audio_list>`
3. Random Shift: `python random_shift.py -vl <original_video_list> -tl <target_audio_list>`
4. Audio Nearest Neighbour: `python audio_nearestneighbour.py -vl <original_video_list> -dl <target_audio_list>`
