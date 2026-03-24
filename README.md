# Online Video Depth Anything

![image/png](ovda.gif)

A [FiftyOne remote source zoo model](https://docs.voxel51.com/user_guide/model_zoo/remote.html) that wraps **Online Video Depth Anything (oVDA)** — a temporally-consistent monocular depth estimator for videos that runs in an online setting with low VRAM consumption.

---

## What is Online Video Depth Anything?

oVDA is introduced in:

> **Online Video Depth Anything: Temporally-Consistent Depth Prediction with Low Memory Consumption**  
> Johann-Friedrich Feiden, Tim Küchler, Denis Zavadski, Bogdan Savchynskyy, Carsten Rother  
> Heidelberg University  
> arXiv:2510.09182 · [https://arxiv.org/abs/2510.09182](https://arxiv.org/abs/2510.09182)

The key insight is borrowing the KV-cache concept from large language models: latent visual features are cached across frames during inference, giving the model temporal context without needing to re-process the entire video history. This makes oVDA suitable for arbitrarily long videos and edge deployment — the authors report 42 FPS on an NVIDIA A100 and 20 FPS on a Jetson edge device.

---

## Available models

| Zoo name | Cache size | Parameters | Speed ↔ Quality |
|---|---|---|---|
| `FriedFeid/oVDA-c16` | 16 frames | 29.0M | Slower, better temporal consistency |
| `FriedFeid/oVDA-c8` | 8 frames | 29.0M | Faster, lower VRAM |

Both use the `vits` (DINOv2 small) backbone with absolute positional encoding (`ape`) and a DPT head.

---

## Installation

```bash
pip install fiftyone easydict natsort huggingface_hub
```

`easydict` and `natsort` must be installed explicitly, they are required by the oVDA model code for config parsing and sorted file loading respectively.

Weights are downloaded automatically from [HuggingFace (`FriedFeid/oVDA`)](https://huggingface.co/FriedFeid/oVDA) on first use.

---

## Usage

```python
import fiftyone as fo
import fiftyone.zoo as foz
from fiftyone.utils.huggingface import load_from_hub

# Load a video dataset
dataset = load_from_hub(
    "harpreetsahota/random_short_videos",
    shuffle=True,
    max_samples=25
    )

# Register this repo as a zoo model source (one-time per environment)
foz.register_zoo_model_source(
    "https://github.com/harpreetsahota204/online_video_depth_anything",
    overwrite=True
)

# Load the model — weights are downloaded on first use
model = foz.load_zoo_model("FriedFeid/oVDA-c16")

# Run depth estimation across the dataset
dataset.apply_model(
    model,
    label_field="ovda_depth",
    batch_size=4,
    num_workers=2,
)
```

Switch to the lighter variant by swapping the model name:

```python
model = foz.load_zoo_model("FriedFeid/oVDA-c8")
```

---

## What gets written to the dataset

For each video sample, `apply_model` writes a [`fo.Heatmap`](https://docs.voxel51.com/api/fiftyone.core.labels.html#fiftyone.core.labels.Heatmap) to every frame under the specified `label_field`:

```
sample.frames[1]["ovda_depth"]   # fo.Heatmap for frame 1
sample.frames[2]["ovda_depth"]   # fo.Heatmap for frame 2
...
```

Each `fo.Heatmap` contains a 2-D array (`map`) whose shape matches the original video frame dimensions. Values encode relative depth — brighter means closer to the camera.

You can inspect predictions in the FiftyOne App with:

```python
session = fo.launch_app(dataset)
```

---

## Depth map design decisions

### Global normalisation per video

The raw model output is scale- and shift-invariant relative depth. Rather than normalising each frame independently (which would cause flickering as the per-frame min/max shifts), depth is normalised globally across the entire video:

```
depth_norm = (depth - video_min) / (video_max - video_min)
```

This preserves temporal consistency — an object that is consistently far away stays consistently dark across all frames.

### Floor value of 0.05

After normalisation, values are clamped to `[0.05, 1.0]` rather than `[0.0, 1.0]`.

The FiftyOne App renders `fo.Heatmap` overlays using the map value as the **overlay opacity** — a value of exactly `0.0` is transparent. Without the floor, pixels that the model drives to zero (e.g. near-black backgrounds after ReLU activations) would disappear entirely in the visualisation, making the depth overlay look like it has holes. The `0.05` floor ensures every pixel renders with at least a faint colormap colour.

### `uint8` storage

Normalised values `[0.05, 1.0]` are scaled to `[0, 255]` and stored as `uint8`. The quantisation step of `1/255 ≈ 0.004` is well below the perceptual threshold for depth heatmap display and results in arrays that are **4× smaller** than `float32`, reducing both memory usage and FiftyOne database write time.

---

## Advanced: loading outside FiftyOne

The model can also be instantiated directly without going through the zoo:

```python
from __init__ import load_model

model = load_model(
    model_name="FriedFeid/oVDA-c16",
    model_path="/path/to/oVDA_c16.pth",
    input_size=518,
    fp32=False,
    device="cuda:0",
)
```

Key constructor parameters:

| Parameter | Default | Description |
|---|---|---|
| `input_size` | `518` | Inference resolution (snapped to a multiple of 14) |
| `fp32` | `False` | Full precision; default is fp16 |
| `device` | auto | `"cuda:0"`, `"cpu"`, etc. |
| `use_xformers` | `False` | Enable xFormers memory-efficient attention |
| `cache_size` | `16` or `8` | Temporal cache depth (set by config) |

---

## License

This repository wraps model weights and architecture code from the oVDA project, which uses **mixed licensing**:

- Portions derived from [Video Depth Anything (VDA)](https://videodepthanything.github.io) are licensed under the **Apache License 2.0** and are marked as such in the upstream source.
- All remaining components are licensed under **NC‑SA‑UHDV1.0** (Heidelberg University Non-Commercial ShareAlike License):

  | | |
  |---|---|
  | ✅ | Free use for non-commercial research, education, and academic work |
  | 🚫 | Commercial use **prohibited** without written permission from Heidelberg University |
  | 🔄 | Modifications must carry the same license (ShareAlike) |
  | 🪪 | Attribution required: credit Heidelberg University and the original authors |
  | ⚠️ | No warranty; use at your own risk |

See the [FriedFeid/oVDA HuggingFace page](https://huggingface.co/FriedFeid/oVDA) and the upstream [GitHub repository](https://github.com/FriedFeid/OnlineVideoDepthAnything) for the full license text.

---

## Citation

```bibtex
@misc{feiden2025onlinevideodepthanything,
      title={Online Video Depth Anything: Temporally-Consistent Depth Prediction with Low Memory Consumption}, 
      author={Johann-Friedrich Feiden and Tim Küchler and Denis Zavadski and Bogdan Savchynskyy and Carsten Rother},
      year={2025},
      eprint={2510.09182},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2510.09182}, 
}
```
