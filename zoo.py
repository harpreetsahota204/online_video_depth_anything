"""FiftyOne zoo model wrapper for Online Video Depth Anything (OVDA).

Architecture note
-----------------
OVDA is a *stateful temporal* model: each frame's depth is conditioned on a
rolling cache of the previous ``cache_size`` frames.  True GPU-level batching
across frames from different videos is therefore not possible.

The pattern used here:
  * GetItem  – returns the raw filepath string (zero I/O in DataLoader workers).
  * predict_all – loads each video in the main process, runs infer_video_depth
    (which handles the full frame-by-frame cache loop), and returns a list of
    per-frame fo.Heatmap labels.
  * batch_size in apply_model controls how many videos are processed before
    FiftyOne flushes results to disk; num_workers controls filepath prefetch.
"""

import os
import sys

import numpy as np
import torch

# Make sure `src/` is importable regardless of the calling working directory.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import fiftyone as fo
from fiftyone import Model
from fiftyone.core.models import SupportsGetItem, TorchModelMixin, SamplesMixin
from fiftyone.utils.torch import GetItem

from src.models.video_depth import onlineVideoDepthAnything
from src.utils.loading_utils import load_video_as_numpy


# ---------------------------------------------------------------------------
# GetItem – lightweight, runs in DataLoader worker processes
# ---------------------------------------------------------------------------

class OVDAGetItem(GetItem):
    """Returns the video filepath string.  All heavy I/O stays in the main
    process inside predict_all so that DataLoader workers stay lean."""

    @property
    def required_keys(self):
        return ["filepath"]

    def __call__(self, sample_dict):
        return sample_dict["filepath"]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class OVDAModel(Model, SamplesMixin, SupportsGetItem, TorchModelMixin):
    """Online Video Depth Anything wrapped as a FiftyOne zoo model.

    Parameters
    ----------
    model_path : str
        Path to the ``.pth`` checkpoint file.
    encoder : {'vits', 'vitl'}
        DINOv2 backbone variant.  Must match the checkpoint.
    features : int
        Number of features in the DPT head.
    out_channels : tuple[int]
        Output channel sizes for the four DPT head stages.
    use_bn : bool
        Use batch normalisation in the DPT head.
    use_clstoken : bool
        Use the DINOv2 class token as an additional feature.
    cache_size : int
        Number of frames held in the temporal cache.
    pe : {'ape', 'rope'}
        Positional encoding mode for temporal attention.
    use_xformers : bool
        Use xFormers memory-efficient attention.  Disable on CPUs / edge
        devices where xFormers is not installed.
    input_size : int
        Approximate inference resolution (exact value is snapped to a
        multiple of 14 while keeping aspect ratio).
    fp32 : bool
        Run in full precision.  Default is fp16 (faster, barely any
        quality loss on modern GPUs).
    device : str or None
        Torch device string, e.g. ``"cuda:0"`` or ``"cpu"``.
        ``None`` auto-selects CUDA if available.
    """

    # ------------------------------------------------------------------
    # Required by fiftyone.core.models.Model
    # ------------------------------------------------------------------

    @property
    def media_type(self):
        return "video"

    @property
    def ragged_batches(self):
        # Must be False to keep FiftyOne's batching enabled.
        # Variable-size inputs are handled by our identity collate_fn.
        return False

    @property
    def transforms(self):
        return None

    @property
    def preprocess(self):
        return self._preprocess

    @preprocess.setter
    def preprocess(self, value):
        self._preprocess = value

    # ------------------------------------------------------------------
    # Required by fiftyone.core.models.TorchModelMixin
    # ------------------------------------------------------------------

    @property
    def has_collate_fn(self):
        return True

    @property
    def collate_fn(self):
        def identity_collate(batch):
            return batch
        return identity_collate

    # ------------------------------------------------------------------
    # Constructor
    # ------------------------------------------------------------------

    def __init__(
        self,
        model_path,
        encoder="vits",
        features=64,
        out_channels=(48, 96, 192, 384),
        use_bn=False,
        use_clstoken=False,
        cache_size=16,
        pe="ape",
        use_xformers=False,
        input_size=518,
        fp32=False,
        device=None,
    ):
        SamplesMixin.__init__(self)
        SupportsGetItem.__init__(self)
        self._preprocess = False

        self._input_size = input_size
        self._fp32 = fp32
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self._model = onlineVideoDepthAnything(
            encoder=encoder,
            features=features,
            out_channels=list(out_channels),
            use_bn=use_bn,
            use_clstoken=use_clstoken,
            cache_size=cache_size,
            pe=pe,
            use_xformers=use_xformers,
        )

        if model_path and os.path.isfile(model_path):
            ckpt = torch.load(model_path, map_location="cpu")
            self._model.load_state_dict(ckpt)

        self._model.eval()

    # ------------------------------------------------------------------
    # Context manager (required: apply_model uses `with model:`)
    # ------------------------------------------------------------------

    def __enter__(self):
        return self

    def __exit__(self, *args):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return False

    # ------------------------------------------------------------------
    # Required by fiftyone.core.models.SupportsGetItem
    # ------------------------------------------------------------------

    def build_get_item(self, field_mapping=None):
        return OVDAGetItem(field_mapping=field_mapping)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, arg, sample=None):
        """Single-video inference.

        FiftyOne's _apply_video_model passes an FFmpegVideoReader as ``arg``
        and the raw fo.Sample as ``sample`` (because SamplesMixin sets
        needs_samples=True).  We ignore the reader and load the video
        ourselves from ``sample.filepath`` so the full frame sequence is
        available for the temporal cache.

        Parameters
        ----------
        arg : etav.FFmpegVideoReader
            Passed by FiftyOne; not used directly.
        sample : fo.Sample
            The FiftyOne sample; provides ``.filepath``.

        Returns
        -------
        dict[int, fo.Heatmap]
            Frame-number (1-indexed) → Heatmap mapping.
            FiftyOne's sample.add_labels() stores each entry at
            sample.frames[t][label_field].
        """
        filepath = sample.filepath
        frames = load_video_as_numpy(filepath)

        # depth: np.ndarray (1, T, H, W) — batch dimension is always 1
        depth = self._model.infer_video_depth(
            frames,
            device=self._device,
            preprocess_device="cpu",
            input_size=self._input_size,
            fp32=self._fp32,
        )
        depth = depth[0]  # (T, H, W)

        # Normalise globally across the video for temporal consistency.
        dmin = float(depth.min())
        dmax = float(depth.max())
        depth_norm = (depth - dmin) / (dmax - dmin + 1e-8)

        # Floor at a small epsilon so relu-clipped pixels (depth == 0) render
        # as the coldest colormap color rather than transparent in the FiftyOne
        # app, which uses the heatmap value as overlay opacity.
        depth_norm = np.clip(depth_norm, 0.05, 1.0)

        # uint8 [0, 255] is sufficient for display and 4x smaller than float32.
        depth_uint8 = (depth_norm * 255).astype(np.uint8)

        # 1-indexed frame numbers to match FiftyOne's frame convention.
        return {
            t + 1: fo.Heatmap(map=depth_uint8[t])
            for t in range(len(depth_norm))
        }

    def predict_all(self, batch, preprocess=None, samples=None):
        """Process a batch of videos.

        Note: for video models FiftyOne's _apply_video_model always calls
        predict() directly (one video at a time) and ignores batch_size /
        num_workers.  This method is provided for completeness and direct use.

        Parameters
        ----------
        batch : list[str]
            Video filepaths.
        preprocess : bool or None
            Unused.
        samples : list[fo.Sample] or None
            Unused; filepaths come from ``batch``.

        Returns
        -------
        list[dict[int, fo.Heatmap]]
            One dict per video; each dict maps 1-indexed frame numbers to
            fo.Heatmap depth predictions.
        """
        results = []
        for filepath in batch:
            frames = load_video_as_numpy(filepath)

            depth = self._model.infer_video_depth(
                frames,
                device=self._device,
                preprocess_device="cpu",
                input_size=self._input_size,
                fp32=self._fp32,
            )
            depth = depth[0]  # (T, H, W)

            dmin = float(depth.min())
            dmax = float(depth.max())
            depth_norm = np.clip(
                (depth - dmin) / (dmax - dmin + 1e-8), 0.05, 1.0
            )
            depth_uint8 = (depth_norm * 255).astype(np.uint8)

            results.append({
                t + 1: fo.Heatmap(map=depth_uint8[t])
                for t in range(len(depth_norm))
            })

        return results
