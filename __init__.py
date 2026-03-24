"""FiftyOne remote source zoo model package for Online Video Depth Anything.

FiftyOne calls three functions from this module:

  download_model  – fetches the checkpoint from HuggingFace Hub
  load_model      – instantiates OVDAModel ready for inference
  resolve_input   – (optional) defines operator UI parameters

HuggingFace repo
----------------
Both checkpoints live in a single repo: https://huggingface.co/FriedFeid/oVDA
  oVDA_c16.pth  –  cache_size=16 (default, higher temporal context)
  oVDA_c8.pth   –  cache_size=8  (lighter, lower memory)

Manifest entries use different base_filename values ("FriedFeid-oVDA-c16" /
"FriedFeid-oVDA-c8") so download_model and load_model can infer which
variant is being requested from the local model_path suffix.
"""

import os
import sys
import shutil

# Ensure this directory is on sys.path so `src` can be found by zoo.py.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from .zoo import OVDAModel


# ---------------------------------------------------------------------------
# Per-variant configuration
# ---------------------------------------------------------------------------

# The actual HuggingFace repo that hosts both checkpoints.
# base_name in manifest.json carries the variant suffix (e.g. "FriedFeid/oVDA-c16")
# so FiftyOne can distinguish the two models; we strip the suffix here for downloads.
_HF_REPO = "FriedFeid/oVDA"

# Keys accepted by onlineVideoDepthAnything.__init__
_MODEL_INIT_KEYS = {
    "encoder", "features", "out_channels", "use_bn",
    "use_clstoken", "cache_size", "pe", "use_xformers",
}

_CONFIGS = {
    "oVDA_c16": {
        "encoder": "vits",
        "features": 64,
        "out_channels": [48, 96, 192, 384],
        "use_bn": False,
        "use_clstoken": False,
        "cache_size": 16,
        "pe": "ape",
        "use_xformers": False,
        "checkpoint": "oVDA_c16.pth",
    },
    "oVDA_c8": {
        "encoder": "vits",
        "features": 64,
        "out_channels": [48, 96, 192, 384],
        "use_bn": False,
        "use_clstoken": False,
        "cache_size": 8,
        "pe": "ape",
        "use_xformers": False,
        "checkpoint": "oVDA_c8.pth",
    },
}


def _infer_config(model_path):
    """Derive 'oVDA_c8' or 'oVDA_c16' from the local model_path suffix."""
    basename = os.path.basename(os.path.normpath(model_path or ""))
    return "oVDA_c8" if "c8" in basename else "oVDA_c16"


# ---------------------------------------------------------------------------
# FiftyOne entry points
# ---------------------------------------------------------------------------

def download_model(model_name, model_path):
    """Download the relevant OVDA checkpoint from HuggingFace Hub.

    FiftyOne calls this automatically the first time a model is loaded via
    foz.load_zoo_model().

    Both checkpoints live in the single repo _HF_REPO ("FriedFeid/oVDA").
    The manifest base_name carries a variant suffix ("FriedFeid/oVDA-c16" or
    "FriedFeid/oVDA-c8") so FiftyOne can distinguish the two models; we infer
    which .pth to fetch from that suffix and download only that file.

    Parameters
    ----------
    model_name : str
        ``base_name`` from manifest.json, e.g. ``"FriedFeid/oVDA-c16"``.
    model_path : str
        Local directory path derived from ``base_filename``,
        e.g. ``".../FriedFeid-oVDA-c16"``.
    """
    from huggingface_hub import hf_hub_download

    cfg = _CONFIGS[_infer_config(model_name)]
    filename = cfg["checkpoint"]  # "oVDA_c16.pth" or "oVDA_c8.pth"

    # hf_hub_download caches to ~/.cache/huggingface; copy into FiftyOne's
    # managed model_path directory so load_model can find it by convention.
    cached = hf_hub_download(repo_id=_HF_REPO, filename=filename)

    os.makedirs(model_path, exist_ok=True)
    dest = os.path.join(model_path, filename)
    if not os.path.isfile(dest):
        shutil.copy2(cached, dest)


def load_model(model_name=None, model_path=None, **kwargs):
    """Instantiate and return an OVDAModel ready for inference.

    FiftyOne calls this after download_model() has placed the checkpoint.

    Parameters
    ----------
    model_name : str or None
        Passed by FiftyOne; not used directly (variant is inferred from
        model_path).
    model_path : str or None
        Directory containing the downloaded ``.pth`` file, e.g.
        ``".../FriedFeid-oVDA-c16"``.  May also be a direct path to a
        ``.pth`` file for manual use outside FiftyOne.
    **kwargs
        Any OVDAModel constructor overrides, e.g.::

            load_model(model_path="...", input_size=384, device="cuda:1",
                       fp32=True)
    """
    if model_path is None:
        raise ValueError(
            "model_path must be provided (path to the checkpoint directory "
            "or a direct .pth file)."
        )

    if os.path.isfile(model_path):
        # Direct path to a .pth file — infer config from the filename.
        ckpt_path = model_path
        config_key = _infer_config(model_name or model_path)
    else:
        # model_name carries the variant suffix when called by FiftyOne;
        # fall back to model_path suffix for direct / manual calls.
        config_key = _infer_config(model_name or model_path)
        cfg = _CONFIGS[config_key]
        ckpt_path = os.path.join(model_path, cfg["checkpoint"])

    cfg = _CONFIGS[config_key]
    model_kwargs = {k: v for k, v in cfg.items() if k in _MODEL_INIT_KEYS}
    model_kwargs.update(kwargs)

    return OVDAModel(model_path=ckpt_path, **model_kwargs)


def resolve_input(model_name, ctx):
    """Optional: define custom FiftyOne operator UI parameters.

    Returns None to use the default apply_model() parameters.
    """
    return None
