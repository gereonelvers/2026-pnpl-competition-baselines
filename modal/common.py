"""
Shared Modal configuration for the PNPL 2026 baseline runs.

Everything data-heavy runs on Modal (local machine has ~3 GB free). We keep:
  - HF downloads on a persistent volume (`hf-cache`)
  - training data / checkpoints / submissions on a persistent volume (`pnpl-vol`)

Local source trees copied into the images:
  - pnpl                (competition + dataset loaders)
  - baselines/dascoli-word-decoding  (deep baseline)
  - baselines/MEG-XL                 (broad baseline)
"""

from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Local source paths (resolved relative to this file)
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
REPO_ROOT = _HERE.parent.parent  # .../2026-competition
PNPL_SRC = REPO_ROOT / "pnpl"
DASCOLI_SRC = REPO_ROOT / "baselines" / "dascoli-word-decoding"
MEGXL_SRC = REPO_ROOT / "baselines" / "MEG-XL"

assert PNPL_SRC.exists(), f"pnpl source not found at {PNPL_SRC}"

# ---------------------------------------------------------------------------
# Persistent volumes
# ---------------------------------------------------------------------------
# HF cache: all huggingface_hub downloads land here (training data + holdout +
# pretrained checkpoints), so we never re-download across runs.
hf_cache = modal.Volume.from_name("pnpl-hf-cache", create_if_missing=True)
# Working volume: training data (pnpl loaders write here), checkpoints, outputs.
work_vol = modal.Volume.from_name("pnpl-work", create_if_missing=True)

HF_CACHE_DIR = "/hf-cache"
WORK_DIR = "/work"

VOLUMES = {HF_CACHE_DIR: hf_cache, WORK_DIR: work_vol}

# Route all HF caches to the volume.
HF_ENV = {
    "HF_HOME": HF_CACHE_DIR,
    "HF_HUB_CACHE": f"{HF_CACHE_DIR}/hub",
    "HF_HUB_ENABLE_HF_TRANSFER": "1",
    "HF_DATASETS_CACHE": f"{HF_CACHE_DIR}/datasets",
}


# ---------------------------------------------------------------------------
# Base image with pnpl (used for holdout loading + submission on CPU)
# ---------------------------------------------------------------------------
def _add_pnpl(img: modal.Image) -> modal.Image:
    """Copy the local pnpl tree into the image and install it editable."""
    return img.add_local_dir(
        str(PNPL_SRC), "/root/pnpl", copy=True, ignore=["~*", "*.pyc", "__pycache__"]
    ).run_commands("pip install --no-deps -e /root/pnpl")


# Lightweight image: just enough to run the holdout loader + write_submission.
pnpl_cpu_image = _add_pnpl(
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy",
        "pandas",
        "h5py",
        "huggingface_hub",
        "hf_transfer",
        "requests",
    )
    .env(HF_ENV)
    .add_local_python_source("common")
)
