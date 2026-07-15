"""
Shared Modal configuration for the PNPL 2026 baseline runs.

Everything data-heavy runs on Modal (local machine has ~3 GB free). We keep:
  - HF downloads on a persistent volume (`pnpl-hf-cache`)
  - training data / checkpoints / submissions on a persistent volume (`pnpl-work`)

Local source trees copied into the images (only referenced during local image
build; guarded by ``modal.is_local()`` so re-importing this module inside a
container never touches non-existent host paths):
  - pnpl                (competition + dataset loaders)
  - baselines/dascoli-word-decoding  (deep baseline)
  - baselines/MEG-XL                 (broad baseline)
"""

from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Local source paths (only valid on the host, during image build)
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
REPO_ROOT = _HERE.parent.parent  # .../2026-competition
PNPL_SRC = REPO_ROOT / "pnpl"
DASCOLI_SRC = REPO_ROOT / "baselines" / "dascoli-word-decoding"
MEGXL_SRC = REPO_ROOT / "baselines" / "MEG-XL"

# ---------------------------------------------------------------------------
# Persistent volumes
# ---------------------------------------------------------------------------
hf_cache = modal.Volume.from_name("pnpl-hf-cache", create_if_missing=True)
work_vol = modal.Volume.from_name("pnpl-work", create_if_missing=True)

HF_CACHE_DIR = "/hf-cache"
WORK_DIR = "/work"

VOLUMES = {HF_CACHE_DIR: hf_cache, WORK_DIR: work_vol}

HF_ENV = {
    "HF_HOME": HF_CACHE_DIR,
    "HF_HUB_CACHE": f"{HF_CACHE_DIR}/hub",
    "HF_HUB_ENABLE_HF_TRANSFER": "1",
    "HF_DATASETS_CACHE": f"{HF_CACHE_DIR}/datasets",
}


def add_pnpl(img: modal.Image) -> modal.Image:
    """Copy the local pnpl tree into the image and install it editable.

    No-op inside a container (image already built) so re-import can't fail on the
    missing host path.
    """
    if not modal.is_local():
        return img
    return img.add_local_dir(
        str(PNPL_SRC), "/root/pnpl", copy=True, ignore=["*.pyc", "__pycache__", ".git"]
    ).run_commands("pip install --no-deps -e /root/pnpl")


# Lightweight image: just enough to run the holdout loader + write_submission.
_base_cpu = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy", "pandas", "h5py", "huggingface_hub", "hf_transfer", "requests")
    .env(HF_ENV)
)
pnpl_cpu_image = add_pnpl(_base_cpu)
if modal.is_local():
    pnpl_cpu_image = pnpl_cpu_image.add_local_python_source("common")
