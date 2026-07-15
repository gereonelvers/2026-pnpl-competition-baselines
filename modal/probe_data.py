"""Probe pnpl LibriBrain h5 + holdout npz structure on Modal (CPU) to plan the
MEG-XL data integration: does the h5 carry channel_names / sensor geometry?"""

import modal
from common import VOLUMES, HF_ENV

app = modal.App("pnpl-probe-data")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("h5py", "numpy", "huggingface_hub", "hf_transfer", "mne")
    .env(HF_ENV)
)


@app.function(image=image, volumes=VOLUMES, timeout=30 * 60)
def probe():
    import h5py, numpy as np
    from huggingface_hub import hf_hub_download

    fn = ("Sherlock1/derivatives/serialised/"
          "sub-10_ses-11_task-Sherlock1_run-1_proc-bads+headpos+sss+notch+bp+ds_meg.h5")
    print("downloading broad h5:", fn)
    p = hf_hub_download("pnpl/LibriBrain2", fn, repo_type="dataset")
    print("path:", p)
    with h5py.File(p, "r") as f:
        print("=== top-level keys ===")
        def show(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"  DATASET {name}: shape={obj.shape} dtype={obj.dtype}")
            else:
                print(f"  GROUP   {name}")
        f.visititems(show)
        print("=== root datasets ===", list(f.keys()))
        print("=== root attrs ===")
        for k, v in f.attrs.items():
            sv = str(v)
            print(f"  attr {k}: {sv[:120]}")
        for key in ("channel_names", "sensor_xyzdir", "sensor_types", "sfreq"):
            if key in f:
                d = f[key]
                print(f"  HAS dataset {key}: shape={getattr(d,'shape',None)} "
                      f"sample={np.asarray(d[:5]) if d.shape else np.asarray(d)}")
            elif key in f.attrs:
                print(f"  HAS attr {key}: {f.attrs[key]}")
            else:
                print(f"  MISSING {key}")
    return {"ok": True}


@app.function(image=image, volumes=VOLUMES, timeout=30 * 60)
def probe_holdout_and_layout():
    """Inspect a holdout npz + derive the standard 306-ch Neuromag layout order."""
    import numpy as np
    from huggingface_hub import hf_hub_download
    p = hf_hub_download("pnpl/LibriBrain-Competition-2026",
                        "COMPETITION_HOLDOUT/subj01_holdout2_word.npz",
                        repo_type="dataset")
    d = np.load(p, allow_pickle=True)
    print("npz keys:", list(d.keys()))
    for k in d.keys():
        a = d[k]
        print(f"  {k}: shape={getattr(a,'shape',None)} dtype={getattr(a,'dtype',None)} "
              f"{'val='+str(a) if a.ndim==0 else ''}")
    # mne standard neuromag channel order
    import mne
    print("mne version:", mne.__version__)
    return {"ok": True}
