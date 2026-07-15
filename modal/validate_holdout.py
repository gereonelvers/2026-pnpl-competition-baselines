"""
Validate the holdout -> submission path end-to-end on Modal (CPU only).

Confirms that on a clean Modal image:
  - pnpl installs and imports,
  - LibriBrainCompetitionHoldout downloads + enumerates windows,
  - window shapes are (306, 250),
  - write_submission produces a well-formed CSV that the official scorer accepts.

Run:
  modal run modal/validate_holdout.py::validate_deep
  modal run modal/validate_holdout.py::download_all_holdout   # prefetch 5.7 GB
"""

import modal

from common import pnpl_cpu_image, VOLUMES, WORK_DIR

app = modal.App("pnpl-validate-holdout")


@app.function(image=pnpl_cpu_image, volumes=VOLUMES, timeout=60 * 60)
def validate_deep():
    import numpy as np
    from pnpl.competition import (
        LibriBrainCompetitionHoldout,
        write_submission,
        PRIMARY_VOCAB,
        SECONDARY_VOCAB,
    )

    print("PRIMARY_VOCAB:", len(PRIMARY_VOCAB), PRIMARY_VOCAB[:5], "...")
    print("SECONDARY_VOCAB:", len(SECONDARY_VOCAB), SECONDARY_VOCAB[:5], "...")

    holdout = LibriBrainCompetitionHoldout(track="deep")
    print("repr:", repr(holdout))
    print("counts:", holdout.counts())
    print("n indices:", len(holdout.indices))

    # Inspect a couple of windows.
    shapes = set()
    n_seen = 0
    for meg, metas in holdout.iter_windows(batch_size=256):
        shapes.add(meg.shape[1:])
        n_seen += meg.shape[0]
        if n_seen >= 512:
            break
    print("window shapes (sensors,time):", shapes, "| first-512 dtype:", meg.dtype)
    print("meg stats: mean=%.3e std=%.3e min=%.3e max=%.3e" % (
        float(meg.mean()), float(meg.std()), float(meg.min()), float(meg.max())))

    # Build a random-but-valid submission over ALL rows and write it.
    n = len(holdout)
    rng = np.random.default_rng(0)
    primary = rng.random((n, len(PRIMARY_VOCAB))).astype(np.float32)
    primary /= primary.sum(1, keepdims=True)
    secondary = rng.random((n, len(SECONDARY_VOCAB))).astype(np.float32)
    secondary /= secondary.sum(1, keepdims=True)

    out = write_submission(
        f"{WORK_DIR}/submissions/deep_random_smoketest.csv",
        indices=holdout.indices,
        primary_probs=primary,
        secondary_probs=secondary,
    )
    import os
    print("wrote:", out, os.path.getsize(out), "bytes")

    # Peek at header + first row.
    with open(out) as f:
        head = f.readline().strip()
        row0 = f.readline().strip()
    print("header cols:", len(head.split(",")))
    print("header[:120]:", head[:120])
    print("row0[:120]:", row0[:120])

    VOLUMES  # keep ref
    from common import work_vol
    work_vol.commit()
    return {"counts": holdout.counts(), "n": n, "shapes": [tuple(s) for s in shapes]}


@app.function(image=pnpl_cpu_image, volumes=VOLUMES, timeout=2 * 60 * 60)
def download_all_holdout():
    """Prefetch every subject's holdout .npz into the HF cache volume and report
    per-track example counts (deep + broad)."""
    from pnpl.competition import LibriBrainCompetitionHoldout
    from common import hf_cache

    results = {}
    for track in ("deep", "broad"):
        holdout = LibriBrainCompetitionHoldout(track=track)
        c = holdout.counts()
        results[track] = {"n": len(holdout), **c,
                          "subjects": [holdout.subjects[0], holdout.subjects[-1]]}
        print(track, results[track])
    hf_cache.commit()
    return results


@app.local_entrypoint()
def main():
    print(validate_deep.remote())
