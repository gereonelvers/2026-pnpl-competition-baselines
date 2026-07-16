"""
Submit our baseline submissions to Kaggle from Modal (Python 3.11 + kaggle>=2.0,
which the modern KGAT_ token requires). Token comes from the Modal secret
`pnpl-kaggle-token` (KAGGLE_API_TOKEN); submission CSVs are read from the work volume.

  inspect  — verify auth + show the competition status / files / my submissions
  submit   — upload a track's submission CSV
"""

import os
import subprocess

import modal

from common import VOLUMES, WORK_DIR

app = modal.App("pnpl-kaggle-submit")

img = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("kaggle>=2.0", "pandas")
)
if modal.is_local():
    img = img.add_local_python_source("common")

SECRET = modal.Secret.from_name("pnpl-kaggle-token")
SLUG = "pnpl-competition-2026-broad"
FILES = {"deep": "deep_dascoli_submission.csv", "broad": "broad_megxl_submission.csv"}


def _kaggle(*args):
    p = subprocess.run(["kaggle", *args], capture_output=True, text=True)
    print("$ kaggle", " ".join(args))
    if p.stdout:
        print(p.stdout)
    if p.stderr:
        print("[stderr]", p.stderr)
    print(f"[rc={p.returncode}]\n---")
    return p


@app.function(image=img, volumes=VOLUMES, secrets=[SECRET], timeout=20 * 60)
def inspect(slug: str = SLUG):
    import glob
    print("submissions on volume:", glob.glob(f"{WORK_DIR}/submissions/*.csv"))
    print("KAGGLE_API_TOKEN set:", bool(os.environ.get("KAGGLE_API_TOKEN")))
    _kaggle("--version")
    _kaggle("competitions", "list", "-s", "pnpl")
    _kaggle("competitions", "files", "-c", slug)
    _kaggle("competitions", "submissions", "-c", slug)
    return {"ok": True}


@app.function(image=img, volumes=VOLUMES, secrets=[SECRET], timeout=30 * 60)
def diagnose(slug: str = SLUG):
    """Surface the full API error, and test the competition's OWN example
    submission to isolate 'my file' vs 'competition/token'."""
    import tempfile
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi(); api.authenticate()
    # 1) competition metadata (is it a code competition? accepting subs?)
    try:
        c = [x for x in api.competitions_list(search="pnpl")]
        for x in c:
            if slug in str(x.ref):
                for a in ("ref", "submissionsDisabled", "isKernelsSubmissionsOnly",
                          "evaluationMetric", "maxDailySubmissions", "userHasEntered"):
                    print(f"  {a}:", getattr(x, a, "?"))
    except Exception as e:
        print("meta err:", repr(e))
    # 2) download the competition's example submission, resubmit it
    d = tempfile.mkdtemp()
    try:
        api.competition_download_files(slug, path=d, quiet=True)
        import zipfile, glob, os
        for z in glob.glob(f"{d}/*.zip"):
            zipfile.ZipFile(z).extractall(d)
        ex = glob.glob(f"{d}/*example*submission*.csv") or glob.glob(f"{d}/*.csv")
        print("example file:", ex)
        if ex:
            try:
                api.competition_submit(ex[0], "example roundtrip", slug)
                print("EXAMPLE submit: OK")
            except Exception as e:
                print("EXAMPLE submit FAILED:", repr(e))
                r = getattr(e, "response", None)
                if r is not None:
                    print("  response body:", r.text[:800])
    except Exception as e:
        print("download err:", repr(e))
    # 3) try my broad file, surface full error
    try:
        api.competition_submit(f"{WORK_DIR}/submissions/{FILES['broad']}", "diag", slug)
        print("MY broad submit: OK")
    except Exception as e:
        print("MY broad submit FAILED:", repr(e))
        r = getattr(e, "response", None)
        if r is not None:
            print("  response body:", r.text[:800])
    return {"ok": True}


@app.function(image=img, volumes=VOLUMES, secrets=[SECRET], timeout=30 * 60)
def submit(track: str = "broad", message: str = "baseline", slug: str = SLUG):
    path = f"{WORK_DIR}/submissions/{FILES[track]}"
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    import pandas as pd
    df = pd.read_csv(path)
    print(f"submitting {track}: {path}  shape={df.shape}  -> competition {slug}")
    p = _kaggle("competitions", "submit", "-c", slug, "-f", path, "-m", message)
    ok = p.returncode == 0 and "successfully submitted" in (p.stdout + p.stderr).lower()
    return {"track": track, "rc": p.returncode, "success": ok,
            "stdout": p.stdout[-500:], "stderr": p.stderr[-500:]}
