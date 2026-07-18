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

from common import VOLUMES, WORK_DIR, add_pnpl

app = modal.App("pnpl-kaggle-submit")

img = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("kaggle>=2.0", "pandas")
)
# The edited local pnpl + kaggle, to exercise pnpl.competition.submit_to_kaggle itself.
# Build steps must precede any add_local_* step, so install into a fresh base then add_pnpl.
helper_img = add_pnpl(
    modal.Image.debian_slim(python_version="3.11").pip_install("numpy", "pandas", "kaggle>=2.0")
)
if modal.is_local():
    img = img.add_local_python_source("common")
    helper_img = helper_img.add_local_python_source("common")

SECRET = modal.Secret.from_name("pnpl-kaggle-token")
SLUG = "pnpl-competition-2026"
FILES = {"deep": "deep_dascoli_submission.csv", "broad": "broad_megxl_submission.csv",
         "test": "test_pnpl2026_submission.csv"}


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


@app.function(image=helper_img, volumes=VOLUMES, secrets=[SECRET], timeout=20 * 60)
def test_pnpl_helper():
    """Exercise the library helper pnpl.competition.submit_to_kaggle end-to-end, incl. the
    new deep/broad shorthands. Combined comp is configured (expect success); deep is not yet
    configured (expect a clean failure that proves the shorthand reached the right slug)."""
    from pnpl.competition import submit_to_kaggle, resolve_competition, PNPL_2026_COMPETITIONS
    print("PNPL_2026_COMPETITIONS =", PNPL_2026_COMPETITIONS)
    print("resolve('deep') =", resolve_competition("deep"),
          "| resolve('broad') =", resolve_competition("broad"))

    print("\n[1] helper -> combined competition (full slug, should SUCCEED):")
    r1 = submit_to_kaggle(f"{WORK_DIR}/submissions/test_pnpl2026_submission.csv",
                          competition="pnpl-competition-2026",
                          message="via pnpl.submit_to_kaggle helper", check=False)
    print("   result:", r1)
    print("   -> competition field:", r1.competition, "| success:", r1.success)

    print("\n[2] helper -> 'deep' shorthand (resolves to per-track slug):")
    r2 = submit_to_kaggle(f"{WORK_DIR}/submissions/deep_dascoli_submission.csv",
                          competition="deep",
                          message="via pnpl helper (deep shorthand)", check=False)
    print("   result:", r2)
    print("   -> competition field:", r2.competition, "| success:", r2.success)
    return {"combined_success": bool(r1.success), "deep_slug": r2.competition}


@app.function(image=img, volumes=VOLUMES, secrets=[SECRET], timeout=15 * 60)
def leaderboard(slug: str = SLUG):
    _kaggle("competitions", "leaderboard", slug, "--show")
    return {"ok": True}


@app.function(image=img, volumes=VOLUMES, secrets=[SECRET], timeout=20 * 60)
def show_example(slug: str = SLUG):
    """Download + display the competition's example submission (expected format)."""
    import tempfile, glob, zipfile
    import pandas as pd
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi(); api.authenticate()
    d = tempfile.mkdtemp()
    api.competition_download_files(slug, path=d, quiet=True)
    for z in glob.glob(f"{d}/*.zip"):
        zipfile.ZipFile(z).extractall(d)
    for f in glob.glob(f"{d}/*.csv"):
        df = pd.read_csv(f)
        print(f"\n=== {f.split('/')[-1]}  shape={df.shape} ===")
        print("columns:", list(df.columns)[:8], "..." if df.shape[1] > 8 else "",
              "| last:", list(df.columns)[-3:])
        print("index range:", df.iloc[:, 0].min(), "..", df.iloc[:, 0].max())
        print(df.head(4).to_string()[:1000])
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
def submit(track: str = "test", message: str = "baseline", slug: str = SLUG,
           filename: str = ""):
    path = f"{WORK_DIR}/submissions/{filename or FILES[track]}"
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    from kaggle.api.kaggle_api_extended import KaggleApi
    import pandas as pd
    df = pd.read_csv(path)
    print(f"submitting {path}  shape={df.shape}  -> competition {slug}")
    api = KaggleApi(); api.authenticate()
    try:
        api.competition_submit(path, message, slug)
        print("submit: OK (accepted)")
        ok = True; err = ""
    except Exception as e:
        r = getattr(e, "response", None)
        err = (r.text if r is not None else str(e))[:800]
        print("submit FAILED:", err)
        ok = False
    return {"file": path, "slug": slug, "success": ok, "error": err}
