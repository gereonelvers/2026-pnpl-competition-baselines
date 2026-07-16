"""
Score our deep + broad submissions against a colleague's label file (uploaded to
the work volume at /work/solution.csv). The solution has per-subject rows with word
labels + rich metadata; we join to the canonical holdout flat index via
(subject, sentence_epoch, onset), convert words -> vocab ids, keep in-vocab rows,
and compute BAcc@10/@1 exactly as the official Kaggle scorer does (macro recall@k
over the classes present).
"""

import modal
from common import pnpl_cpu_image, VOLUMES, WORK_DIR, work_vol

app = modal.App("pnpl-score-solution")


def _norm(w):
    return "".join(e for e in str(w) if e.isalnum() or e in ["-", "'"]).lower()


def _bacc(probs, y_true, k):
    import numpy as np
    kk = min(k, probs.shape[1])
    topk = np.argpartition(-probs, kth=kk - 1, axis=1)[:, :kk]
    hit = (topk == y_true[:, None]).any(axis=1).astype(float)
    recalls = [hit[y_true == c].mean() for c in np.unique(y_true)]
    return float(np.mean(recalls))


@app.function(image=pnpl_cpu_image, volumes=VOLUMES, timeout=40 * 60)
def build_test_submission(out_name: str = "test_pnpl2026_submission.csv"):
    """Build ONE combined submission matching solution.csv (the pnpl-competition-2026
    solution, per balanced-acccuracy.ipynb): id + 50 primary + 50 moses probs, one row
    per solution row. Deep-model probs for subject 0, broad-model for subjects 1-39."""
    import numpy as np, pandas as pd
    from pnpl.competition import LibriBrainCompetitionHoldout, load_vocabulary

    vocab = list(load_vocabulary("primary"))
    moses_cols = [f"moses_{w}" for w in load_vocabulary("moses")]
    prob_cols = vocab + moses_cols                       # 100 columns

    sol = pd.read_csv("/work/solution.csv")
    luts, subs = {}, {}
    for track in ("deep", "broad"):
        ho = LibriBrainCompetitionHoldout(track=track)
        luts[track] = {(m["subject"], m["epoch"], round(float(m["onset_s"]), 4)): m["index"]
                       for m in ho.metadata if m["source"] == "sentence"}
        fn = "deep_dascoli" if track == "deep" else "broad_megxl"
        subs[track] = pd.read_csv(f"/work/submissions/{fn}_submission.csv").set_index("index")

    out = np.full((len(sol), len(prob_cols)), 1.0 / len(vocab), dtype=float)
    n_miss = 0
    for i, r in enumerate(sol.itertuples()):
        subj = int(r.subj_id)
        track = "deep" if subj == 0 else "broad"
        fi = luts[track].get((subj, int(r.sentence_epoch_index), round(float(r.word_onset_s), 4)))
        if fi is None:
            n_miss += 1
            continue
        out[i] = subs[track].loc[fi, prob_cols].to_numpy(dtype=float)

    df = pd.DataFrame(out, columns=prob_cols)
    df.insert(0, "id", sol["id"].to_numpy())
    path = f"{WORK_DIR}/submissions/{out_name}"
    df.to_csv(path, index=False)
    work_vol.commit()
    valid01 = bool(((df.iloc[:, 1:] >= 0) & (df.iloc[:, 1:] <= 1)).all().all())
    print(f"wrote {path}  shape={df.shape}  misses={n_miss}")
    print(f"first col={df.columns[0]!r}  n_cols={df.shape[1]}  values_in_[0,1]={valid01}")
    print(f"id range {df['id'].min()}..{df['id'].max()}  subjects: deep={int((sol['subj_id']==0).sum())} broad={int((sol['subj_id']!=0).sum())}")
    return {"rows": len(df), "cols": df.shape[1], "misses": n_miss, "valid01": valid01}


@app.function(image=pnpl_cpu_image, volumes=VOLUMES, timeout=20 * 60)
def build_noise_submission(out_name: str = "noise_pnpl2026_submission.csv", seed: int = 0):
    """A public-facing 'random' baseline: uniform-random probability distributions
    (per row, per vocab block) over the same rows as solution.csv. Correct format
    (id + 50 primary + 50 moses, values in [0,1]) so it submits cleanly."""
    import numpy as np, pandas as pd
    from pnpl.competition import load_vocabulary

    vocab = list(load_vocabulary("primary"))
    moses_cols = [f"moses_{w}" for w in load_vocabulary("moses")]
    sol = pd.read_csv("/work/solution.csv")
    rng = np.random.default_rng(seed)
    prim = rng.random((len(sol), len(vocab)));       prim /= prim.sum(1, keepdims=True)
    mos = rng.random((len(sol), len(moses_cols)));    mos /= mos.sum(1, keepdims=True)
    df = pd.DataFrame(np.hstack([prim, mos]), columns=vocab + moses_cols)
    df.insert(0, "id", sol["id"].to_numpy())
    path = f"{WORK_DIR}/submissions/{out_name}"
    df.to_csv(path, index=False)
    work_vol.commit()
    print(f"wrote {path}  shape={df.shape}  seed={seed}")
    return {"rows": len(df), "cols": df.shape[1]}


@app.function(image=pnpl_cpu_image, volumes=VOLUMES, timeout=20 * 60)
def score_with_notebook(sub_name: str = "test_pnpl2026_submission.csv"):
    """Run balanced-acccuracy.ipynb's score() VERBATIM on our built submission vs
    solution.csv — confirms the format passes its checks and shows the score it returns."""
    import numpy as np, pandas as pd
    import pandas.api.types

    def score(solution, submission, row_id_column_name):
        if row_id_column_name != "test":
            if submission.shape[1] != 51 and submission.shape[1] != 101:
                raise ValueError("columns must be 51 or 101")
        if solution.shape[0] != submission.shape[0]:
            raise ValueError("row count mismatch")
        if submission.columns[0].lower() != 'id':
            raise ValueError("first column must be id")
        for col in submission.columns:
            if not pd.api.types.is_numeric_dtype(submission[col]):
                raise ValueError(f"{col} must be numeric")
        if not ((submission.iloc[:, 1:] >= 0) & (submission.iloc[:, 1:] <= 1)).all().all():
            raise ValueError("values not in [0,1]")
        target_classes = ['is','the','a','to','it','i','not','was','we','be','he','that','have','this','they','of',
                          'there','and','are','in','but','will','so','all','my','for','she','were','any','really',
                          'at','out','our','am','its','had','him','an','very','has','do','can','time','think','good',
                          'always','new','people','as','on']
        submission = submission[[c for c in submission.columns if 'moses' not in c]]
        solution['label'] = solution['label'].str.lower()
        filt = solution['label'].isin(target_classes)
        solution_f = solution[filt]; submission_f = submission[filt]
        sub_np = submission_f.to_numpy()
        sub_np = np.argsort(sub_np, axis=1)[:, :10]
        solution_f = solution_f.copy()
        solution_f['label_encoded'] = pd.Categorical(solution_f['label'], categories=target_classes, ordered=True).codes
        le = solution_f['label_encoded'].to_numpy()
        correct = (sub_np == le[:, None]).any(axis=1)
        cpc = np.bincount(le, minlength=len(target_classes))
        cor = np.bincount(le, weights=correct.astype(int), minlength=len(target_classes))
        mask = cpc != 0
        return float(np.sum(cor[mask] / cpc[mask]) / len(target_classes))

    sol = pd.read_csv("/work/solution.csv")[["id", "label"]].sort_values("id").reset_index(drop=True)
    sub = pd.read_csv(f"{WORK_DIR}/submissions/{sub_name}").sort_values("id").reset_index(drop=True)
    s = score(sol.copy(), sub.copy(), "id")
    print(f"notebook score() on {sub_name}: {s}")
    return {"notebook_score": s, "rows": len(sub), "cols": sub.shape[1]}


@app.function(image=pnpl_cpu_image, volumes=VOLUMES, timeout=15 * 60)
def oracle_debug():
    import numpy as np, pandas as pd
    from pnpl.competition import load_vocabulary
    target = ['is','the','a','to','it','i','not','was','we','be','he','that','have','this','they','of',
              'there','and','are','in','but','will','so','all','my','for','she','were','any','really',
              'at','out','our','am','its','had','him','an','very','has','do','can','time','think','good',
              'always','new','people','as','on']
    vocab = list(load_vocabulary("primary"))
    print("vocab == target_classes (same order):", vocab == target)
    print("set diff vocab-target:", set(vocab) - set(target), "| target-vocab:", set(target) - set(vocab))
    mism = [(i, vocab[i], target[i]) for i in range(50) if vocab[i] != target[i]]
    print("position mismatches:", mism[:10])
    sol = pd.read_csv("/work/solution.csv")
    lab = sol["label"].str.lower()
    v2i = {w: i for i, w in enumerate(vocab)}
    unmapped = sorted(set(lab[lab.map(v2i).isna()]))
    print("labels not in vocab (sample):", unmapped[:15], "| n_unmapped_rows:", int(lab.map(v2i).isna().sum()))
    in_t = lab.isin(target)
    print("in-target rows:", int(in_t.sum()), "| distinct in-target words:", lab[in_t].nunique())
    return {"vocab_eq_target": vocab == target}


@app.function(image=pnpl_cpu_image, volumes=VOLUMES, timeout=20 * 60)
def verify_fixed_notebook(nb_name: str = "balanced-accuracy-fixed.ipynb"):
    """Load the ACTUAL corrected notebook from the volume, exec its score(), and run it
    on the real submissions + a real oracle. Expect: oracle 1.0, noise ~0.20, and the
    per-track numbers to match our independent scorer (deep ~0.47 / broad ~0.31)."""
    import json, numpy as np, pandas as pd
    from pnpl.competition import load_vocabulary

    nb = json.load(open(f"{WORK_DIR}/{nb_name}"))
    src = nb["cells"][0]["source"]
    if isinstance(src, list):
        src = "".join(src)
    ns = {}
    exec(src, ns)
    score = ns["score"]

    vocab = list(load_vocabulary("primary"))
    moses_cols = [f"moses_{w}" for w in load_vocabulary("moses")]
    sol_meta = pd.read_csv("/work/solution.csv")
    sol = sol_meta[["id", "label"]]
    deep_ids = set(sol_meta.loc[sol_meta["subj_id"] == 0, "id"])
    broad_ids = set(sol_meta.loc[sol_meta["subj_id"] != 0, "id"])

    def run(sub, subset=None):
        s, u = sol, sub
        if subset is not None:
            s = sol[sol["id"].isin(subset)]
            u = sub[sub["id"].isin(subset)]
        s = s.sort_values("id").reset_index(drop=True)
        u = u.sort_values("id").reset_index(drop=True)
        return round(score(s.copy(), u.copy(), "id"), 4)

    test = pd.read_csv(f"{WORK_DIR}/submissions/test_pnpl2026_submission.csv")
    noise = pd.read_csv(f"{WORK_DIR}/submissions/noise_pnpl2026_submission.csv")

    # Real oracle: p=1.0 on the true word (0 elsewhere), matching solution.csv rows
    lab = sol_meta["label"].str.lower()
    v2i = {w: i for i, w in enumerate(vocab)}
    oi = lab.map(v2i)  # NaN if OOV
    P = np.zeros((len(sol_meta), len(vocab)))
    ok = oi.notna().to_numpy()
    P[np.arange(len(sol_meta))[ok], oi[ok].astype(int).to_numpy()] = 1.0
    oracle = pd.DataFrame(P, columns=vocab)
    for c in moses_cols:
        oracle[c] = 0.0
    oracle.insert(0, "id", sol_meta["id"].to_numpy())

    print("=== fixed notebook on real data ===")
    print("  oracle (p=1 on true word)   :", run(oracle), "  (expect 1.0)")
    print("  noise baseline              :", run(noise), "  (expect ~0.20)")
    print("  combined baseline (all rows):", run(test))
    print("  combined -> deep rows only  :", run(test, deep_ids), "  (indep scorer: 0.4672)")
    print("  combined -> broad rows only :", run(test, broad_ids), " (indep scorer: 0.3077)")
    return {"ok": True}


@app.function(image=pnpl_cpu_image, volumes=VOLUMES, timeout=20 * 60)
def diagnose_scorer():
    """Sanity-check balanced-acccuracy.ipynb by scoring synthetic submissions whose
    answer we KNOW: a perfect oracle (p=1 on the true word) MUST score ~1.0 under a
    correct scorer. Also test oracle shifted by +/-1 (off-by-one) and an inverted
    oracle (p=1 on the true word, then 1-p) to characterise what it actually rewards."""
    import numpy as np, pandas as pd
    import pandas.api.types

    target_classes = ['is','the','a','to','it','i','not','was','we','be','he','that','have','this','they','of',
                      'there','and','are','in','but','will','so','all','my','for','she','were','any','really',
                      'at','out','our','am','its','had','him','an','very','has','do','can','time','think','good',
                      'always','new','people','as','on']

    def score(solution, submission, row_id_column_name="id"):
        submission = submission[[c for c in submission.columns if 'moses' not in c]]
        solution = solution.copy()
        solution['label'] = solution['label'].str.lower()
        filt = solution['label'].isin(target_classes)
        sol_f = solution[filt].copy(); sub_f = submission[filt]
        sub_np = np.argsort(sub_f.to_numpy(), axis=1)[:, :10]
        sol_f['le'] = pd.Categorical(sol_f['label'], categories=target_classes, ordered=True).codes
        le = sol_f['le'].to_numpy()
        correct = (sub_np == le[:, None]).any(axis=1)
        cpc = np.bincount(le, minlength=len(target_classes))
        cor = np.bincount(le, weights=correct.astype(int), minlength=len(target_classes))
        m = cpc != 0
        return float(np.sum(cor[m] / cpc[m]) / len(target_classes))

    sol = pd.read_csv("/work/solution.csv")[["id", "label"]].sort_values("id").reset_index(drop=True)
    lab = sol['label'].str.lower()
    code = pd.Categorical(lab, categories=target_classes, ordered=True).codes  # -1 if OOV
    N, V = len(sol), len(target_classes)

    def make(col_for_row, invert=False):
        # col_for_row[i] = which of the 50 primary columns gets prob 1.0 for row i (or -1 = none)
        p = np.zeros((N, V), dtype=float)
        ok = col_for_row >= 0
        p[np.arange(N)[ok], col_for_row[ok]] = 1.0
        if invert:
            p = 1.0 - p
        df = pd.DataFrame(p, columns=target_classes)
        df.insert(0, "id", sol["id"].to_numpy())
        return df

    results = {
        "oracle (p=1 on true word)":            score(sol, make(code)),
        "oracle shifted +1":                    score(sol, make(np.where(code >= 0, (code + 1) % V, -1))),
        "oracle shifted -1":                    score(sol, make(np.where(code >= 0, (code - 1) % V, -1))),
        "inverted oracle (1 - onehot_true)":    score(sol, make(code, invert=True)),
    }
    for k, v in results.items():
        print(f"  {k:38s}: {round(v, 4)}")
    print("  (a CORRECT scorer would give oracle=1.0, inverted≈0; chance=0.20)")
    return results


@app.function(image=pnpl_cpu_image, volumes=VOLUMES, timeout=40 * 60)
def score():
    import numpy as np, pandas as pd
    from pnpl.competition import LibriBrainCompetitionHoldout, load_vocabulary

    vocab = load_vocabulary("primary")               # 50 words, column order
    vnorm = [_norm(w) for w in vocab]
    v2id = {w: i for i, w in enumerate(vnorm)}

    sol = pd.read_csv("/work/solution.csv")
    sol["labelid"] = sol["label"].map(lambda w: v2id.get(_norm(w)))
    print("solution rows:", len(sol), "| in-vocab rows:", int(sol['labelid'].notna().sum()),
          "| distinct subjects:", sol['subj_id'].nunique())

    out = {}
    for track, subj_set in (("deep", [0]), ("broad", list(range(1, 40)))):
        ho = LibriBrainCompetitionHoldout(track=track)
        # lookup: (subject, sentence-epoch, onset) -> canonical flat index
        lut = {(m["subject"], m["epoch"], round(float(m["onset_s"]), 4)): m["index"]
               for m in ho.metadata if m["source"] == "sentence"}
        n_sent = len(lut)

        fn = "deep_dascoli" if track == "deep" else "broad_megxl"
        sub = pd.read_csv(f"/work/submissions/{fn}_submission.csv").set_index("index")
        probs_all = sub[vocab]  # 50 primary cols

        s = sol[sol["subj_id"].isin(subj_set)].copy()
        s["flat"] = [lut.get((int(r.subj_id), int(r.sentence_epoch_index),
                               round(float(r.word_onset_s), 4)))
                     for r in s.itertuples()]
        match_rate = float(s["flat"].notna().mean())

        keep = s.dropna(subset=["flat", "labelid"])
        y = keep["labelid"].astype(int).to_numpy()
        probs = probs_all.loc[keep["flat"].astype(int).to_numpy()].to_numpy(dtype=float)

        # Empirical chance on THESE exact rows: average BAcc over random-score
        # submissions (theory: k/V = 10/50 = 0.20 for BAcc@10, 1/50 = 0.02 for BAcc@1,
        # independent of class balance since it's macro-averaged).
        rng = np.random.default_rng(0)
        c10 = [_bacc(rng.random((len(y), len(vocab))), y, 10) for _ in range(20)]
        c1 = [_bacc(rng.random((len(y), len(vocab))), y, 1) for _ in range(20)]

        res = {
            "solution_rows_for_track": len(s),
            "sentence_index_entries": n_sent,
            "join_match_rate": round(match_rate, 4),
            "n_scored_in_vocab": len(keep),
            "n_classes_present": int(len(np.unique(y))),
            "BAcc@10": round(_bacc(probs, y, 10), 4),
            "BAcc@1": round(_bacc(probs, y, 1), 4),
            "chance_BAcc@10": round(float(np.mean(c10)), 4),
            "chance_BAcc@10_std": round(float(np.std(c10)), 4),
            "chance_BAcc@1": round(float(np.mean(c1)), 4),
            "theoretical_chance_BAcc@10": round(10 / len(vocab), 4),
            "theoretical_chance_BAcc@1": round(1 / len(vocab), 4),
        }
        print(f"\n=== {track.upper()} ({fn}_submission.csv) ===")
        for k, v in res.items():
            print(f"  {k}: {v}")
        out[track] = res
    return out
