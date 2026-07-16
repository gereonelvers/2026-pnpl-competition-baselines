"""
Score our deep + broad submissions against a colleague's label file (uploaded to
the work volume at /work/solution.csv). The solution has per-subject rows with word
labels + rich metadata; we join to the canonical holdout flat index via
(subject, sentence_epoch, onset), convert words -> vocab ids, keep in-vocab rows,
and compute BAcc@10/@1 exactly as the official Kaggle scorer does (macro recall@k
over the classes present).
"""

import modal
from common import pnpl_cpu_image, VOLUMES

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
