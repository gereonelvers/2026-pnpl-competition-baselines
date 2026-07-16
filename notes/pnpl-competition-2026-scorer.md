# Test competition `pnpl-competition-2026` + a scorer bug

This is the third, "testing" Kaggle competition
(<https://www.kaggle.com/competitions/pnpl-competition-2026/>). Its evaluation **is**
wired up (unlike `-deep` / `-broad`, which currently reject *all* submissions —
including their own example CSVs — with *"An Evaluation system has not been configured
for this competition"*). It is scored by the notebook `balanced-acccuracy.ipynb` against
the colleague's `solution.csv`.

## Submission format (from the notebook, not their 816-byte example)

- First column must be named **`id`** (not `index`).
- **51 or 101 columns**: `id` + 50 primary-vocab probs (vocabulary.csv order) + optional
  50 `moses_<word>` probs.
- All numeric; every value except `id` in `[0, 1]`.
- **Row count must equal the solution's** — one row per `solution.csv` row (34 720; `id`
  0–34 719). Solution format is `id, label(word), Usage`.

Because there is a single solution spanning all 40 subjects, one **combined** submission
covers both tracks: deep-model probs for subject 0 (868 rows), broad-model for subjects
1–39 (33 852 rows). Built by `score_solution.py::build_test_submission`.

## Submissions made (both accepted)

| Submission | File | Notebook score (full) | Public LB |
|------------|------|-----------------------|-----------|
| Combined baseline (deep+broad) | `test_pnpl2026_submission.csv` | 0.172 | **0.17435** |
| Random noise (uniform per-row) | `noise_pnpl2026_submission.csv` | 0.191 | ~0.19 |

## ⚠️ The scorer is broken — a perfect submission scores at chance

`score_solution.py::diagnose_scorer` scores synthetic submissions whose answer we know:

| Submission | Notebook score | A correct scorer should give |
|------------|----------------|------------------------------|
| **Oracle** (p=1 on the true word) | **0.20** | 1.0 |
| Oracle shifted +1 | 0.20 | — |
| Oracle shifted −1 | 0.00 | — |
| Inverted oracle | 0.18 | ~0 |

A **perfect** submission scores 0.20 (= chance = 10/50), and the random baseline (0.19)
actually **outscores** the real model (0.174). The metric does not reward correct
predictions. Two bugs in `balanced-acccuracy.ipynb`:

```python
# after dropping moses, `submission` still contains the id column:
submission = submission[[col for col in submission.columns if 'moses' not in col]]
submission_np = submission_filtered.to_numpy()             # cols = [id, p_is, ..., p_on]
submission_np = np.argsort(submission_np, axis=1)[:, :10]  # BUG 1 + BUG 2
```

1. **Wrong direction.** `np.argsort(...)[:, :10]` takes the 10 **smallest** values →
   the 10 *lowest*-probability words. Top-10 needs the largest: `argsort(-x)[:, :10]`
   (or `argpartition(-x, 10)`).
2. **`id` column shifts everything by one.** `id` stays in the array during `argsort`,
   so column indices are `0=id, 1=is, …, 50=on` while `label_encoded` is `0=is, …,
   49=on`. Even with the direction fixed, index `c` never lines up with the word at code
   `c`. Drop `id` (and moses) *before* `argsort`.

### Suggested fix

```python
target_classes = [...]                      # 50 words, vocabulary.csv order
probs = submission[target_classes].to_numpy()          # drop id + moses first
top10 = np.argsort(-probs, axis=1)[:, :10]             # 10 HIGHEST-prob classes
solution['label_encoded'] = pd.Categorical(
    solution['label'].str.lower(), categories=target_classes, ordered=True).codes
correct = (top10 == solution['label_encoded'].to_numpy()[:, None]).any(axis=1)
# ... then the existing per-class balanced-accuracy averaging is fine.
```

With this fix the oracle scores 1.0, random ≈ 0.20, and the combined baseline should land
around its true holdout BAcc@10 (deep ≈ 0.47 on subject 0, broad ≈ 0.31 on 1–39).

## Reproduce

```bash
modal run score_solution.py::build_test_submission      # -> test_pnpl2026_submission.csv
modal run score_solution.py::build_noise_submission     # -> noise_pnpl2026_submission.csv
modal run score_solution.py::score_with_notebook --sub-name test_pnpl2026_submission.csv
modal run score_solution.py::diagnose_scorer            # oracle/inverted sanity checks
modal run kaggle_submit.py::submit --track test --slug pnpl-competition-2026 \
    --filename test_pnpl2026_submission.csv --message "combined baseline"
```

Auth: the modern `KGAT_` token needs kaggle>=2.0 (Python>=3.11), so this runs on Modal;
the token is stored in the Modal secret `pnpl-kaggle-token` (never in code/git).
