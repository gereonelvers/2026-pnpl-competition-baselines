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

## Corrected notebook — `balanced-accuracy-fixed.ipynb` (verified)

A drop-in replacement is in the repo root: `runner/balanced-accuracy-fixed.ipynb` (same
structure + format checks as the original; only the two scoring lines changed). Verified
by `score_solution.py::verify_fixed_notebook`, loading the actual notebook file and
scoring known-answer submissions:

| Submission | Broken notebook | **Fixed notebook** | Expected |
|------------|-----------------|--------------------|----------|
| Oracle (p=1 on true word) | 0.20 | **1.00** | 1.0 |
| Random noise | 0.19 | **0.2043** | ~0.20 |
| Combined baseline (all 34 720 rows) | **0.17435** | **0.3105** | above chance |
| &nbsp;&nbsp;↳ deep rows only (subject 0) | — | **0.4672** | 0.4672 (matches our independent scorer) |
| &nbsp;&nbsp;↳ broad rows only (subj 1–39) | — | **0.3065** | ≈0.3077 (matches our independent scorer) |

The oracle scores exactly 1.0 and the per-track numbers reproduce our independent scorer,
which is the strongest confirmation the fix is correct. The doctest still returns `0.06`.
(Broad is 0.3065 vs 0.3077 because the fixed notebook *additionally* normalises `it's`/`its`
spelling and so correctly scores a few rows the stricter matching had dropped.)

## Third issue (handled): `it's` vs `its` — class list drift

`load_vocabulary("primary")[34]` / `vocabulary.csv` is **`it's`** (curly apostrophe ’, =
"it is"); the *original* notebook hardcoded **`its`** (possessive) there. Same position, so
scoring happened to be correct for a solution that also uses `its` — but `target_classes`
also *filters* solution rows, so a solution using the `vocabulary.csv` spelling `it's`
would silently drop that class and cap everyone near 49/50.

The fixed notebook removes this drift two ways:

1. **Resolve the class list from `vocabulary.csv`** when the scoring environment provides
   it (`_resolve_target_classes`), falling back to an embedded copy in vocabulary order.
   Best-effort + exception-safe, so a missing file never breaks scoring. (Note:
   `vocabulary.csv` is **not** currently one of the competition's data files — only
   `example-submission.csv` is — so today it uses the embedded fallback. Adding
   `vocabulary.csv` to the competition data would make the notebook track it automatically.)
2. **Normalise both sides** (`_norm`: lower-case + strip all non-alphanumerics) before
   matching, so `it's` / `its` / `It's` compare equal. This is what actually guarantees the
   metric, `vocabulary.csv`, and the solution can't drift on punctuation/case — verified
   above (oracle with `it's` columns + `its` labels → 1.0).

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
