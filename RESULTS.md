# Results

Both baseline architectures were trained in full on Modal GPUs and produced valid,
Kaggle-format submissions. Numbers below are **Top-10 Balanced Accuracy (BAcc@10)** —
the competition's primary metric — with BAcc@1 as the tie-break.

## Scored on a real holdout label file

Scored against a colleague's holdout label file (`runner/solution.csv`: word-level
ground truth for the **`sentence`-source** examples of all 40 holdout subjects, 868
words/subject). Each track is scored on its own subjects (deep = subject 0, broad =
subjects 1–39) using the *official* Kaggle scorer's metric (macro recall@k over the
50-word competition vocabulary, in-vocab rows only). The solution↔submission join is on
`(subject, sentence_epoch, word_onset)` and matched **100%** of rows.

| Track | Baseline | Subjects | Submission | Rows scored | **BAcc@10** | BAcc@1 | Chance BAcc@10 | Chance BAcc@1 | Above chance |
|-------|----------|----------|------------|-------------|-------------|--------|----------------|---------------|--------------|
| **Deep** (within-subject) | d'Ascoli | 0 | `submissions/deep_dascoli_submission.csv` | 451 | **0.467** | 0.072 | 0.20 | 0.02 | ✅ ~2.3× |
| **Broad** (cross-subject) | MEG-XL | 1–39 | `submissions/broad_megxl_submission.csv` | 17 589 | **0.308** | 0.033 | 0.20 | 0.02 | ✅ ~1.5× |

**Chance** for a 50-word top-k, macro-averaged, is `k / 50` regardless of class balance:
**0.20** at k=10 and **0.02** at k=1. Confirmed empirically on these exact rows by
averaging 20 random-score submissions (deep 0.195 ± 0.025 — noisier on 451 rows; broad
0.200 ± 0.004). Both submissions clear it comfortably.

Notes:
- The label file covers only the `sentence`-source rows (~90% of the holdout), not the
  ~10% isolated `word`-source rows. That's where our submissions are strongest (they use
  reconstructed sentence context), so a full-holdout score would be marginally lower.
- Deep's 0.467 here (real holdout, 451 in-vocab rows) is lower than the 0.567 measured on
  the Sherlock1 **test** split — expected for held-out data + a small slice. Broad's 0.308
  matches the 0.322 validation it trained to — a good cross-check that the submission is sound.

## Training-time validation (for reference)

Measured on each pipeline's own val/test split before the holdout existed:

| Track | val BAcc@10 | test BAcc@10 |
|-------|-------------|--------------|
| Deep (dascoli) | 0.629 | 0.567 |
| Broad (MEG-XL) | 0.322 | — |

---

# Reproduction

Everything runs on [Modal](https://modal.com) (nothing heavy touches the local disk).
Persistent Modal volumes hold the data/checkpoints (`pnpl-hf-cache`, `pnpl-work`); the
final submission CSVs land on `pnpl-work` under `/submissions/`.

## 0. Setup

```bash
pip install modal
modal token set --token-id <id> --token-secret <secret>
cd runner/modal          # all commands below are run from here
```

The Modal images copy three local source trees into the container at build time:
`../../pnpl`, `../../baselines/dascoli-word-decoding`, `../../baselines/MEG-XL`
(so keep the repo layout intact). We apply a few small, documented patches to the
baseline source at image-build time — see `deep_dascoli.py` / `broad_megxl.py`
(`run_commands(... sed ...)`) and the `[patch]`-tagged edits in the MEG-XL eval script.

## 1. Deep track — d'Ascoli (within-subject, subject 0)

```bash
modal run deep_dascoli.py::smoke_imports                                  # sanity: imports + model forward
modal run deep_dascoli.py::run_training --n-epochs 50 --duration 1.0 --batch-size 128
modal run deep_submit.py::validate                                        # reproduces val/test BAcc@10
modal run deep_submit.py::generate --track deep                           # -> /submissions/deep_dascoli_submission.csv
```

The model maps a MEG window → a 1024-d `t5-large` embedding (contrastive/SigLIP). Its
decoding quality comes from the sentence-context transformer, so the submission
reconstructs each holdout sentence and runs the transformer with context.

## 2. Broad track — MEG-XL (cross-subject, subjects 1–39)

```bash
modal run broad_megxl.py::smoke                                           # loads BioCodec + pretrained MEG-XL
modal run broad_megxl.py::make_sensor_json                                # Neuromag-306 sensor geometry (from MNE)
modal run broad_megxl.py::download_data --subjects 32                     # subjects 1–32
modal run --detach broad_megxl.py::finetune \
    --subjects 32 --num-epochs 15 --batch-size 4 \
    --words-per-segment 20 --subsegment-duration 1.0                      # ~20 s context segments; resume-safe
modal run broad_submit.py::generate --track broad                         # -> /submissions/broad_megxl_submission.csv
```

MEG-XL's criss-cross transformer needs temporal context (pretrained on long segments),
so we fine-tune on ~20 s multi-word segments and reconstruct each holdout sentence at
inference. `--detach` + `resume_checkpoint` make the run survive Modal preemption. Watch
progress with `modal app logs <app-id>`; inspect the best checkpoint with
`modal run broad_megxl.py::inspect_ckpt --which best`.

## 3. Fetch the submissions locally

```bash
modal volume get pnpl-work /submissions/deep_dascoli_submission.csv ../submissions/
modal volume get pnpl-work /submissions/broad_megxl_submission.csv  ../submissions/
```

## 4. Score against a holdout label file

Given a solution CSV with per-subject rows (`subj_id, sentence_epoch_index,
word_onset_s, label, …`):

```bash
modal volume put pnpl-work solution.csv /solution.csv --force
modal run score_solution.py::score       # prints BAcc@10/@1 + chance for both tracks
```

The scorer joins solution rows to the canonical holdout flat index on
`(subject, sentence_epoch, onset)`, converts word labels → 50-word vocab ids, keeps
in-vocab rows, and computes BAcc@k exactly like the official Kaggle scorer.

## Uploading to Kaggle (optional)

```python
from pnpl.competition import submit_to_kaggle
submit_to_kaggle("deep_dascoli_submission.csv",  competition="pnpl-competition-2026-deep")
submit_to_kaggle("broad_megxl_submission.csv",   competition="pnpl-competition-2026-broad")
```
