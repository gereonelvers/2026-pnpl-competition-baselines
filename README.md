# PNPL 2026 Competition — Baseline Submissions

Orchestration code + notes for producing baseline submissions for the two
tracks of the PNPL / LibriBrain 2026 word-decoding competition, trained on
[Modal](https://modal.com) GPUs.

## Tracks & baselines

| Track | Subjects | Baseline | Approach |
|-------|----------|----------|----------|
| **Deep** (within-subject) | subject 0 | `dascoli-word-decoding` | Contrastive brain→text-embedding (d'Ascoli et al. 2025) + retrieval over 50-word vocab |
| **Broad** (cross-subject) | subjects 1–39 | `MEG-XL` | Pre-trained MEG foundation model, fine-tuned for 50-way word classification |

## Task

For each holdout word window `(306 channels, 250 samples)` @ 250 Hz, output a
probability distribution over the 50-word competition vocabulary. Scored with
**Top-10 Balanced Accuracy** (BAcc@10); random ≈ 0.20.

Submissions are built with `pnpl.competition.LibriBrainCompetitionHoldout` +
`write_submission`.

## Layout

```
modal/        Modal apps: data prep, training, submission generation
scripts/      Local helper scripts (submission validation, scoring)
notes/        Architecture deep-dive notes for each baseline
submissions/  Generated submission CSVs (committed — they are small)
QUESTIONS.md  Async Q&A / decision log
```

## Results

Both baselines trained on Modal and produced valid, Kaggle-format submissions.
**Chance BAcc@10 = 0.20** (50-word top-10, macro-averaged), chance BAcc@1 = 0.02.

Scored on a real holdout label file (`solution.csv`), both are well above chance:

| Track | Baseline | Submission | **Holdout BAcc@10** | (train val / test) |
|-------|----------|------------|---------------------|--------------------|
| Deep | dascoli | `submissions/deep_dascoli_submission.csv` (960 rows) | **0.467** (~2.3× chance) | 0.629 / 0.567 |
| Broad | MEG-XL | `submissions/broad_megxl_submission.csv` (37 439 rows) | **0.308** (~1.5× chance) | 0.322 / — |

See **[`RESULTS.md`](RESULTS.md)** for the full scored table (incl. empirical chance
baselines) and **step-by-step reproduction instructions**. Method notes are in `notes/`;
the running decision log is in `QUESTIONS.md`.

### Key methodological finding

Both baselines are **retrieval models** (predict a 1024-d `t5-large` word embedding,
rank by cosine similarity) whose decoding quality depends on **sentence context**:

- **dascoli**: the sentence-context transformer does the work — the CNN branch alone
  is *below* chance on isolated words.
- **MEG-XL**: its criss-cross transformer was pretrained on long (~625-step) segments,
  so isolated 1 s windows (~5 steps) fail; it needs multi-word context.

Since ~90 % of holdout rows are `sentence`-source, the submissions **reconstruct each
sentence** and give the model its context; the ~10 % isolated `word`-source rows are
handled without it. Holdout preprocessing was validated to reproduce each pipeline's
own val/test BAcc@10 before trusting the holdout.

## Status

Deep track complete. Broad track: a valid submission is in place (val 0.322); the
fine-tune may still be running to a natural stop — the submission is regenerated if a
later epoch beats the current best. See `QUESTIONS.md` for the running log.
