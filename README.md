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

## Status

Work in progress. See `QUESTIONS.md` for the running log.
