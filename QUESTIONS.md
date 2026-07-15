# Open Questions

This file is where I (Claude) log open questions and decisions for Gereon to
answer/review asynchronously. I keep working and make reasonable default choices,
noting them here. Newest at the top.

**Status legend:** ❓ open · ✅ answered · 💡 decision I made (proceeding unless
you object)

---

## Context / what I'm doing

Goal: produce valid baseline submissions for both competition tracks by running
the two provided baseline architectures faithfully (not toy runs), on Modal GPUs
(budget < $100):

- **Deep track** (within-subject, subject 0): `baselines/dascoli-word-decoding`
  — contrastive brain→text-embedding model (d'Ascoli et al. 2025).
- **Broad track** (cross-subject, subjects 1–39): `baselines/MEG-XL` —
  pre-trained MEG foundation model, fine-tuned for word decoding.

Submission = for each holdout word window `(306, 250)` @ 250 Hz, a probability
distribution over the 50-word competition vocabulary. Metric: Top-10 Balanced
Accuracy (BAcc@10); random ≈ 0.20.

---

## Key finding (both baselines)

Both baselines are **retrieval models**, NOT softmax classifiers. Each maps a MEG
window → a **1024-dim `t5-large` word embedding**; BAcc@10 is computed by ranking
the predicted embedding (cosine similarity) against the `t5-large` embeddings of
the candidate vocab words. So for the submission I:
1. run the model to get a predicted 1024-d embedding per holdout window,
2. cosine-sim it against the 50 competition words' `t5-large` embeddings,
3. softmax → the 50-way probability row.
BAcc@10 depends only on the ranking, so this exactly reproduces each baseline's
own offline metric. Same recipe for the moses-50 secondary columns.

---

## Decisions I'm making by default (override in this file if you disagree)

- 💡 **Data availability confirmed.** MEG-XL checkpoints exist on HF
  (`pnpl/MEG-XL`: `meg-xl-med-v2.ckpt`, `meg-xl-med.ckpt`, ~283 MB each). Holdout
  5.68 GB, Sherlock1 downsampled `.h5` ~4.5 GB. BioCodec tokenizer (38 MB) ships
  in the MEG-XL repo. All downloadable. Using `meg-xl-med-v2.ckpt` as backbone.
- 💡 **Run the baselines' real pipelines** (dascoli `run_grid`; MEG-XL hydra
  `evaluate_criss_cross_word_classification`), adapting only paths/config, rather
  than reimplementing. Fall back to a faithful reimplementation (their model +
  loss classes in a clean loop) only if their infra proves too fragile to run on
  Modal within budget.
- 💡 **Window matching (important).** Both baselines were built for **3 s @ 50 Hz**
  word windows, but the competition holdout is scored on **1 s @ 250 Hz** windows.
  To avoid a train/test domain shift (and, for MEG-XL, to cut GPU memory), I will
  **train each baseline with a 1 s window** (resampled to the baseline's native
  50 Hz), keeping the rest of the architecture/recipe faithful. The tutorial's own
  baseline also trains directly on the 1 s @ 250 Hz competition window.
  → If you'd rather I keep the literal 3 s recipe and accept the domain shift at
  inference, say so here.
- 💡 **Holdout preprocessing parity.** I replicate each baseline's preprocessing on
  the raw holdout windows (band-pass 0.1–40 Hz, resample→50 Hz, RobustScaler,
  baseline, clamp). RobustScaler is fit per-recording in training but I only have
  isolated epochs at inference, so I fit it per-epoch (an approximation). I
  validate the whole submission path by reproducing each baseline's val/test
  BAcc@10 before trusting the holdout numbers.
- 💡 **dascoli submission head.** Use the CNN branch output (well-defined for
  isolated words) rather than the sentence-context transformer; validated on
  val/test. (The transformer needs sentence grouping that isolated holdout word
  windows don't have.)
- 💡 **Moses secondary columns.** Filled with the same model's retrieval over the
  moses-50 words (free given the predicted embedding). Not scored on primary LB.

---

## Open questions

- ❓ **Nothing blocking yet.** Proceeding with the deep (dascoli) baseline first
  (cheaper GPU, ~8–16 GB), then broad (MEG-XL, needs A100-80GB/H100). Will add
  concrete questions here if I hit a real fork. Check the decisions above and veto
  any you disagree with.

---

## Answered

_(none yet)_
