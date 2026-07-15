# Open Questions

This file is where I (Claude) log open questions and decisions for Gereon to
answer/review asynchronously. I will keep working and make reasonable default
choices where I can, noting them here. Newest questions at the top.

**Status legend:** ❓ open · ✅ answered · 💡 decision I made (proceeding unless
you object)

---

## Context / what I'm doing

Goal: produce valid baseline submissions for both competition tracks by running
the two provided baseline architectures faithfully (not toy runs), on Modal GPUs
(budget < $100):

- **Deep track** (within-subject, subject 0): `baselines/dascoli-word-decoding`
  — a contrastive brain→text-embedding model (d'Ascoli et al. 2025).
- **Broad track** (cross-subject, subjects 1–39): `baselines/MEG-XL` — a
  pre-trained MEG foundation model, fine-tuned for word classification.

Submission = for each holdout word window `(306, 250)` @ 250 Hz, a probability
distribution over the 50-word competition vocabulary. Metric: Top-10 Balanced
Accuracy (BAcc@10); random ≈ 0.20.

---

## Decisions I'm making by default (override if you disagree)

- 💡 **Data availability confirmed.** MEG-XL checkpoints exist on HF
  (`pnpl/MEG-XL`: `meg-xl-med-v2.ckpt`, `meg-xl-med.ckpt`, ~283 MB each). Holdout
  is 5.68 GB. Sherlock1 downsampled `.h5` is ~4.5 GB. All downloadable.
- 💡 I'll use `meg-xl-med-v2.ckpt` (the newer v2) as the MEG-XL pretrained
  backbone unless you tell me otherwise.
- 💡 For the "moses" secondary columns in the submission, I'll fill them with a
  reasonable distribution too (either a second retrieval head or uniform) since
  they're not scored on the primary leaderboard. Priority is the primary 50-word
  distribution.

---

## Open questions

- ❓ **Nothing blocking yet.** Still reading the two baseline architectures in
  detail (subagents running). Will populate this section with concrete questions
  about training scope (how much data = "faithful full run"), GPU choice, and any
  code adaptation needed once I understand the baselines fully.

---

## Answered

_(none yet)_
