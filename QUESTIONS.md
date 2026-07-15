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

## Progress log

- ✅ **Deep (dascoli) DONE.** Final pipeline metrics on the 50-word vocab:
  **val BAcc@10 = 0.629, test BAcc@10 = 0.572** (macro; chance ≈ 0.20). Strong
  within-subject baseline (subject 0, Sherlock1, 1 s windows, contrastive t5
  retrieval). Checkpoints saved on the Modal volume. Now generating the deep-track
  submission (validating my holdout-window preprocessing reproduces ~0.57 first).
- ✅ **MEG-XL quick 2-subject/1-epoch validation confirms the full fine-tuning
  loop works** (train → per-epoch eval → BAcc@10 metric → checkpoint). 1 epoch is
  at chance (0.21) as expected — the word MLP is 321M randomly-init params and
  needs real training. Fix applied: `num_workers=0` (h5py open-handles + DataLoader
  fork deadlocked the first batch). Launching the full run (subjects 1–12, ~20
  epochs, isolated 1 s windows) next.
- ✅ **Broad (MEG-XL) fine-tuning pipeline runs end-to-end on Modal** (A100-80GB).
  Generated the Neuromag-306 sensor JSON from MNE (validated: 0 missing channels,
  0 mag/grad mismatches vs the LibriBrain h5), downloaded subjects 1–12 ses-11/12
  into the loader's layout, and the repo's hydra fine-tuning entry point loads the
  BioCodec tokenizer + pretrained MEG-XL checkpoint, finds the data, preprocesses
  to 50 Hz, computes t5 targets, and trains. Using isolated 1 s windows
  (words_per_segment=1, window_onset_offset=0) to match the competition holdout.
  Quick 2-subject/1-epoch validation run in progress; full run next.
- ✅ **Deep (dascoli) is training successfully on Modal** (L4 GPU). Full pipeline
  works end-to-end: pnpl data download → neuralset preprocessing (filter/resample/
  RobustScaler) → t5-large targets → SimpleConv+Transformer contrastive training.
  Pre-training chance baseline = 0.213 BAcc@10 (≈ random 0.20, as expected); by
  epoch 4 it's at **0.399 and still climbing**. 50 epochs w/ early stopping.
  Getting there required patching a few research-code quirks (offline-wandb
  save_dir, LazyModule + inference_mode, exca editable-package check) — all noted
  in the modal scripts.
- ✅ **Deep submission validated & generating.** Key finding from the val/test
  validation harness: the dascoli model's decoding quality lives entirely in the
  **sentence-context transformer** — the CNN branch alone (my first plan for
  isolated words) is *below chance* (0.17) with near-collapsed embeddings, and even
  the transformer on a single isolated word is at chance (0.19). The model needs
  the surrounding words. Luckily **~90% of holdout rows are `sentence`-source**
  (868/960 deep), so I reconstruct each sentence from the holdout npz and run the
  transformer over the whole sentence (context) — the ~10% isolated `word`-source
  rows get a length-1 pass. With this, my holdout-style preprocessing reproduces
  the pipeline exactly: **val 0.592 / test 0.567** BAcc@10 (pipeline: 0.629/0.572).
  Generating the deep submission now.

## Open questions / FYIs (broad / MEG-XL)

- 💡 **Broad training data is only subjects 1–12** (not 1–32). The LibriBrain2
  serialised h5 upload is incomplete: events.tsv exist for subjects 1–32 but the
  MEG `.h5` files only exist for **subjects 1–12** (13–32 return HTTP 404). So
  MEG-XL fine-tunes on subjects 1–12 (Sherlock1 ses-11 = train, ses-12 = test, per
  the paper's broad split — broad has no train partition). This is still a genuine
  cross-subject setup: the holdout evaluates subjects 1–39, so 13–39 are entirely
  unseen. → If more subjects get uploaded, I can re-run; flag if you expect them.
- 💡 **Sensor geometry for MEG-XL.** The model needs 3D sensor positions/orientations,
  but the pnpl h5 files carry only `channel_names` (standard Neuromag MEG0111…MEG2643)
  + `channel_types` (mag/grad), **no 3D geometry**, and no `meg_sensors_information.json`
  ships in the repos. I generate the JSON from MNE's standard Neuromag-306 device
  geometry — which is exactly what the pretrained MEG-XL expects, since its
  pretraining corpora (CamCAN/MOUS/SMN4Lang) are all Elekta Neuromag 306 systems
  with the same fixed helmet geometry. Positions are normalized + the backbone is
  fine-tuned, so this is faithful.
- ❓ Nothing blocking. Veto any decision above if you disagree.

---

## Answered

_(none yet)_
