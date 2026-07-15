# MEG-XL "Broad" Baseline — Architecture Analysis

> Deep-dive of `baselines/MEG-XL` for the cross-subject (broad) track.
> **Critical:** MEG-XL word decoding is a **retrieval** model (predicts a
> 1024-dim T5 word embedding, ranks candidate words by cosine similarity), NOT a
> softmax classifier. There is a **window/rate mismatch** (train: 3 s @ 50 Hz;
> holdout: 1 s @ 250 Hz) and the **sensor geometry** must be reconstructed from a
> fixed LibriBrain sensor JSON.

## 1. Architecture
Pipeline: `raw MEG [B,C,T] → BioCodec tokenizer (frozen) → RVQ code embeddings →
sensor/RoPE embeddings → CrissCross transformer → features → word MLP → 1024-dim embedding`.

- **BioCodec tokenizer (frozen)**: `neuro_tokenizers/biocodec/model.py`, built via
  `BioCodecModel._get_optimized_model()`. SEANet encoder + RVQ. Downsample 12×;
  `n_q=6`, `bins=256`, `codebook_dim=16`. Operates 1 channel at a time
  (`[B,C,T]→[B*C,1,T]`). Output codes `[B,C,Q=6,T'=T/12]`.
  **Checkpoint `brainstorm/neuro_tokenizers/biocodec_ckpt.pt` (~38 MB) is present in the repo.**
- **CrissCross transformer** (`models/criss_cross_transformer.py`,
  `CrissCrossTransformerModule`): RVQ codes → codebook embeddings (`6*16=96`) →
  `Linear(96,512)`; + Gaussian-Fourier sensor position emb (xyz), + orientation
  emb (abc), + sensor-type embedding (2). `SpatialTemporalEncoder(dim=512,
  depth=8, heads=8)`; each block: half temporal self-attn w/ RoPE, half spatial
  self-attn. Uses PyTorch SDPA (**no flash-attn/xformers**). RoPE buffers are
  recomputed, so `rope_embedding_layer.rotate` keys are filtered on load.
  - `forward(raw_meg, sensor_xyz, sensor_abc, sensor_type, sensor_mask, apply_mask=False)`
    → dict with `features: [B, C, T', 512]` (used by word head).
- **Word head** `CrissCrossWordEmbeddingExtractor`
  (`evaluate_criss_cross_word_classification.py:291`): slice features over the
  word's encoded-time range, mean-pool over time → `[C,512]`, flatten
  (`306*512=156672`), optional subject-FiLM, MLP `156672→2048→1024`. Output:
  **1024-dim predicted T5 embedding**.

## 2. Pretrained checkpoints
- **BioCodec** — present in repo (`biocodec_ckpt.pt`, 38 MB).
- **MEG-XL transformer** — download from HF `pnpl/MEG-XL`. Files:
  `meg-xl-med-v2.ckpt`, `meg-xl-med.ckpt` (~283 MB each). Lightning ckpt with
  `hyper_parameters` + `state_dict`. Load with `CrissCrossTransformerModule(
  tokenizer, **hparams)` then `load_state_dict(strict=False)` after filtering
  `rope_embedding_layer.rotate` keys. Override config path via
  `model.criss_cross_checkpoint=...`.

## 3. Data / sampling / windowing
- Loader `LibriBrain100WordAlignedDataset`
  (`data/libribrain100_word_aligned_dataset.py`): HDF5 recordings, task-first
  layout `<task>/derivatives/{events, serialised[,serialised_competition]}`.
- **Model runs at 50 Hz** (`l_freq=0.1, h_freq=40, target_sfreq=50`). MNE
  `RawArray.filter(0.1,40).resample(50)`. **Holdout is 250 Hz → must resample to
  50 Hz → (306,50).**
- **Train window = 3 s @ 50 Hz** (`subsegment_duration=3.0,
  window_onset_offset=-0.5` ⇒ `[onset-0.5, onset+2.5]` = 150 samples).
  `words_per_segment=50`, concatenated → 150 s segment → BioCodec → `T'=625` →
  features `[B,306,625,512]`. **Holdout 1 s window ≠ 3 s train window** (domain shift).
- Per-window preprocessing (`preprocessing.py:_process_single_chunk`): baseline
  correct (first 0.5 s), `sklearn.RobustScaler` per sensor-type group (mag/grad
  separately), clip `(-5,5)`.
- Batch dict: `meg [B,306,7500]`, `word_labels [B*50]`, `subsegment_info`,
  `word_metadata`, `sensor_xyzdir [B,306,6]`, `sensor_types [B,306]`,
  `sensor_mask [B,306]`.

## 4. Fine-tuning
```
python -m brainstorm.evaluate_criss_cross_word_classification \
  --config-name=eval_criss_cross_word_classification_libribrain100_multisub_train \
  model.criss_cross_checkpoint=/path/to/megxl.ckpt
```
- Multisub config: `subjects=sub-1..sub-32` (subj0 excluded), `tasks=[Sherlock1]`.
  Splits by content: Sherlock1 ses-11 first half→train, ses-11 second half→val,
  ses-12→test, other sessions→train. Genuinely cross-subject.
- AdamW, differential LR (backbone `1e-5`, word MLP `1e-3`), wd `1e-4`,
  `ReduceLROnPlateau(max, factor .5, patience 5)`. Loss: **SigLipLoss**
  (contrastive, pred vs T5 target). `batch_size=1`, `num_epochs=50`, `patience=10`,
  grad clip 1.0. bf16 autocast. Best → `logs/.../checkpoint_best.pt` (plain torch
  dict: `criss_cross_state_dict`, `word_mlp_state_dict`, ...).
- `use_gradient_checkpointing=false` in config; enable to cut memory. README: **≥80 GB
  VRAM** with default segments → A100-80GB or H100. 40 GB only for linear-probe /
  gradient-checkpointing + shorter segments.

## 5. Vocabulary — retrieval, not softmax
- Vocab = **all unique words** (freq-ordered), not 50. "50" appears only at eval as
  retrieval sets (`datafit50`, `moses50`, freq-50/250).
- **Target embeddings**: `generate_word_embeddings()` runs **T5-large layer 12,
  mean-pooled** → `[vocab, 1024]`, cached. Needs `transformers` + `t5-large`.
- **For submission**: cosine-sim predicted 1024-dim emb vs T5 embeddings of the
  **competition's own 50 words** → softmax → 50-way probs.
- Subject FiLM: per-train-subject embedding; `subject_idx=-1` → mean embedding.
  Recommend `subject_idx=-1` (or disable FiLM) for broad, since 33–39 unseen.

## 6. Sensor geometry reconstruction (CRITICAL)
Holdout `(306,250)` has no sensor metadata → reconstruct once, reuse for all:
- `meg_sensors_information.json` parsed by `load_libribrain_sensors`
  (`preprocessing.py:12`). Per sensor: pos=`loc[0:3]`; orient=`loc[9:12]` (mag,
  `coil_type==3024`) or `loc[3:6]` (grad, `3012`); type=1 mag / 0 grad. Fallback:
  h5-embedded `sensor_xyzdir`/`sensor_types`; else `get_sensor_positions()` from
  MNE `info` (`ch['loc']`, `coil_type`).
- Normalization `norm_sensor_positions` (`data/utils.py`): center xyz by
  channel-mean, scale by `sqrt(3*mean(sum(xyz^2)))`; orientation left raw. Pad to
  306, `sensor_mask=ones`.
- **Single fixed Neuromag 306-ch layout, shared across all subjects/windows.**
  Channel ORDER must match the holdout array rows (pnpl ordering) — must verify.

## 7. Dependencies
Python ≥3.12. `torch>=2.0`, `pytorch-lightning>=2.0`, `hydra-core`, `omegaconf`,
`numpy`, `h5py`, `mne>=1.4`, `pandas`, `scikit-learn`, `wandb`, `tqdm`,
`einops`, `transformers>=5.0` (t5-large), `ovmi` (git, optional/guarded).
No flash-attn/xformers/deepspeed. bf16 → Ampere+ (A100/H100).

## 8. Submission integration
Reuse (all in `evaluate_criss_cross_word_classification.py`):
- `load_tokenizer(biocodec_ckpt, device)`.
- Build backbone + load `criss_cross_state_dict`; word head
  `CrissCrossWordEmbeddingExtractor(...)` + `word_mlp_state_dict`.
- `generate_word_embeddings(comp_50_words, layer=12)` → `[50,1024]`.
Per holdout window `(306,250)@250Hz`: resample→50Hz (306,50); baseline/RobustScaler/clip;
reconstruct sensor tensors; `features = model(...apply_mask=False)['features']`;
`word_emb = word_mlp(features[0], subject_idx=-1)`; cosine-sim vs 50 T5 embs → softmax.

## 9. Risks
1. **Window/rate mismatch** (top): train 3s@50Hz vs holdout 1s@250Hz. Mitigation:
   fine-tune with `subsegment_duration≈1.0` to match holdout.
2. Resampling 250→50 Hz mandatory & must be consistent.
3. Preprocessing parity (baseline/RobustScaler/clip) — confirm what pnpl holdout applies.
4. Sensor geometry JSON availability + exact channel ordering vs pnpl.
5. Retrieval-not-classification: need t5-large + exact comp vocab.
6. Pretrained ckpt from HF (283 MB, confirmed present).
7. GPU: ≥80 GB → A100-80GB / H100. Inference is cheap.
8. Cross-subject: subjects 33–39 unseen (FiLM mean emb).
9. Hardcoded config paths (`/data/engs-asr/...`) must be overridden.
10. wandb.init unconditional → set `WANDB_MODE=offline`. OVMI optional.
