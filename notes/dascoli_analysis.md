# d'Ascoli "Deep" Baseline — Architecture Analysis

> Deep-dive of `baselines/dascoli-word-decoding` for the within-subject (deep) track.
> **Key:** a contrastive brain→text-embedding model. MEG window → 1024-dim
> `t5-large` embedding; BAcc@10 = cosine-ranking of the predicted embedding
> against the 50 vocab words' `t5-large` embeddings. Same retrieval scoring idea
> as MEG-XL.

## 1. Architecture (SimpleConvTimeAgg + Transformer)
`MEG (B,306,T) → SimpleConvTimeAgg → (B,1024) → Transformer (sentence-grouped) → (B,1024)`,
both unit-normalized; contrastive (SigLIP) loss vs frozen `t5-large` word embeddings.
- **SimpleConvTimeAgg** (`neuraltrain/models/simpleconv.py`): `forward(x, subject_ids,
  channel_positions)`. ChannelMerger `(306→270)` using Fourier emb of
  `channel_positions` + per-subject attention → `initial_linear Conv1d(270→512)`
  → SubjectLayers (per-subject 512→512) → 5 residual dilated conv blocks
  (hidden 160, GLU) emitting 1024 ch → Bahdanau **time-attention pooling** →
  `(B,1024)`. Time-attention ⇒ tolerates variable T (1 s vs 3 s).
- **TransformerEncoder** (`neuraltrain/models/transformer.py`, x_transformers
  `Encoder`, depth 16, heads 16, dim 1024, rotary). Groups CNN outputs into
  sentences (`sequence_id`+`timeline`), refines each word emb with sentence
  context, un-groups, unit-normalizes.
- Config in `sentence_decoding/grids/defaults.py` + `grids/libribrain100.py`.

## 2. Data flow
- pnpl → neuralset adapter `neuralset/studies/libribrain100.py`: iterates
  `pnpl.datasets.libribrain100.RUN_RECORDS`, uses `pnpl.datasets.LibriBrain100(...,
  task=WordClassification(tmin=0,tmax=0.5), standardize=False, download=True)`.
- **Default split (Deep)**: subj `LibriBrain100/0`, Sherlock1 sessions 1–10 train,
  11 val, 12 test (split taken from PNPL-provided `partition` column).
- **Window**: `start=0.0, duration=3.0` ⇒ `[onset, onset+3s]`.
- **MEG preprocessing** (`neuralset/features/neuro.py`), per recording (cached):
  pick meg (306), **band-pass 0.1–40 Hz**, **resample 250→50 Hz**, **RobustScaler
  per channel** (median/IQR, fit on whole recording). Per window: slice →
  `(306,150)`, **baseline** subtract mean over [0,0.5]s, **clamp ±5**.
- **channel_positions** `(306,2)`: normalized 2-D coords via `mne.find_layout`;
  `-0.1` for channels not in layout. Constant across windows.
- **Target text emb** (`neuralset/features/text.py`): `HuggingFaceText(t5-large,
  aggregation=trigger, layers=0.5, add_special_tokens=False, token_agg=mean,
  contextualized=False)` → `t5-large` encoder, **mean over tokens**, **layer 12 of
  25**, → 1024-dim per word.
- Batch dict: `neuro (B,306,150)`, `feature (B,1024)`, `subject_id (B,)`,
  `channel_positions (B,306,2)`.

## 3. Training
- Entry: `python -m sentence_decoding.grids.libribrain100`. Uses
  `neuraltrain.utils.run_grid` → local `LocalJob` runner (⚠ caches results incl.
  failures; a too-fast `Done.` can mask a cached/failed job).
- AdamW `lr=1e-4`, wd 0, **CosineAnnealingLR** `T_max=n_epochs`, `n_epochs=50`,
  `patience=10`, `batch_size=128`. Loss **SigLIP** (`norm_kind=y`, learnable
  temp/bias). EarlyStopping + ModelCheckpoint monitor
  `val_retrieval_acc10_vocab=libribrain50_macro_0` (= BAcc@10).
- ~8–16 GB GPU, a few hours. First run downloads LibriBrain + t5-large (~3 GB) +
  caches embeddings.

## 4. Metric → 50-way distribution (CRITICAL)
`TestRetrieval` callback (`sentence_decoding/callbacks.py`): cosine-sim
(`Rank._compute_sim`, norm_kind=y) of predicted (unit-norm) emb vs candidate word
embeddings; rank of true word; `TopkAcc` macro-average over words = BAcc@10.
- **50-word vocab** (`grids/libribrain100.py`): `is,the,a,to,it,i,not,was,...,as,on`
  (matches `pnpl` vocabulary.csv). Embeddings computed with the **same** t5-large
  recipe (layer 12, mean tokens, no special tokens).
- **Submission**: `e = model(window)`, L2-normalize, `s_k = cos(e, w_k)` for the 50
  vocab embeddings, `probs = softmax(s_k / T)`. BAcc@10 depends only on ranking,
  so T only shapes magnitudes.

## 5. Dependencies / env
Python 3.11, torch 2.3.1. Editable installs `neuralset[dev]` + `neuraltrain`.
Deps: pandas, numpy≥2, **mne≥1.4**, scikit-learn, **exca**, pydantic≥2.5,
**torchmetrics**, **braindecode @ git**, lightning, wandb, **x_transformers**,
**kenlm** (import-time in `decoder.py` even though unused), **transformers**,
**sentencepiece**, h5py, submitit, cloudpickle, scipy. t5-large downloaded on
first use. **W&B**: keep `use_wandb=True` + `WANDB_MODE=offline` (disabling it
sets logger=None which breaks `TestRetrieval` save path).
Env vars: `SAVEPATH`, `DATAPATH`, `LIBRIBRAIN100_PATH`.

## 6. Submission integration
- Load: rebuild `brain_model = SimpleConvTimeAggConfig.build(n_in_channels=306,
  n_outputs=1024)`, `transformer = TransformerEncoderConfig.build(dim=1024)`, then
  `BrainModule.load_from_checkpoint(best.ckpt, strict=False, model=..., ...)`.
  Ckpts at `$SAVEPATH/results/sentence_decoding/libribrain100/<uid>/best.ckpt`.
- Preprocess raw `(306,250)@250Hz` holdout window to match training: band-pass
  0.1–40, resample→50Hz, RobustScaler per channel, baseline [0,0.5]s, clamp ±5.
- `channel_positions (306,2)` built once from LibriBrain montage (same order as
  pnpl); `subject_ids=0` (deep).
- Forward CNN branch (**option A, robust for isolated words**) → 1024 emb →
  cosine vs 50 vocab embs → softmax. (Option C: for `sentence`-source rows, group
  a sentence's words and run the real transformer with context — higher fidelity,
  more work.)

## 7. Risks
1. **Train/holdout window+rate mismatch** (top): train 3s@50Hz vs holdout
   1s@250Hz. Mitigation: train with `duration=1.0` to match holdout, OR accept
   domain shift (attention pooling tolerates length). → **Decision: match 1 s.**
2. **RobustScaler** fit per-recording in training but only isolated epochs at test
   → fit per-epoch (approx). Moves scores; validate.
3. **channel_positions / channel order** must match training montage + pnpl order.
4. Keep `WANDB_MODE=offline` (don't set use_wandb=False).
5. **kenlm** import-time dep — must be installed.
6. Local task-runner caches failures; sub-infras (StudyLoader, Meg, text emb)
   cache under `$SAVEPATH/cache/...` — clear if data-prep fails midway.
7. Hardcoded `/Users/hans/...` paths in README; set env vars + mkdir.
8. Offline retrieval set = observed words only; for submission rank against ALL 50.

## 8. Key files
Grid: `sentence_decoding/grids/{libribrain100,defaults}.py`. Train:
`sentence_decoding/{main,pl_module}.py`. Metric: `sentence_decoding/callbacks.py`,
`neuraltrain/metrics/metrics.py`. Model: `neuraltrain/models/{simpleconv,transformer,common}.py`.
Loss: `neuraltrain/losses/losses.py`. Data: `neuralset/studies/libribrain100.py`,
`neuralset/features/{neuro,text}.py`.
