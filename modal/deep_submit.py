"""
Deep track submission from a trained d'Ascoli checkpoint.

The model maps a MEG window -> a 1024-d t5-large embedding; we score by cosine
similarity against the t5-large embeddings of the 50 competition words and
softmax -> the primary probability row. Same for the moses-50 secondary block.

Steps:
  validate   (GPU)  run the trained model over pnpl val/test (subject 0), our own
                    holdout-style preprocessing, report BAcc@10 vs the pipeline.
  generate   (GPU)  run over the competition holdout (track=deep) -> submission CSV.

We reuse dascoli_image (has torch/mne/transformers/neuralset/pnpl).
"""

import os
import sys
import glob

import modal

from common import VOLUMES, WORK_DIR, work_vol
from deep_dascoli import dascoli_image, RUN_ENV, SAVEPATH

# The container needs these sibling modules mounted (Modal 1.x requires explicit
# local python source). dascoli_image already adds "common"; add the rest.
submit_image = dascoli_image
if modal.is_local():
    submit_image = submit_image.add_local_python_source("deep_dascoli", "deep_submit")

app = modal.App("pnpl-deep-submit")

# t5-large recipe (matches neuralset HuggingFaceText defaults used in training):
#   AutoModelForTextEncoding, add_special_tokens=False, mean over word tokens,
#   layer index int(0.5*25 - 1e-6) = 12 of the 25 hidden states.
T5_MODEL = "t5-large"
T5_LAYER = 12


def _normalize_word(w: str) -> str:
    """Replicate sentence_decoding.utils.preprocess_text: keep alnum + '-' + straight
    apostrophe, then lowercase (so curly 'it’s' -> 'its')."""
    return "".join(e for e in str(w) if e.isalnum() or e in ["-", "'"]).lower()


def _t5_embed(words, device):
    """Return (len(words), 1024) t5-large embeddings, exactly as training targets."""
    import torch
    from transformers import AutoTokenizer, AutoModelForTextEncoding

    tok = AutoTokenizer.from_pretrained(T5_MODEL, truncation_side="left")
    model = AutoModelForTextEncoding.from_pretrained(T5_MODEL).to(device).eval()
    embs = []
    with torch.no_grad():
        for w in words:
            inp = tok([w], add_special_tokens=False, return_tensors="pt",
                      padding=True, truncation=True).to(device)
            out = model(**inp, output_hidden_states=True)
            hs = torch.stack([l for l in out.hidden_states])  # (25,1,tokens,1024)
            layer = hs[T5_LAYER, 0]                            # (tokens,1024)
            embs.append(layer.mean(dim=0).cpu())               # mean over tokens
    return torch.stack(embs)  # (N,1024)


def _resample_windows(meg_np, sfreq_in=250.0, sfreq_out=50.0):
    """(B,306,250)@250 -> (B,306,~50)@50 via mne resample (anti-aliased). No scaling."""
    import numpy as np, mne
    mne.set_log_level("ERROR")
    B, C, T = meg_np.shape
    info = mne.create_info([f"MEG{i}" for i in range(C)], sfreq_in, "mag")
    out = []
    for b in range(B):
        raw = mne.io.RawArray(meg_np[b].astype(np.float64), info, verbose=False)
        raw.resample(sfreq_out, verbose=False)
        out.append(raw.get_data().astype(np.float32))
    return np.stack(out)  # (B,C,~50)


def _robust_stats(resampled):
    """GLOBAL per-channel RobustScaler stats (median, IQR) over ALL windows — the
    key fidelity point: training fits RobustScaler on the *whole recording*, so we
    aggregate all of a subject's windows rather than fitting per 1 s window."""
    import numpy as np
    C = resampled.shape[1]
    flat = resampled.transpose(1, 0, 2).reshape(C, -1)  # (C, B*T)
    median = np.median(flat, axis=1)
    q75, q25 = np.percentile(flat, [75, 25], axis=1)
    scale = (q75 - q25)
    scale[scale == 0] = 1.0
    return median.astype(np.float32), scale.astype(np.float32)


def _robust_stats_list(arrays):
    """GLOBAL per-channel RobustScaler stats over a list of (306, variable) arrays
    (concatenated over time) — used to pool a subject's whole holdout recording."""
    import numpy as np
    flat = np.concatenate(arrays, axis=1)  # (306, total_T)
    median = np.median(flat, axis=1)
    q75, q25 = np.percentile(flat, [75, 25], axis=1)
    scale = q75 - q25
    scale[scale == 0] = 1.0
    return median.astype(np.float32), scale.astype(np.float32)


def _apply_scaling(resampled, median, scale, baseline_s=0.5, sfreq_out=50.0, clamp=5.0):
    """RobustScaler(global) -> baseline subtract (first 0.5 s) -> clamp, per window."""
    import numpy as np
    d = (resampled - median[None, :, None]) / scale[None, :, None]
    n_base = int(round(baseline_s * sfreq_out))
    d = d - d[:, :, :n_base].mean(axis=2, keepdims=True)
    if clamp is not None:
        d = np.clip(d, -clamp, clamp)
    return d.astype(np.float32)


def _load_model(device):
    """Rebuild BrainModule from the best checkpoint on the work volume."""
    os.environ.update(RUN_ENV)
    sys.path.append("/root/dascoli")
    import torch
    from neuraltrain.utils import update_config
    from neuraltrain.models import SimpleConvTimeAggConfig, TransformerEncoderConfig
    from sentence_decoding.grids.libribrain100 import updated_config
    from sentence_decoding.main import Experiment
    from sentence_decoding.pl_module import BrainModule

    ckpts = sorted(glob.glob(f"{SAVEPATH}/results/**/best.ckpt", recursive=True))
    if not ckpts:
        ckpts = sorted(glob.glob(f"{SAVEPATH}/results/**/last.ckpt", recursive=True))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint under {SAVEPATH}/results")
    ckpt_path = ckpts[-1]
    print("Loading checkpoint:", ckpt_path)

    cfg = update_config(updated_config, {"infra.workdir": None, "data.duration": 1.0})
    brain_model = SimpleConvTimeAggConfig(**cfg["brain_model_config"]).build(
        n_in_channels=306, n_outputs=1024)
    transformer = TransformerEncoderConfig(**cfg["transformer_config"]).build(dim=1024)

    # Materialize lazy params with a dummy forward, then load state.
    dummy = torch.randn(2, 306, 50)
    brain_model(dummy, torch.zeros(2, dtype=torch.long), torch.rand(2, 306, 2))

    ck = torch.load(ckpt_path, map_location="cpu")
    state = ck["state_dict"] if "state_dict" in ck else ck
    module = BrainModule(model=brain_model, transformer=transformer,
                         loss=None, metrics={}, retrieval_metrics={},
                         trainer_config=cfg["trainer_config"])
    missing, unexpected = module.load_state_dict(state, strict=False)
    print(f"loaded state: {len(missing)} missing, {len(unexpected)} unexpected")
    module.to(device).eval()
    return module


def _channel_positions(device):
    import numpy as np, torch
    d = np.load(f"{SAVEPATH}/submission_assets.npz")
    cp = torch.from_numpy(d["channel_positions"]).float().to(device)  # (306,2)
    return cp


@app.function(image=submit_image, gpu="L4", volumes=VOLUMES, timeout=2 * 60 * 60)
def debug_pipeline():
    """DIAGNOSTIC: run the pipeline's own (correctly-preprocessed) val loader through
    the model; report retrieval BAcc@10 for the CNN branch AND the transformer using
    our t5 vocab embeddings. Isolates whether the submission gap is preprocessing,
    the CNN-vs-transformer branch, or our retrieval/embeddings."""
    os.environ.update(RUN_ENV)
    _ensure = None
    sys.path.append("/root/dascoli")
    import numpy as np, torch
    from neuraltrain.utils import update_config
    from sentence_decoding.grids.libribrain100 import updated_config
    from sentence_decoding.main import Experiment
    from pnpl.competition import PRIMARY_VOCAB

    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocab = [_normalize_word(w) for w in PRIMARY_VOCAB]
    vocab_to_id = {w: i for i, w in enumerate(vocab)}
    vocab_embs = _t5_embed(vocab, device).numpy()

    module = _load_model(device)

    rm = [m for m in updated_config["retrieval_metrics"] if m["name"] in ("Rank", "TopkAcc")]
    cfg = update_config(updated_config, {
        "infra.workdir": None, "data.duration": 1.0, "data.feature.device": "cuda",
        "data.num_workers": 4, "retrieval_metrics": rm})
    exp = Experiment(**cfg)
    loaders = exp.setup_run()
    val = loaders["val"]
    if isinstance(val, list):
        val = val[0]

    # compare saved channel_positions with the pipeline's
    sample = next(iter(val))
    cp_pipe = sample.data["channel_positions"][0].cpu().numpy()
    cp_saved = np.load(f"{SAVEPATH}/submission_assets.npz")["channel_positions"]
    print("cp shapes:", cp_pipe.shape, cp_saved.shape,
          "match:", np.allclose(cp_pipe, cp_saved), "range:", float(cp_pipe.min()), float(cp_pipe.max()))
    print("neuro shape from pipeline:", tuple(sample.data["neuro"].shape))

    cnn_e, tr_e, iso_e, labels = [], [], [], []
    with torch.no_grad():
        for batch in val:
            for k in list(batch.data.keys()):
                if torch.is_tensor(batch.data[k]):
                    batch.data[k] = batch.data[k].to(device)
            y_cnn = module.cnn_forward(batch)
            y_cnn = y_cnn / y_cnn.norm(dim=1, keepdim=True)
            y_tr = module.transformer_forward(batch, y_cnn)
            # isolated: each word as its own length-1 "sentence" (what the holdout
            # gives us — no sentence context)
            tr_in = y_cnn.unsqueeze(1)  # (B,1,1024)
            m = torch.ones(y_cnn.shape[0], 1, device=device).bool()
            y_iso = module.transformer(tr_in, mask=m).squeeze(1)
            y_iso = y_iso / y_iso.norm(dim=1, keepdim=True)
            for i, seg in enumerate(batch.segments):
                w = _normalize_word(seg._trigger["text"])
                if w in vocab_to_id:
                    cnn_e.append(y_cnn[i].cpu().numpy())
                    tr_e.append(y_tr[i].cpu().numpy())
                    iso_e.append(y_iso[i].cpu().numpy())
                    labels.append(vocab_to_id[w])
    labels = np.array(labels)
    print("val in-vocab:", len(labels))
    for name, embs in (("CNN", np.stack(cnn_e)), ("TRANSFORMER", np.stack(tr_e)),
                       ("TRANSFORMER-ISOLATED", np.stack(iso_e))):
        probs = _retrieval_probs(embs, vocab_embs)
        b10 = _bacc_topk(probs, labels, 10, 50)
        b1 = _bacc_topk(probs, labels, 1, 50)
        print(f"[pipeline-val {name}] BAcc@10={b10:.4f} BAcc@1={b1:.4f}  "
              f"emb_std_across_samples={float(np.stack(embs).std(0).mean()):.4f}")
    return {"ok": True}


def _predict_embeddings(module, cp, meg_windows_50, device, batch=256):
    """meg_windows_50: (N,306,50) preprocessed -> (N,1024) L2-normalized CNN embeddings."""
    import numpy as np, torch
    embs = []
    N = meg_windows_50.shape[0]
    with torch.no_grad():
        for i in range(0, N, batch):
            x = torch.from_numpy(meg_windows_50[i:i + batch]).float().to(device)
            bs = x.shape[0]
            sid = torch.zeros(bs, dtype=torch.long, device=device)
            cpb = cp.unsqueeze(0).expand(bs, -1, -1)
            y = module.model(x, sid, cpb)  # CNN branch -> (bs,1024)
            y = y / y.norm(dim=1, keepdim=True)
            embs.append(y.cpu())
    return torch.cat(embs).numpy()


def _retrieval_probs(pred_embs, vocab_embs, temperature=1.0):
    """cosine sim (vocab embs normalized by their norm, matching Rank norm_kind='y')
    -> softmax over vocab -> (N, V) probabilities."""
    import numpy as np
    pe = pred_embs / (np.linalg.norm(pred_embs, axis=1, keepdims=True) + 1e-9)
    ve = vocab_embs / (np.linalg.norm(vocab_embs, axis=1, keepdims=True) + 1e-9)
    sims = pe @ ve.T  # (N,V) cosine
    z = sims / temperature
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _bacc_topk(probs, labels, k, n_classes):
    import numpy as np
    topk = np.argsort(-probs, axis=1)[:, :k]
    hit = np.array([labels[i] in topk[i] for i in range(len(labels))], dtype=float)
    recalls = [hit[labels == c].mean() for c in range(n_classes) if (labels == c).any()]
    return float(np.mean(recalls)) if recalls else 0.0


@app.function(image=submit_image, gpu="L4", volumes=VOLUMES, timeout=3 * 60 * 60)
def validate(highpass: float = -1.0):
    """Run trained model on pnpl val+test (subject 0), our holdout-style preprocessing,
    report BAcc@10/@1 to compare against the training pipeline's reported numbers."""
    os.environ.update(RUN_ENV)
    sys.path.append("/root/dascoli")
    import numpy as np, torch
    from pnpl.datasets import LibriBrainWord
    from pnpl.competition import PRIMARY_VOCAB

    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocab = [_normalize_word(w) for w in PRIMARY_VOCAB]
    vocab_to_id = {w: i for i, w in enumerate(vocab)}
    vocab_embs = _t5_embed(vocab, device).numpy()

    module = _load_model(device)
    cp = _channel_positions(device)

    from collections import defaultdict
    data_path = f"{RUN_ENV['DATAPATH']}/libribrain_word"
    for partition in ("validation", "test"):
        ds = LibriBrainWord(data_path=data_path, partition=partition, tmin=0.0, tmax=1.0,
                            standardize=False, include_info=False, preload_files=False)
        # ALL words (context needs non-vocab words too); samples: (subj,ses,task,run,
        # onset, word, sent_idx, word_idx)
        megs = np.stack([ds[i][0].numpy() for i in range(len(ds))])  # (N,306,250)
        sent_ids = [s[6] for s in ds.samples]
        word_idx = [s[7] for s in ds.samples]
        words = [_normalize_word(s[5]) for s in ds.samples]
        # our holdout-style preprocessing, then CNN, then transformer per sentence
        resamp = _resample_windows(megs)
        med, scl = _robust_stats(resamp)
        pre = _apply_scaling(resamp, med, scl)
        cnn = _cnn_embed(module, cp, pre, device)  # (N,1024) on device
        embs = [None] * len(megs)
        groups = defaultdict(list)
        for i, sid in enumerate(sent_ids):
            groups[sid].append(i)
        for sid, idxs in groups.items():
            idxs = sorted(idxs, key=lambda i: word_idx[i])
            c = cnn[idxs]
            m = torch.ones(1, len(idxs), device=device).bool()
            tr = module.transformer(c.unsqueeze(0), mask=m).squeeze(0)
            tr = tr / tr.norm(dim=1, keepdim=True)
            for j, i in enumerate(idxs):
                embs[i] = tr[j].detach().cpu().numpy()
        labels, pe = [], []
        for i, w in enumerate(words):
            if w in vocab_to_id:
                labels.append(vocab_to_id[w]); pe.append(embs[i])
        labels = np.array(labels)
        probs = _retrieval_probs(np.stack(pe), vocab_embs)
        b10 = _bacc_topk(probs, labels, 10, 50)
        b1 = _bacc_topk(probs, labels, 1, 50)
        print(f"[{partition}] in-vocab={len(labels)}  BAcc@10={b10:.4f}  BAcc@1={b1:.4f}"
              f"  (random ~0.20 / ~0.02)")
    return {"ok": True}


def _cnn_embed(module, cp, windows_np, device, batch=256):
    """(nw,306,~50) -> (nw,1024) L2-normalized CNN embeddings (on device)."""
    import torch
    embs = []
    with torch.no_grad():
        for i in range(0, len(windows_np), batch):
            x = torch.from_numpy(windows_np[i:i + batch]).float().to(device)
            bs = x.shape[0]
            sid = torch.zeros(bs, dtype=torch.long, device=device)
            cpb = cp.unsqueeze(0).expand(bs, -1, -1)
            y = module.model(x, sid, cpb)
            embs.append(y / y.norm(dim=1, keepdim=True))
    return torch.cat(embs)


@app.function(image=submit_image, gpu="L4", volumes=VOLUMES, timeout=4 * 60 * 60)
def generate(track: str = "deep", temperature: float = 0.5):
    """Sentence-aware submission. The dascoli model's decoding quality comes from the
    transformer refining each word using its SENTENCE context, so for the ~90% of
    holdout rows from the `sentence` source we reconstruct each sentence from the
    holdout npz, run CNN->transformer over the whole sentence, and read off each
    word's refined embedding. The ~10% isolated `word`-source rows get a length-1
    transformer pass (no context available)."""
    os.environ.update(RUN_ENV)
    sys.path.append("/root/dascoli")
    import numpy as np, torch
    from pnpl.competition import (LibriBrainCompetitionHoldout, write_submission,
                                  PRIMARY_VOCAB, SECONDARY_VOCAB)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    prim = [_normalize_word(w) for w in PRIMARY_VOCAB]
    sec = [_normalize_word(w) for w in SECONDARY_VOCAB]
    prim_embs = _t5_embed(prim, device).numpy()
    sec_embs = _t5_embed(sec, device).numpy()

    module = _load_model(device)
    cp = _channel_positions(device)

    holdout = LibriBrainCompetitionHoldout(track=track)
    print("holdout:", repr(holdout), holdout.counts())
    idx_of = {(m["subject"], m["source"], m["epoch"], m["word"]): m["index"]
              for m in holdout.metadata}
    N = len(holdout)
    primary = np.zeros((N, len(prim)), dtype=np.float32)
    secondary = np.zeros((N, len(sec)), dtype=np.float32)

    def _emit(emb, ix):
        e = emb.detach().cpu().numpy()[None]
        primary[ix] = _retrieval_probs(e, prim_embs, temperature)[0]
        secondary[ix] = _retrieval_probs(e, sec_embs, temperature)[0]

    for subj in holdout.subjects:
        # ---- sentence source: reconstruct each sentence, run transformer w/ context
        sp = holdout._ensure_file(subj, "sentence")
        with np.load(sp, allow_pickle=True) as d:
            meg = np.asarray(d["meg"], dtype=np.float32)          # (Ns,306,T)
            n_times = np.asarray(d["sentence_n_times"]).astype(int)
            onsets = np.asarray(d["word_onsets_s"], dtype=np.float64)
            wmask = np.asarray(d["word_mask"])
        # pass 1: resample each sentence, gather global RobustScaler stats
        sent_r = [_resample_windows(meg[si:si + 1, :, :n_times[si]])[0]
                  for si in range(meg.shape[0])]
        med, scl = _robust_stats_list(sent_r)
        # pass 2: per sentence -> words -> CNN -> transformer (context)
        for si, r in enumerate(sent_r):
            rs = (r - med[:, None]) / scl[:, None]
            valid = np.nonzero(wmask[si])[0]
            wins = []
            for wi in valid:
                st = int(round(float(onsets[si, wi]) * 50.0))
                w = rs[:, st:st + 50]
                w = w - w[:, :25].mean(axis=1, keepdims=True)
                wins.append(np.clip(w, -5, 5).astype(np.float32))
            if not wins:
                continue
            cnn = _cnn_embed(module, cp, np.stack(wins), device)      # (nw,1024)
            m = torch.ones(1, cnn.shape[0], device=device).bool()
            tr = module.transformer(cnn.unsqueeze(0), mask=m).squeeze(0)  # (nw,1024)
            tr = tr / tr.norm(dim=1, keepdim=True)
            for j, wi in enumerate(valid):
                _emit(tr[j], idx_of[(subj, "sentence", si, int(wi))])
        # ---- word source: isolated length-1 transformer
        wp = holdout._ensure_file(subj, "word")
        with np.load(wp, allow_pickle=True) as d:
            wmeg = np.asarray(d["meg"], dtype=np.float32)  # (Nw,306,S)
        wr = _resample_windows(wmeg)
        medw, sclw = _robust_stats(wr)
        wpre = _apply_scaling(wr, medw, sclw)
        cnn = _cnn_embed(module, cp, wpre, device)         # (Nw,1024)
        m = torch.ones(cnn.shape[0], 1, device=device).bool()
        tr = module.transformer(cnn.unsqueeze(1), mask=m).squeeze(1)  # (Nw,1024)
        tr = tr / tr.norm(dim=1, keepdim=True)
        for ei in range(wmeg.shape[0]):
            _emit(tr[ei], idx_of[(subj, "word", ei, -1)])
        print(f"  subject {subj}: sentence+word done")

    out_dir = f"{WORK_DIR}/submissions"
    os.makedirs(out_dir, exist_ok=True)
    out = write_submission(f"{out_dir}/{track}_dascoli_submission.csv",
                           indices=holdout.indices, primary_probs=primary,
                           secondary_probs=secondary)
    work_vol.commit()
    am = primary.argmax(1)
    uniq, cnt = np.unique(am, return_counts=True)
    print("wrote:", out, "shape:", primary.shape, "distinct argmax:", len(uniq))
    return {"path": str(out), "n": int(N)}
