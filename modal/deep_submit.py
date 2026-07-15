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


def _preprocess_windows(meg_np, highpass=None, lowpass=40.0, sfreq_in=250.0,
                        sfreq_out=50.0, baseline_s=0.5, clamp=5.0):
    """(B,306,250)@250Hz raw -> (B,306,50)@50Hz, matching training preprocessing:
    band-pass 0.1-40, resample to 50, RobustScaler per channel, baseline subtract
    (first 0.5 s), clamp +/-5. RobustScaler is fit per epoch (we only have isolated
    windows), an approximation of the per-recording fit used in training."""
    import numpy as np
    import mne
    from sklearn.preprocessing import RobustScaler

    mne.set_log_level("ERROR")
    B, C, T = meg_np.shape
    info = mne.create_info(ch_names=[f"MEG{i}" for i in range(C)],
                           sfreq=sfreq_in, ch_types="mag")
    out = []
    for b in range(B):
        x = meg_np[b].astype(np.float64)  # (C,T)
        raw = mne.io.RawArray(x, info, verbose=False)
        if highpass is not None or lowpass is not None:
            raw.filter(highpass, lowpass, fir_design="firwin",
                       verbose=False, pad="reflect_limited")
        raw.resample(sfreq_out, verbose=False)
        d = raw.get_data()  # (C, T2)
        # RobustScaler per channel (fit on this epoch)
        d = RobustScaler().fit_transform(d.T).T
        # baseline: subtract per-channel mean over first baseline_s seconds
        n_base = int(round(baseline_s * sfreq_out))
        d = d - d[:, :n_base].mean(axis=1, keepdims=True)
        if clamp is not None:
            d = np.clip(d, -clamp, clamp)
        out.append(d.astype(np.float32))
    return np.stack(out)  # (B,C,T2)


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

    data_path = f"{RUN_ENV['DATAPATH']}/libribrain_word"
    for partition in ("validation", "test"):
        ds = LibriBrainWord(data_path=data_path, partition=partition, tmin=0.0, tmax=1.0,
                            standardize=False, include_info=True, preload_files=False)
        megs, labels = [], []
        for i in range(len(ds)):
            meg, _lab, info = ds[i]
            w = _normalize_word(info["word"])
            if w in vocab_to_id:
                megs.append(meg.numpy())
                labels.append(vocab_to_id[w])
        megs = np.stack(megs)  # (N,306,250)
        labels = np.array(labels)
        print(f"[{partition}] in-vocab examples: {len(labels)}")
        pre = _preprocess_windows(megs, highpass=(None if highpass < 0 else highpass))
        pred = _predict_embeddings(module, cp, pre, device)
        probs = _retrieval_probs(pred, vocab_embs)
        b10 = _bacc_topk(probs, labels, 10, 50)
        b1 = _bacc_topk(probs, labels, 1, 50)
        print(f"[{partition}] BAcc@10={b10:.4f}  BAcc@1={b1:.4f}  (random ~0.20 / ~0.02)")
    return {"ok": True}


@app.function(image=submit_image, gpu="L4", volumes=VOLUMES, timeout=4 * 60 * 60)
def generate(track: str = "deep", highpass: float = -1.0, temperature: float = 0.5):
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

    primary_probs, secondary_probs = [], []
    buf = []
    BATCH = 256

    def flush(buf):
        megs = np.stack(buf)  # (b,306,250)
        pre = _preprocess_windows(megs, highpass=(None if highpass < 0 else highpass))
        pred = _predict_embeddings(module, cp, pre, device)
        primary_probs.append(_retrieval_probs(pred, prim_embs, temperature))
        secondary_probs.append(_retrieval_probs(pred, sec_embs, temperature))

    for meg, _meta in holdout.iter_windows(batch_size=None):
        buf.append(meg)
        if len(buf) == BATCH:
            flush(buf); buf = []
    if buf:
        flush(buf)

    primary = np.concatenate(primary_probs)
    secondary = np.concatenate(secondary_probs)
    out_dir = f"{WORK_DIR}/submissions"
    os.makedirs(out_dir, exist_ok=True)
    out = write_submission(
        f"{out_dir}/{track}_dascoli_submission.csv",
        indices=holdout.indices,
        primary_probs=primary,
        secondary_probs=secondary,
    )
    work_vol.commit()
    print("wrote:", out, "shape:", primary.shape)
    # quick sanity: argmax distribution
    am = primary.argmax(1)
    uniq, cnt = np.unique(am, return_counts=True)
    print("distinct argmax classes:", len(uniq), "top:", sorted(zip(cnt, uniq))[-5:])
    return {"path": str(out), "n": int(primary.shape[0])}
