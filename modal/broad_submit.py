"""
Broad track submission from a fine-tuned MEG-XL checkpoint.

For each holdout window (306,250)@250Hz: resample->50Hz, MEG-XL preprocessing
(baseline/robust-scale/clip), reconstruct sensor geometry (our Neuromag-306 JSON),
run the fine-tuned backbone + word MLP -> 1024-d embedding, cosine-retrieve
against the 50 competition words' t5-large embeddings -> softmax.

Reuses MEG-XL's own functions (load_tokenizer, load_criss_cross_model,
generate_word_embeddings, _process_single_chunk, load_libribrain_sensors,
norm_sensor_positions) so the submission matches training exactly.
"""

import os
import sys
import glob

import modal

from common import VOLUMES, WORK_DIR, work_vol
from broad_megxl import (megxl_image, MEGXL_WORK, DATA_ROOT, CACHE_DIR, LOG_DIR,
                         SENSOR_JSON, BIOCODEC_CKPT, MEGXL_HF_REPO, MEGXL_CKPT)

submit_image = megxl_image
if modal.is_local():
    submit_image = submit_image.add_local_python_source("broad_megxl", "broad_submit")

app = modal.App("pnpl-broad-submit")


def _normalize_word(w: str) -> str:
    """MEG-XL matches the competition vocab via datafit50 which includes 'it's'
    (curly). t5 tokenization handles the string directly; we lowercase only."""
    return str(w).strip().lower()


def _load_models(device):
    sys.path.insert(0, "/root/megxl")
    import torch
    from huggingface_hub import hf_hub_download
    from brainstorm.evaluate_criss_cross_word_classification import (
        load_tokenizer, load_criss_cross_model, CrissCrossWordEmbeddingExtractor,
    )

    tok = load_tokenizer(BIOCODEC_CKPT, device=device)
    pretrained = hf_hub_download(MEGXL_HF_REPO, MEGXL_CKPT)
    model = load_criss_cross_model(pretrained, tok, device=device)

    ckpts = sorted(glob.glob(f"{LOG_DIR}/**/checkpoint_best.pt", recursive=True))
    if not ckpts:
        ckpts = sorted(glob.glob(f"{LOG_DIR}/**/checkpoint_latest.pt", recursive=True))
    ck = torch.load(ckpts[-1], map_location="cpu", weights_only=False)
    print("fine-tuned ckpt:", ckpts[-1])

    # overwrite backbone with fine-tuned weights
    cc_state = ck["criss_cross_state_dict"]
    cc_state = {k: v for k, v in cc_state.items() if "rope_embedding_layer.rotate" not in k}
    m, u = model.load_state_dict(cc_state, strict=False)
    print(f"backbone load: {len(m)} missing, {len(u)} unexpected")
    model.eval()

    # word MLP: infer num_subjects + film from the saved state dict
    wstate = ck["word_mlp_state_dict"]
    use_film = "subject_embedding.weight" in wstate
    num_subjects = wstate["subject_embedding.weight"].shape[0] if use_film else 0
    subj_dim = wstate["subject_embedding.weight"].shape[1] if use_film else 64
    hidden = wstate["mlp.0.weight"].shape[0]
    word_mlp = CrissCrossWordEmbeddingExtractor(
        num_channels=306, latent_dim=model.latent_dim, embed_dim=1024,
        hidden_dim=hidden, use_subject_film=use_film, num_subjects=num_subjects,
        subject_embedding_dim=subj_dim,
    )
    word_mlp.load_state_dict(wstate)
    word_mlp.to(device).eval()
    print(f"word_mlp: film={use_film} num_subjects={num_subjects} hidden={hidden}")
    return model, word_mlp


def _sensor_tensors(device):
    """Reconstruct (xyz, abc, types, mask) in the LibriBrain channel order used by
    the holdout windows (standard Neuromag MEG0111..MEG2643)."""
    sys.path.insert(0, "/root/megxl")
    import numpy as np, torch, h5py
    from brainstorm.data.preprocessing import load_libribrain_sensors
    from brainstorm.data.utils import norm_sensor_positions

    # channel order from a real LibriBrain h5
    ref = sorted(glob.glob(f"{DATA_ROOT}/Sherlock1/derivatives/serialised/*.h5"))[0]
    with h5py.File(ref, "r") as f:
        ch_names = [s.strip() for s in f.attrs["channel_names"].split(",")]
        ch_types = [s.strip() for s in f.attrs["channel_types"].split(",")]
    ch_names = [c for c, t in zip(ch_names, ch_types) if c.startswith("MEG")]

    xyzdir_d, types_d = load_libribrain_sensors(SENSOR_JSON)
    xyzdir = np.stack([xyzdir_d[c] for c in ch_names]).astype(np.float64)  # (306,6)
    types = np.array([types_d[c] for c in ch_names])                       # (306,)
    xyzdir_n = norm_sensor_positions(xyzdir.copy())  # (306,6), normalizes in place
    xyz = torch.from_numpy(xyzdir_n[:, :3]).float().to(device)
    abc = torch.from_numpy(xyzdir_n[:, 3:]).float().to(device)
    stypes = torch.from_numpy(types).long().to(device)
    smask = torch.ones(306, device=device)
    return ch_names, xyz, abc, stypes, smask


def _resample_250_to_50(meg_250):
    """(306, T)@250 -> (306, ~T/5)@50 (mne anti-aliased resample of a whole sentence)."""
    import numpy as np, mne
    mne.set_log_level("ERROR")
    C = meg_250.shape[0]
    info = mne.create_info([f"MEG{i}" for i in range(C)], 250.0, "mag")
    raw = mne.io.RawArray(meg_250.astype(np.float64), info, verbose=False)
    raw.resample(50.0, verbose=False)
    return raw.get_data().astype(np.float32)


def _proc_chunk(w50, sensor_types_np):
    """(306, S)@50 -> MEG-XL baseline/robust-scale/clip (per word window, as in training)."""
    sys.path.insert(0, "/root/megxl")
    from brainstorm.data.preprocessing import _process_single_chunk
    return _process_single_chunk(w50.astype("float64"), sensor_types_np, sfreq=50.0,
                                 baseline_duration=0.5, clip_range=(-5, 5)).astype("float32")


def _retrieval_probs(pred, vocab_embs, temperature=1.0):
    import numpy as np
    pe = pred / (np.linalg.norm(pred, axis=1, keepdims=True) + 1e-9)
    ve = vocab_embs / (np.linalg.norm(vocab_embs, axis=1, keepdims=True) + 1e-9)
    z = (pe @ ve.T) / temperature
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


@app.function(image=submit_image, gpu="L40S", volumes=VOLUMES, timeout=6 * 60 * 60)
def generate(track: str = "broad", temperature: float = 0.5):
    """Context-aware submission: reconstruct each holdout sentence (concatenate its
    word windows into one MEG-XL segment), run the backbone once, slice each word's
    encoded-time features -> word MLP -> t5 retrieval. Isolated `word`-source rows
    are single-word segments."""
    os.environ.setdefault("HF_HOME", "/hf-cache")
    sys.path.insert(0, "/root/megxl")
    import numpy as np, torch
    from brainstorm.evaluate_criss_cross_word_classification import (
        generate_word_embeddings, map_raw_to_encoded_timesteps)
    from pnpl.competition import (LibriBrainCompetitionHoldout, write_submission,
                                  PRIMARY_VOCAB, SECONDARY_VOCAB)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, word_mlp = _load_models(device)
    ch_names, xyz, abc, stypes, smask = _sensor_tensors(device)
    stypes_np = stypes.cpu().numpy()
    xyz1, abc1 = xyz.unsqueeze(0), abc.unsqueeze(0)
    typ1, msk1 = stypes.unsqueeze(0), smask.unsqueeze(0)

    prim = [_normalize_word(w) for w in PRIMARY_VOCAB]
    sec = [_normalize_word(w) for w in SECONDARY_VOCAB]
    prim_embs = generate_word_embeddings(prim, layer=12, cache_dir=f"{MEGXL_WORK}/emb_cache",
                                         device=device, dataset_type="comp_primary").numpy()
    sec_embs = generate_word_embeddings(sec, layer=12, cache_dir=f"{MEGXL_WORK}/emb_cache",
                                        device=device, dataset_type="comp_moses").numpy()

    holdout = LibriBrainCompetitionHoldout(track=track)
    print("holdout:", repr(holdout), holdout.counts())
    idx_of = {(m["subject"], m["source"], m["epoch"], m["word"]): m["index"]
              for m in holdout.metadata}
    N = len(holdout)
    primary = np.zeros((N, len(prim)), dtype=np.float32)
    secondary = np.zeros((N, len(sec)), dtype=np.float32)
    W = 50  # samples per word window @ 50 Hz

    def emit(emb_np, ix):
        e = emb_np[None]
        primary[ix] = _retrieval_probs(e, prim_embs, temperature)[0]
        secondary[ix] = _retrieval_probs(e, sec_embs, temperature)[0]

    def word_embs_from_segment(word_windows):
        """word_windows: list of (306,50) processed -> list of (1024,) embeddings via
        one backbone pass over the concatenated segment + per-word feature slice."""
        seg = np.concatenate(word_windows, axis=1)  # (306, nw*50)
        meg = torch.from_numpy(seg).float().unsqueeze(0).to(device)
        with torch.no_grad():
            feats = model(meg, xyz1, abc1, typ1, msk1, apply_mask=False)["features"][0]
            outs = []
            for j in range(len(word_windows)):
                s_t, e_t = map_raw_to_encoded_timesteps(j * W, (j + 1) * W)
                e_t = min(max(e_t, s_t + 1), feats.shape[1])
                wf = feats[:, s_t:e_t, :]                       # (306, T_sub, 512)
                outs.append(word_mlp(wf, subject_idx=-1).cpu().numpy())
        return outs

    for subj in holdout.subjects:
        # ---- sentence source: reconstruct each sentence as a segment
        sp = holdout._ensure_file(subj, "sentence")
        with np.load(sp, allow_pickle=True) as d:
            meg = np.asarray(d["meg"], dtype=np.float32)
            n_times = np.asarray(d["sentence_n_times"]).astype(int)
            onsets = np.asarray(d["word_onsets_s"], dtype=np.float64)
            wmask = np.asarray(d["word_mask"])
        for si in range(meg.shape[0]):
            sent50 = _resample_250_to_50(meg[si, :, :n_times[si]])   # (306, nt50)
            valid = np.nonzero(wmask[si])[0]
            wins = []
            for wi in valid:
                st = int(round(float(onsets[si, wi]) * 50.0))
                w = sent50[:, st:st + W]
                if w.shape[1] < W:  # pad tail if needed
                    w = np.pad(w, ((0, 0), (0, W - w.shape[1])))
                wins.append(_proc_chunk(w, stypes_np))
            if not wins:
                continue
            embs = word_embs_from_segment(wins)
            for j, wi in enumerate(valid):
                emit(embs[j], idx_of[(subj, "sentence", si, int(wi))])
        # ---- word source: isolated single-word segments (batched)
        wp = holdout._ensure_file(subj, "word")
        with np.load(wp, allow_pickle=True) as d:
            wmeg = np.asarray(d["meg"], dtype=np.float32)  # (Nw,306,250)
        for ei in range(wmeg.shape[0]):
            w = _proc_chunk(_resample_250_to_50(wmeg[ei]), stypes_np)
            emit(word_embs_from_segment([w])[0], idx_of[(subj, "word", ei, -1)])
        print(f"  subject {subj}: sentence+word done")

    out_dir = f"{WORK_DIR}/submissions"
    os.makedirs(out_dir, exist_ok=True)
    out = write_submission(f"{out_dir}/{track}_megxl_submission.csv",
                           indices=holdout.indices, primary_probs=primary,
                           secondary_probs=secondary)
    work_vol.commit()
    am = primary.argmax(1)
    uniq, cnt = np.unique(am, return_counts=True)
    print("wrote:", out, "shape:", primary.shape, "distinct argmax:", len(uniq))
    return {"path": str(out), "n": int(N)}
