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


def _preprocess(meg_250, sensor_types_np):
    """(B,306,250)@250 -> (B,306,~50)@50 with MEG-XL's baseline/robust-scale/clip."""
    sys.path.insert(0, "/root/megxl")
    import numpy as np, mne
    from brainstorm.data.preprocessing import _process_single_chunk
    mne.set_log_level("ERROR")
    B, C, T = meg_250.shape
    info = mne.create_info([f"MEG{i}" for i in range(C)], 250.0, "mag")
    out = []
    for b in range(B):
        raw = mne.io.RawArray(meg_250[b].astype(np.float64), info, verbose=False)
        raw.resample(50.0, verbose=False)
        d = raw.get_data()  # (C, ~50)
        d = _process_single_chunk(d, sensor_types_np, sfreq=50.0,
                                  baseline_duration=0.5, clip_range=(-5, 5))
        out.append(d.astype(np.float32))
    return np.stack(out)


def _retrieval_probs(pred, vocab_embs, temperature=1.0):
    import numpy as np
    pe = pred / (np.linalg.norm(pred, axis=1, keepdims=True) + 1e-9)
    ve = vocab_embs / (np.linalg.norm(vocab_embs, axis=1, keepdims=True) + 1e-9)
    z = (pe @ ve.T) / temperature
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


@app.function(image=submit_image, gpu="A100-40GB", volumes=VOLUMES, timeout=6 * 60 * 60)
def generate(track: str = "broad", temperature: float = 0.5, batch: int = 64):
    os.environ.setdefault("HF_HOME", "/hf-cache")
    sys.path.insert(0, "/root/megxl")
    import numpy as np, torch
    from brainstorm.evaluate_criss_cross_word_classification import generate_word_embeddings
    from pnpl.competition import (LibriBrainCompetitionHoldout, write_submission,
                                  PRIMARY_VOCAB, SECONDARY_VOCAB)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, word_mlp = _load_models(device)
    ch_names, xyz, abc, stypes, smask = _sensor_tensors(device)
    stypes_np = stypes.cpu().numpy()

    prim = [_normalize_word(w) for w in PRIMARY_VOCAB]
    sec = [_normalize_word(w) for w in SECONDARY_VOCAB]
    prim_embs = generate_word_embeddings(prim, layer=12, cache_dir=f"{MEGXL_WORK}/emb_cache",
                                         device=device, dataset_type="comp_primary").numpy()
    sec_embs = generate_word_embeddings(sec, layer=12, cache_dir=f"{MEGXL_WORK}/emb_cache",
                                        device=device, dataset_type="comp_moses").numpy()

    holdout = LibriBrainCompetitionHoldout(track=track)
    print("holdout:", repr(holdout), holdout.counts())

    prim_probs, sec_probs = [], []
    xyz_b = abc_b = types_b = mask_b = None

    def run(buf):
        pre = _preprocess(np.stack(buf), stypes_np)   # (b,306,~50)
        embs = []
        with torch.no_grad():
            for i in range(pre.shape[0]):
                meg = torch.from_numpy(pre[i:i+1]).float().to(device)  # (1,306,~50)
                out = model(meg, xyz.unsqueeze(0), abc.unsqueeze(0),
                            stypes.unsqueeze(0), smask.unsqueeze(0), apply_mask=False)
                feats = out["features"][0]            # (306, T', 512)
                emb = word_mlp(feats, subject_idx=-1)  # (1024,)
                embs.append(emb.cpu().numpy())
        embs = np.stack(embs)
        prim_probs.append(_retrieval_probs(embs, prim_embs, temperature))
        sec_probs.append(_retrieval_probs(embs, sec_embs, temperature))

    buf = []
    for meg, _meta in holdout.iter_windows(batch_size=None):
        buf.append(meg)
        if len(buf) == batch:
            run(buf); buf = []
            if len(prim_probs) % 10 == 0:
                print(f"  processed ~{len(prim_probs)*batch} windows")
    if buf:
        run(buf)

    primary = np.concatenate(prim_probs)
    secondary = np.concatenate(sec_probs)
    out_dir = f"{WORK_DIR}/submissions"
    os.makedirs(out_dir, exist_ok=True)
    out = write_submission(f"{out_dir}/{track}_megxl_submission.csv",
                           indices=holdout.indices, primary_probs=primary,
                           secondary_probs=secondary)
    work_vol.commit()
    am = primary.argmax(1)
    uniq, cnt = np.unique(am, return_counts=True)
    print("wrote:", out, "shape:", primary.shape, "distinct argmax:", len(uniq))
    return {"path": str(out), "n": int(primary.shape[0])}
