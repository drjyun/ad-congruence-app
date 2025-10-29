# ad_context_api.py
# FastAPI microservice that turns your existing audio2/visual2 notebooks
# into production endpoints for audiovisual congruence scoring.
# Focus: embeddings + sliding-window cosine similarity (no EEG or extra features).
#
# Endpoints
# - POST /upload_ad : upload an ad video; server extracts audio + frames (0.96 s)
# - POST /score/{ad_id} : compute audio + visual congruence vs. TV contexts
# - GET  /plot/{ad_id}  : quick PNG of time-resolved congruence for a show
#
# Notes
# - Precompute & store TV context embeddings as *.npy using your visual2/audio2 pipelines:
#   Visual:  <VISUAL_DIR>/<Show>_1_sampled_960ms_features.npy, <Show>_2_... (two segments per show)
#   Audio:   <AUDIO_DIR>/<Show>_1_features.npy, <Show>_2_features.npy
# - The service averages the two context segments internally.
# - Windows: 10 s, step 5 s (configurable)

import os, io, uuid, glob, math, shutil, subprocess
from typing import Dict, List, Optional

import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse, StreamingResponse

# ── ML deps (pin in requirements.txt) ─────────────────────────────────────────
# Audio
import librosa
import tensorflow as tf
import tensorflow_hub as hub

# Visual
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from transformers import ViTImageProcessor, ViTModel

# Plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Config ───────────────────────────────────────────────────────────────────
DATA_ROOT      = os.getenv("DATA_ROOT", os.getcwd())
UPLOAD_ROOT    = os.path.join(DATA_ROOT, "uploads")
AUDIO_DIR      = os.path.join(DATA_ROOT, "audio")
VISUAL_DIR     = os.path.join(DATA_ROOT, "visual")
CONTEXT_SHOWS  = [s.strip() for s in os.getenv("CONTEXT_SHOWS", "CNN,FOXNEWS,NFL,LEGO,PICKERS,CONTINENTAL,BIGMOOD").split(",")]
SAMPLE_SEC     = float(os.getenv("SAMPLE_SEC", "0.96"))
WINDOW_SEC     = float(os.getenv("WINDOW_SEC", "10"))
STEP_SEC       = float(os.getenv("STEP_SEC", "5"))

# Ensure dirs
for d in (UPLOAD_ROOT, AUDIO_DIR, VISUAL_DIR):
    os.makedirs(d, exist_ok=True)

# ── Helpers ──────────────────────────────────────────────────────────────────
def sliding_indices(n: int, win: int, step: int):
    return range(0, max(n - win + 1, 0), step)

def fisher_z(r: np.ndarray) -> np.ndarray:
    r = np.clip(r, -0.999999, 0.999999)
    return 0.5 * np.log((1 + r) / (1 - r))

def inv_fisher_z(z: np.ndarray) -> np.ndarray:
    return np.tanh(z)

# ── Audio: VGGish ────────────────────────────────────────────────────────────
print("Loading VGGish from TF Hub…")
vggish = hub.load('https://tfhub.dev/google/vggish/1')
print("VGGish loaded.")

def extract_audio_wav(video_path: str, wav_path: str):
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-ar", "16000",  # 16 kHz
        "-ac", "1",       # mono
        wav_path
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def vggish_embeddings(wav_path: str) -> np.ndarray:
    y, sr = librosa.load(wav_path, sr=16000)
    audio_tensor = tf.convert_to_tensor(y, dtype=tf.float32)
    emb = vggish(audio_tensor)  # (T, 128) at ~0.96 s hop
    return emb.numpy()

# ── Visual: ViT-base/16 ──────────────────────────────────────────────────────
print("Loading ViT…")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
fe     = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
vit    = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k").to(device)
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=fe.image_mean, std=fe.image_std)
])
print("ViT ready.")

class FrameDataset(Dataset):
    def __init__(self, paths: List[str]):
        self.paths = sorted(paths)
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, i):
        return transform(Image.open(self.paths[i]).convert("RGB"))

def sample_frames(video_path: str, out_dir: str, sample_sec: float = 0.96):
    os.makedirs(out_dir, exist_ok=True)
    # Use ffmpeg to output 1 frame per ~0.96s (≈25/24 fps approximation). Alternatively, use cv2 stride.
    # Here we use ffmpeg -vf fps=… to target 1 / 0.96 ≈ 1.0417 fps.
    fps = 1.0 / sample_sec
    pattern = os.path.join(out_dir, "frame_%05d.jpg")
    cmd = ["ffmpeg", "-y", "-i", video_path, "-vf", f"fps={fps}", pattern]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def vit_embeddings_from_frames(frame_dir: str, batch_size: int = 16) -> np.ndarray:
    frames = sorted(glob.glob(os.path.join(frame_dir, "frame_*.jpg")))
    if not frames:
        return np.empty((0, 768))
    dl = DataLoader(FrameDataset(frames), batch_size=batch_size, shuffle=False)
    feats = []
    vit.eval()
    with torch.no_grad():
        for batch in dl:
            batch = batch.to(device)
            out = vit(batch).last_hidden_state.mean(dim=1).cpu().numpy()  # (B, 768)
            feats.append(out)
    return np.vstack(feats)

# ── Context loading (precomputed *.npy) ───────────────────────────────────────

def load_context_avg_arrays(kind: str) -> Dict[str, List[np.ndarray]]:
    """kind ∈ {"audio","visual"}; returns dict[show] -> list of segment arrays"""
    d = AUDIO_DIR if kind == "audio" else VISUAL_DIR
    out = {}
    for show in CONTEXT_SHOWS:
        segs = sorted(glob.glob(os.path.join(d, f"{show}_*_features.npy"))) if kind=="audio" else \
               sorted(glob.glob(os.path.join(d, f"{show}_*_sampled_960ms_features.npy")))
        arrs = [np.load(p) for p in segs]
        if arrs:
            out[show] = arrs
    if not out:
        raise RuntimeError(f"No context features found for kind={kind} in {d}")
    return out

# ── Congruence core ──────────────────────────────────────────────────────────

def sliding_congruence(tv_arrays: List[np.ndarray], ad_array: np.ndarray, window_sec: float, step_sec: float, chunk_dur: float) -> np.ndarray:
    """Compute Fisher-z-averaged cosine similarity between a TV context (two segments)
    and an ad, using sliding windows. Returns array of rows with columns:
    [tv_t0_sec, ad_t0_sec, z_cosine].
    """
    W = max(int(round(window_sec / chunk_dur)), 1)
    S = max(int(round(step_sec   / chunk_dur)), 1)

    # Collect per (t0, a0) across segments
    buf = {}
    for tv in tv_arrays:
        n_tv = tv.shape[0]
        for t0 in sliding_indices(n_tv, W, S):
            tv_win = tv[t0:t0+W].reshape(1, -1)
            for a0 in sliding_indices(ad_array.shape[0], W, S):
                ad_win = ad_array[a0:a0+W].reshape(1, -1)
                # cosine_similarity for row vectors
                num = float(tv_win @ ad_win.T)
                den = float(np.linalg.norm(tv_win) * np.linalg.norm(ad_win) + 1e-8)
                r   = num / den
                z   = float(fisher_z(np.array([r]))[0])
                buf.setdefault((t0, a0), []).append(z)

    rows = []
    for (t0, a0), zs in buf.items():
        z_mean = float(np.mean(zs))
        rows.append([t0 * chunk_dur, a0 * chunk_dur, z_mean])
    return np.array(rows, dtype=float)

# ── FastAPI app ──────────────────────────────────────────────────────────────
app = FastAPI(title="Ad–Context Congruence API", version="0.1.1")

@app.post("/upload_ad")
def upload_ad(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".mp4", ".mov", ".mkv")):
        raise HTTPException(400, "Please upload a video file (.mp4/.mov/.mkv).")
    ad_id = str(uuid.uuid4())
    ad_dir = os.path.join(UPLOAD_ROOT, ad_id)
    os.makedirs(ad_dir, exist_ok=True)
    dst = os.path.join(ad_dir, file.filename)
    with open(dst, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"ad_id": ad_id, "filename": file.filename}

@app.post("/score/{ad_id}")
def score_ad(ad_id: str, include_audio: bool = True, include_visual: bool = True):
    ad_dir = os.path.join(UPLOAD_ROOT, ad_id)
    if not os.path.isdir(ad_dir):
        raise HTTPException(404, "Unknown ad_id.")
    # Find the uploaded video
    vids = [p for p in glob.glob(os.path.join(ad_dir, "*")) if p.lower().endswith((".mp4",".mov",".mkv"))]
    if not vids:
        raise HTTPException(400, "No video found for this ad_id.")
    video_path = vids[0]

    result = {"ad_id": ad_id, "window_sec": WINDOW_SEC, "step_sec": STEP_SEC, "chunk_dur": SAMPLE_SEC, "shows": {}}

    # AUDIO
    if include_audio:
        try:
            wav_path = os.path.join(ad_dir, "audio.wav")
            extract_audio_wav(video_path, wav_path)
            ad_audio = vggish_embeddings(wav_path)  # (T, 128)
            ctx_audio = load_context_avg_arrays("audio")
            for show, segs in ctx_audio.items():
                A = sliding_congruence(segs, ad_audio, WINDOW_SEC, STEP_SEC, SAMPLE_SEC)
                result["shows"].setdefault(show, {})["audio"] = A.tolist()
        except Exception as e:
            raise HTTPException(500, f"Audio scoring error: {e}")

    # VISUAL
    if include_visual:
        try:
            frames_dir = os.path.join(ad_dir, "frames")
            sample_frames(video_path, frames_dir, SAMPLE_SEC)
            ad_vis = vit_embeddings_from_frames(frames_dir)  # (T, 768)
            ctx_vis = load_context_avg_arrays("visual")
            for show, segs in ctx_vis.items():
                V = sliding_congruence(segs, ad_vis, WINDOW_SEC, STEP_SEC, SAMPLE_SEC)
                result["shows"].setdefault(show, {})["visual"] = V.tolist()
        except Exception as e:
            raise HTTPException(500, f"Visual scoring error: {e}")

    return JSONResponse(result)

@app.get("/plot/{ad_id}")
def plot_timeseries(
    ad_id: str,
    show: str = Query(..., description="Context show name, e.g., CNN"),
    modality: str = Query("audio", pattern="^(audio|visual)$"),
    agg: str = Query("ad_time", pattern="^(ad_time|tv_time)$"),
):
    """Quick diagnostic plot.
    agg=ad_time → average over all TV windows at each ad window (ad_t0). Shows mean z and 95% CI.
    agg=tv_time → average over all ad windows at each TV window (tv_t0).
    """
    ad_dir = os.path.join(UPLOAD_ROOT, ad_id)
    json_path = None  # we call score endpoint logic again (simple route) for brevity

    # Recreate minimal in-memory result for this show/modality
    vids = [p for p in glob.glob(os.path.join(ad_dir, "*")) if p.lower().endswith((".mp4",".mov",".mkv"))]
    if not vids: raise HTTPException(400, "No video found for this ad_id.")
    video_path = vids[0]

    # Compute only the requested modality for requested show
    if modality == "audio":
        wav_path = os.path.join(ad_dir, "audio.wav")
        if not os.path.exists(wav_path): extract_audio_wav(video_path, wav_path)
        ad_arr   = vggish_embeddings(wav_path)
        ctx_dict = load_context_avg_arrays("audio")
    else:
        frames_dir = os.path.join(ad_dir, "frames")
        if not os.path.exists(frames_dir) or not glob.glob(os.path.join(frames_dir, "frame_*.jpg")):
            sample_frames(video_path, frames_dir, SAMPLE_SEC)
        ad_arr   = vit_embeddings_from_frames(frames_dir)
        ctx_dict = load_context_avg_arrays("visual")

    if show not in ctx_dict:
        raise HTTPException(404, f"Show '{show}' not found in contexts.")

    rows = sliding_congruence(ctx_dict[show], ad_arr, WINDOW_SEC, STEP_SEC, SAMPLE_SEC)
    # rows: [tv_t0, ad_t0, z]
    tv_t0  = rows[:,0]
    ad_t0  = rows[:,1]
    z      = rows[:,2]

    # Aggregate
    if agg == "ad_time":
        x = sorted(set(ad_t0))
        means, cis = [], []
        for t in x:
            ys = z[ad_t0 == t]
            means.append(float(np.mean(ys)))
            # simple normal approx CI
            se = float(np.std(ys, ddof=1) / math.sqrt(max(len(ys),1)))
            cis.append(1.96 * se)
        xlabel = "Ad time (s)"
    else:
        x = sorted(set(tv_t0))
        means, cis = [], []
        for t in x:
            ys = z[tv_t0 == t]
            means.append(float(np.mean(ys)))
            se = float(np.std(ys, ddof=1) / math.sqrt(max(len(ys),1)))
            cis.append(1.96 * se)
        xlabel = "TV context time (s)"

    # Plot
    fig, ax = plt.subplots(figsize=(9,4))
    ax.plot(x, means, linewidth=2)
    ax.fill_between(x, np.array(means)-np.array(cis), np.array(means)+np.array(cis), alpha=0.2)
    ax.set_title(f"{show} – {modality} congruence ({agg})")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Fisher-z cosine similarity")
    ax.grid(True, alpha=.3)
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format='png', dpi=150)
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")

@app.get("/plot_both/{ad_id}")
def plot_both(
    ad_id: str,
    show: str = Query(..., description="Context show name, e.g., CNN"),
    agg: str = Query("ad_time", pattern="^(ad_time|tv_time)$"),
):
    """Two-panel plot: audio (top) and visual (bottom) congruence curves for the same show.
    Aggregation works like /plot: ad_time or tv_time.
    """
    ad_dir = os.path.join(UPLOAD_ROOT, ad_id)
    vids = [p for p in glob.glob(os.path.join(ad_dir, "*")) if p.lower().endswith((".mp4",".mov",".mkv"))]
    if not vids: raise HTTPException(400, "No video found for this ad_id.")
    video_path = vids[0]

    # AUDIO arrays
    wav_path = os.path.join(ad_dir, "audio.wav")
    if not os.path.exists(wav_path): extract_audio_wav(video_path, wav_path)
    ad_audio = vggish_embeddings(wav_path)
    ctx_audio = load_context_avg_arrays("audio")
    if show not in ctx_audio: raise HTTPException(404, f"Show '{show}' not found in audio contexts.")
    rows_a = sliding_congruence(ctx_audio[show], ad_audio, WINDOW_SEC, STEP_SEC, SAMPLE_SEC)

    # VISUAL arrays
    frames_dir = os.path.join(ad_dir, "frames")
    if not os.path.exists(frames_dir) or not glob.glob(os.path.join(frames_dir, "frame_*.jpg")):
        sample_frames(video_path, frames_dir, SAMPLE_SEC)
    ad_vis = vit_embeddings_from_frames(frames_dir)
    ctx_vis = load_context_avg_arrays("visual")
    if show not in ctx_vis: raise HTTPException(404, f"Show '{show}' not found in visual contexts.")
    rows_v = sliding_congruence(ctx_vis[show], ad_vis, WINDOW_SEC, STEP_SEC, SAMPLE_SEC)

    def agg_curve(rows, which):
        tv_t0, ad_t0, z = rows[:,0], rows[:,1], rows[:,2]
        if which == "ad_time":
            xs = sorted(set(ad_t0))
            xlabel = "Ad time (s)"
            means, cis = [], []
            for t in xs:
                ys = z[ad_t0 == t]
                means.append(float(np.mean(ys)))
                se = float(np.std(ys, ddof=1) / max(np.sqrt(len(ys)), 1))
                cis.append(1.96 * se)
        else:
            xs = sorted(set(tv_t0))
            xlabel = "TV context time (s)"
            means, cis = [], []
            for t in xs:
                ys = z[tv_t0 == t]
                means.append(float(np.mean(ys)))
                se = float(np.std(ys, ddof=1) / max(np.sqrt(len(ys)), 1))
                cis.append(1.96 * se)
        return xs, means, cis, xlabel

    xa, ma, cia, xlabel = agg_curve(rows_a, agg)
    xv, mv, civ, _      = agg_curve(rows_v, agg)

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=False)
    for ax, x, m, ci, title in [
        (axes[0], xa, ma, cia, f"{show} – audio congruence ({agg})"),
        (axes[1], xv, mv, civ, f"{show} – visual congruence ({agg})"),
    ]:
        ax.plot(x, m, linewidth=2)
        ax.fill_between(x, np.array(m)-np.array(ci), np.array(m)+np.array(ci), alpha=0.2)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Fisher-z cosine similarity")
        ax.grid(True, alpha=.3)

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150)
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")

# ── Action schema for Custom GPT (Assistant) ─────────────────────────────────
# Use this JSON when registering your API as a tool/action in a Custom GPT.
OPENAI_ACTION_SCHEMA = {
  "schema_version": "v1",
  "name_for_human": "Ad–Context Congruence API",
  "name_for_model": "congruence_api",
  "description_for_model": (
    "Call this tool to score an uploaded ad video against predefined TV contexts. "
    "Primary outputs are time-resolved Fisher-z cosine similarities for audio and visual embeddings."
  ),
  "api": {
    "type": "openapi",
    "url": "https://<your-domain>/openapi.json",
    "has_user_authentication": False
  },
  "auth": {"type": "none"}
}

# Example usage inside the Custom GPT instructions:
#  - After the user uploads a video, call congruence_api.upload_ad to get an ad_id.
#  - Then call congruence_api.score with that ad_id and return a summary plus a plot URL
#    like https://<your-domain>/plot/{ad_id}?show=CNN&modality=audio&agg=ad_time

# --- Inject public server URL into OpenAPI (for Custom GPT import) ---
import os
from fastapi.openapi.utils import get_openapi

#PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "https://3dc8094eaa99.ngrok-free.app")
PUBLIC_BASE_URL = os.getenv(
    "PUBLIC_BASE_URL",
    "https://sitting-profits-twiki-articles.trycloudflare.com"
)

def custom_openapi():
    if app.openapi_schema:
        # ensure servers stays set even if cached
        app.openapi_schema["servers"] = [{"url": PUBLIC_BASE_URL}]
        return app.openapi_schema
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        routes=app.routes,
    )
    openapi_schema["servers"] = [{"url": PUBLIC_BASE_URL}]
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

# add to ad_context_congruence_api_fast_api_backend.py
from fastapi import Body
import requests

@app.post("/upload_ad_url")
def upload_ad_url(url: str = Body(..., embed=True)):
    ad_id = str(uuid.uuid4())
    ad_dir = os.path.join(UPLOAD_ROOT, ad_id)
    os.makedirs(ad_dir, exist_ok=True)
    dst = os.path.join(ad_dir, "ad.mp4")
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    with open(dst, "wb") as f:
        f.write(r.content)
    return {"ad_id": ad_id, "filename": "ad.mp4"}

# Place this below your /plot_both endpoint, before the __main__ block.
from fastapi import Query

@app.get("/rank/{ad_id}")
def rank_ad(
    ad_id: str,
    combine: str = Query("mean", pattern="^(mean|audio|visual)$")
):
    """
    Compute per-show mean Fisher-z congruence.
      combine:
        - 'mean'   : average of audio & visual means
        - 'audio'  : audio only
        - 'visual' : visual only
    Returns shows sorted descending by the selected metric.
    """
    # Locate uploaded video
    ad_dir = os.path.join(UPLOAD_ROOT, ad_id)
    vids = [p for p in glob.glob(os.path.join(ad_dir, "*"))
            if p.lower().endswith((".mp4", ".mov", ".mkv"))]
    if not vids:
        raise HTTPException(400, "No video found for this ad_id.")
    video_path = vids[0]

    # ---- AUDIO (VGGish) ----
    wav_path = os.path.join(ad_dir, "audio.wav")
    if not os.path.exists(wav_path):
        extract_audio_wav(video_path, wav_path)
    ad_audio = vggish_embeddings(wav_path)                   # (T, 128)
    ctx_audio = load_context_avg_arrays("audio")             # dict[show] -> [seg1, seg2]

    # ---- VISUAL (ViT) ----
    frames_dir = os.path.join(ad_dir, "frames")
    if not os.path.exists(frames_dir) or not glob.glob(os.path.join(frames_dir, "frame_*.jpg")):
        sample_frames(video_path, frames_dir, SAMPLE_SEC)
    ad_visual = vit_embeddings_from_frames(frames_dir)       # (T, 768)
    ctx_visual = load_context_avg_arrays("visual")           # dict[show] -> [seg1, seg2]

    out = {}
    for show in CONTEXT_SHOWS:
        if show not in ctx_audio or show not in ctx_visual:
            continue

        # Audio congruence
        Ra = sliding_congruence(ctx_audio[show], ad_audio,
                                WINDOW_SEC, STEP_SEC, SAMPLE_SEC)
        mean_a = float(np.mean(Ra[:, 2])) if Ra.size else float("nan")

        # Visual congruence
        Rv = sliding_congruence(ctx_visual[show], ad_visual,
                                WINDOW_SEC, STEP_SEC, SAMPLE_SEC)
        mean_v = float(np.mean(Rv[:, 2])) if Rv.size else float("nan")

        if combine == "audio":
            score = mean_a
        elif combine == "visual":
            score = mean_v
        else:  # 'mean'
            score = float(np.nanmean([mean_a, mean_v]))

        out[show] = {"audio": mean_a, "visual": mean_v, "combined": score}

    metric_key = {"mean": "combined", "audio": "audio", "visual": "visual"}[combine]
    ranking = sorted(out.items(), key=lambda kv: kv[1][metric_key], reverse=True)
    return {
        "ad_id": ad_id,
        "metric": metric_key,
        "ranking": [{"show": s, **m} for s, m in ranking]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
