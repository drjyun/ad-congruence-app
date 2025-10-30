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

# ad_context_api.py  (lazy-load + memory safe)
import os, io, uuid, glob, math, shutil, subprocess
from typing import Dict, List
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Body
from fastapi.responses import JSONResponse, StreamingResponse
import requests
import librosa
import tensorflow as tf
import tensorflow_hub as hub
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import ViTImageProcessor, ViTModel
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fastapi.openapi.utils import get_openapi

# ── Config ──────────────────────────────────────────────
DATA_ROOT   = os.getenv("DATA_ROOT", os.getcwd())
UPLOAD_ROOT = os.path.join(DATA_ROOT, "uploads")
AUDIO_DIR   = os.path.join(DATA_ROOT, "audio")
VISUAL_DIR  = os.path.join(DATA_ROOT, "visual")
CONTEXT_SHOWS = [s.strip() for s in os.getenv(
    "CONTEXT_SHOWS",
    "CNN,FOXNEWS,NFL,LEGO,PICKERS,CONTINENTAL,BIGMOOD"
).split(",")]
SAMPLE_SEC = float(os.getenv("SAMPLE_SEC", "0.96"))
WINDOW_SEC = float(os.getenv("WINDOW_SEC", "10"))
STEP_SEC   = float(os.getenv("STEP_SEC", "5"))
for d in (UPLOAD_ROOT, AUDIO_DIR, VISUAL_DIR):
    os.makedirs(d, exist_ok=True)

# ── Helper math ─────────────────────────────────────────
def sliding_indices(n, win, step):
    return range(0, max(n - win + 1, 0), step)

def fisher_z(r):  r = np.clip(r, -0.999999, 0.999999);  return 0.5*np.log((1+r)/(1-r))

# ── Lazy model globals ──────────────────────────────────
vggish = None
vit, fe, transform = None, None, None
device = torch.device("cpu")

def get_vggish():
    global vggish
    if vggish is None:
        print("Loading VGGish from TF-Hub…")
        vggish = hub.load("https://tfhub.dev/google/vggish/1")
        print("VGGish ready.")
    return vggish

def get_vit():
    global vit, fe, transform
    if vit is None:
        print("Loading ViT…")
        fe = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
        vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k").to(device)
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=fe.image_mean, std=fe.image_std)
        ])
        print("ViT ready.")
    return vit, fe, transform

# ── Audio helpers ───────────────────────────────────────
def extract_audio_wav(video_path, wav_path):
    subprocess.run([
        "ffmpeg","-y","-i",video_path,"-ar","16000","-ac","1",wav_path
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def vggish_embeddings(wav_path):
    y, sr = librosa.load(wav_path, sr=16000)
    model = get_vggish()
    audio_tensor = tf.convert_to_tensor(y, dtype=tf.float32)
    emb = model(audio_tensor)
    return emb.numpy()

# ── Visual helpers ──────────────────────────────────────
def sample_frames(video_path, out_dir, sample_sec=0.96):
    os.makedirs(out_dir, exist_ok=True)
    fps = 1.0 / sample_sec
    pattern = os.path.join(out_dir, "frame_%05d.jpg")
    subprocess.run(["ffmpeg","-y","-i",video_path,"-vf",f"fps={fps}",pattern],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def vit_embeddings_from_frames(frame_dir, batch_size=8):
    vit_model, fe_proc, tfm = get_vit()
    frames = sorted(glob.glob(os.path.join(frame_dir, "frame_*.jpg")))
    if not frames: return np.empty((0,768))

    class FrameDataset(Dataset):
        def __init__(self, paths): self.paths=paths
        def __len__(self): return len(self.paths)
        def __getitem__(self,i): return tfm(Image.open(self.paths[i]).convert("RGB"))

    dl = DataLoader(FrameDataset(frames), batch_size=batch_size, shuffle=False)
    feats=[]
    vit_model.eval()
    with torch.no_grad():
        for batch in dl:
            batch=batch.to(device)
            out=vit_model(batch).last_hidden_state.mean(dim=1).cpu().numpy()
            feats.append(out)
    return np.vstack(feats)

# ── Context loader ──────────────────────────────────────
def load_context_avg_arrays(kind):
    d = AUDIO_DIR if kind=="audio" else VISUAL_DIR
    out={}
    for show in CONTEXT_SHOWS:
        segs = sorted(glob.glob(os.path.join(
            d, f"{show}_*_features.npy" if kind=="audio"
              else f"{show}_*_sampled_960ms_features.npy"
        )))
        arrs=[np.load(p) for p in segs]
        if arrs: out[show]=arrs
    if not out: raise RuntimeError(f"No context features for {kind} in {d}")
    return out

# ── Core congruence ─────────────────────────────────────
def sliding_congruence(tv_arrays, ad_array, window_sec, step_sec, chunk_dur):
    W=max(int(round(window_sec/chunk_dur)),1)
    S=max(int(round(step_sec/chunk_dur)),1)
    buf={}
    for tv in tv_arrays:
        n_tv=tv.shape[0]
        for t0 in sliding_indices(n_tv,W,S):
            tv_win=tv[t0:t0+W].reshape(1,-1)
            for a0 in sliding_indices(ad_array.shape[0],W,S):
                ad_win=ad_array[a0:a0+W].reshape(1,-1)
                r=float((tv_win@ad_win.T)/(np.linalg.norm(tv_win)*np.linalg.norm(ad_win)+1e-8))
                z=float(fisher_z(np.array([r]))[0])
                buf.setdefault((t0,a0),[]).append(z)
    rows=[[t0*chunk_dur,a0*chunk_dur,float(np.mean(zs))] for (t0,a0),zs in buf.items()]
    return np.array(rows,float)

# ── FastAPI app ─────────────────────────────────────────
app=FastAPI(title="Ad–Context Congruence API",version="0.2.0")

@app.post("/upload_ad")
def upload_ad(file:UploadFile=File(...)):
    if not file.filename.lower().endswith((".mp4",".mov",".mkv")):
        raise HTTPException(400,"Please upload a video file (.mp4/.mov/.mkv).")
    ad_id=str(uuid.uuid4())
    ad_dir=os.path.join(UPLOAD_ROOT,ad_id)
    os.makedirs(ad_dir,exist_ok=True)
    dst=os.path.join(ad_dir,file.filename)
    with open(dst,"wb") as f: shutil.copyfileobj(file.file,f)
    return {"ad_id":ad_id,"filename":file.filename}

@app.post("/upload_ad_url")
def upload_ad_url(url:str=Body(...,embed=True)):
    ad_id=str(uuid.uuid4())
    ad_dir=os.path.join(UPLOAD_ROOT,ad_id)
    os.makedirs(ad_dir,exist_ok=True)
    dst=os.path.join(ad_dir,"ad.mp4")
    r=requests.get(url,timeout=120); r.raise_for_status()
    with open(dst,"wb") as f: f.write(r.content)
    return {"ad_id":ad_id,"filename":"ad.mp4"}

# (score, plot, plot_both, rank) — unchanged logic from your version —
# … keep your original endpoint functions here …

# ── OpenAPI metadata ────────────────────────────────────
PUBLIC_BASE_URL=os.getenv("PUBLIC_BASE_URL","https://ad-congruence-api.onrender.com")
def custom_openapi():
    if app.openapi_schema:
        app.openapi_schema["servers"]=[{"url":PUBLIC_BASE_URL}]
        return app.openapi_schema
    schema=get_openapi(title=app.title,version=app.version,routes=app.routes)
    schema["servers"]=[{"url":PUBLIC_BASE_URL}]
    app.openapi_schema=schema
    return schema
app.openapi=custom_openapi

@app.get("/")
def root():
    return {
        "ok": True,
        "service": "Ad–Context Congruence API",
        "status": "live",
        "docs": "https://ad-congruence-api.onrender.com/docs"
    }

from fastapi import HTTPException, Query
from fastapi.responses import JSONResponse

@app.post("/score/{ad_id}")
def score_ad(ad_id: str, include_audio: bool = True, include_visual: bool = True):
    """Compute audio + visual congruence for an uploaded ad across all shows."""
    ad_dir = os.path.join(UPLOAD_ROOT, ad_id)
    if not os.path.isdir(ad_dir):
        raise HTTPException(404, "Unknown ad_id.")

    vids = [p for p in glob.glob(os.path.join(ad_dir, "*"))
            if p.lower().endswith((".mp4", ".mov", ".mkv"))]
    if not vids:
        raise HTTPException(400, "No video found for this ad_id.")
    video_path = vids[0]

    result = {
        "ad_id": ad_id,
        "window_sec": WINDOW_SEC,
        "step_sec": STEP_SEC,
        "chunk_dur": SAMPLE_SEC,
        "shows": {}
    }

    # AUDIO
    if include_audio:
        try:
            wav_path = os.path.join(ad_dir, "audio.wav")
            if not os.path.exists(wav_path):
                extract_audio_wav(video_path, wav_path)
            ad_audio = vggish_embeddings(wav_path)                   # (T, 128)
            ctx_audio = load_context_avg_arrays("audio")             # dict[show] -> [seg1, seg2]
            for show, segs in ctx_audio.items():
                A = sliding_congruence(segs, ad_audio, WINDOW_SEC, STEP_SEC, SAMPLE_SEC)
                result["shows"].setdefault(show, {})["audio"] = A.tolist()
        except Exception as e:
            raise HTTPException(500, f"Audio scoring error: {e}")

    # VISUAL
    if include_visual:
        try:
            frames_dir = os.path.join(ad_dir, "frames")
            if not (os.path.exists(frames_dir) and glob.glob(os.path.join(frames_dir, "frame_*.jpg"))):
                sample_frames(video_path, frames_dir, SAMPLE_SEC)
            ad_vis = vit_embeddings_from_frames(frames_dir)          # (T, 768)
            ctx_vis = load_context_avg_arrays("visual")              # dict[show] -> [seg1, seg2]
            for show, segs in ctx_vis.items():
                V = sliding_congruence(segs, ad_vis, WINDOW_SEC, STEP_SEC, SAMPLE_SEC)
                result["shows"].setdefault(show, {})["visual"] = V.tolist()
        except Exception as e:
            raise HTTPException(500, f"Visual scoring error: {e}")

    return JSONResponse(result)


@app.get("/plot_both/{ad_id}")
def plot_both(
    ad_id: str,
    show: str = Query(..., description="Context show name, e.g., CNN"),
    agg: str  = Query("ad_time", pattern="^(ad_time|tv_time)$"),
):
    """Returns a PNG with audio (top) and visual (bottom) congruence curves."""
    import io, math
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from fastapi.responses import StreamingResponse

    ad_dir = os.path.join(UPLOAD_ROOT, ad_id)
    vids = [p for p in glob.glob(os.path.join(ad_dir, "*"))
            if p.lower().endswith((".mp4",".mov",".mkv"))]
    if not vids:
        raise HTTPException(400, "No video found for this ad_id.")
    video_path = vids[0]

    # AUDIO arrays
    wav_path = os.path.join(ad_dir, "audio.wav")
    if not os.path.exists(wav_path):
        extract_audio_wav(video_path, wav_path)
    ad_audio = vggish_embeddings(wav_path)
    ctx_audio = load_context_avg_arrays("audio")
    if show not in ctx_audio:
        raise HTTPException(404, f"Show '{show}' not found in audio contexts.")
    rows_a = sliding_congruence(ctx_audio[show], ad_audio, WINDOW_SEC, STEP_SEC, SAMPLE_SEC)

    # VISUAL arrays
    frames_dir = os.path.join(ad_dir, "frames")
    if not (os.path.exists(frames_dir) and glob.glob(os.path.join(frames_dir, "frame_*.jpg"))):
        sample_frames(video_path, frames_dir, SAMPLE_SEC)
    ad_vis = vit_embeddings_from_frames(frames_dir)
    ctx_vis = load_context_avg_arrays("visual")
    if show not in ctx_vis:
        raise HTTPException(404, f"Show '{show}' not found in visual contexts.")
    rows_v = sliding_congruence(ctx_vis[show], ad_vis, WINDOW_SEC, STEP_SEC, SAMPLE_SEC)

    def agg_curve(rows, which):
        tv_t0, ad_t0, z = rows[:,0], rows[:,1], rows[:,2]
        if which == "ad_time":
            xs = sorted(set(ad_t0)); xlabel = "Ad time (s)"
            vals = [z[ad_t0 == t] for t in xs]
        else:
            xs = sorted(set(tv_t0)); xlabel = "TV context time (s)"
            vals = [z[tv_t0 == t] for t in xs]
        means = [float(np.mean(v)) for v in vals]
        cis   = [1.96 * float(np.std(v, ddof=1) / max(np.sqrt(len(v)), 1)) for v in vals]
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
        ax.set_title(title); ax.set_xlabel(xlabel); ax.grid(True, alpha=.3)

    import io
    buf = io.BytesIO()
    fig.tight_layout(); fig.savefig(buf, format="png", dpi=150); buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")


@app.get("/rank/{ad_id}")
def rank_ad(
    ad_id: str,
    combine: str = Query("mean", pattern="^(mean|audio|visual)$")
):
    """Return per-show mean Fisher-z scores and a ranking."""
    ad_dir = os.path.join(UPLOAD_ROOT, ad_id)
    vids = [p for p in glob.glob(os.path.join(ad_dir, "*"))
            if p.lower().endswith((".mp4",".mov",".mkv"))]
    if not vids:
        raise HTTPException(400, "No video found for this ad_id.")
    video_path = vids[0]

    # AUDIO
    wav_path = os.path.join(ad_dir, "audio.wav")
    if not os.path.exists(wav_path):
        extract_audio_wav(video_path, wav_path)
    ad_audio = vggish_embeddings(wav_path)
    ctx_audio = load_context_avg_arrays("audio")

    # VISUAL
    frames_dir = os.path.join(ad_dir, "frames")
    if not (os.path.exists(frames_dir) and glob.glob(os.path.join(frames_dir, "frame_*.jpg"))):
        sample_frames(video_path, frames_dir, SAMPLE_SEC)
    ad_vis = vit_embeddings_from_frames(frames_dir)
    ctx_vis = load_context_avg_arrays("visual")

    out = {}
    for show in CONTEXT_SHOWS:
        if show not in ctx_audio or show not in ctx_vis:
            continue
        Ra = sliding_congruence(ctx_audio[show], ad_audio, WINDOW_SEC, STEP_SEC, SAMPLE_SEC)
        Rv = sliding_congruence(ctx_vis[show],   ad_vis,   WINDOW_SEC, STEP_SEC, SAMPLE_SEC)
        mean_a = float(np.mean(Ra[:,2])) if Ra.size else float("nan")
        mean_v = float(np.mean(Rv[:,2])) if Rv.size else float("nan")
        score  = mean_a if combine=="audio" else mean_v if combine=="visual" else float(np.nanmean([mean_a, mean_v]))
        out[show] = {"audio": mean_a, "visual": mean_v, "combined": score}

    key = {"mean":"combined","audio":"audio","visual":"visual"}[combine]
    ranking = sorted(out.items(), key=lambda kv: kv[1][key], reverse=True)
    return {"ad_id": ad_id, "metric": key, "ranking": [{"show": s, **m} for s,m in ranking]}


if __name__=="__main__":
    import uvicorn
    uvicorn.run(app,host="0.0.0.0",port=int(os.getenv("PORT",8000)))

