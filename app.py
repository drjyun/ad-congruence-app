import os
import io
import uuid
import glob
import shutil
import subprocess
import numpy as np
import gradio as gr
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
import pandas as pd
from datetime import datetime

# ===== CONFIG =====
DATA_ROOT = os.getcwd()
UPLOAD_ROOT = os.path.join(DATA_ROOT, "uploads")
AUDIO_DIR = os.path.join(DATA_ROOT, "audio")
VISUAL_DIR = os.path.join(DATA_ROOT, "visual")
CONTEXT_SHOWS = ["CNN", "FOXNEWS", "NFL", "LEGO", "PICKERS", "CONTINENTAL", "BIGMOOD"]
SAMPLE_SEC = 0.96
WINDOW_SEC = 10.0
STEP_SEC = 5.0

for d in (UPLOAD_ROOT, AUDIO_DIR, VISUAL_DIR):
    os.makedirs(d, exist_ok=True)

# ===== HELPER FUNCTIONS =====
def sliding_indices(n, win, step):
    return range(0, max(n - win + 1, 0), step)

def fisher_z(r):
    r = np.clip(r, -0.999999, 0.999999)
    return 0.5 * np.log((1 + r) / (1 - r))

# ===== LAZY MODEL LOADING =====
vggish = None
vit, fe, transform = None, None, None
device = torch.device("cpu")

def get_vggish():
    global vggish
    if vggish is None:
        print("Loading VGGish from TF-Hub…")
        # Set cache directory to persist models
        os.environ['TFHUB_CACHE_DIR'] = '/tmp/tfhub_cache'
        vggish = hub.load("https://tfhub.dev/google/vggish/1")
        print("VGGish ready.")
    return vggish

def get_vit():
    global vit, fe, transform
    if vit is None:
        print("Loading ViT…")
        # Cache HuggingFace models
        os.environ['TRANSFORMERS_CACHE'] = '/tmp/transformers_cache'
        fe = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
        vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k").to(device)
        vit.eval()  # Set to evaluation mode for speed
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=fe.image_mean, std=fe.image_std)
        ])
        print("ViT ready.")
    return vit, fe, transform

# ===== AUDIO PROCESSING =====
def extract_audio_wav(video_path, wav_path):
    subprocess.run([
        "ffmpeg", "-y", "-i", video_path, "-ar", "16000", "-ac", "1", wav_path
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def vggish_embeddings(wav_path):
    y, sr = librosa.load(wav_path, sr=16000)
    model = get_vggish()
    audio_tensor = tf.convert_to_tensor(y, dtype=tf.float32)
    emb = model(audio_tensor)
    return emb.numpy()

# ===== VISUAL PROCESSING =====
def sample_frames(video_path, out_dir, sample_sec=0.96):
    os.makedirs(out_dir, exist_ok=True)
    fps = 1.0 / sample_sec
    pattern = os.path.join(out_dir, "frame_%05d.jpg")
    subprocess.run(["ffmpeg", "-y", "-i", video_path, "-vf", f"fps={fps}", pattern],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def vit_embeddings_from_frames(frame_dir, batch_size=16):  # Increased batch size
    vit_model, fe_proc, tfm = get_vit()
    frames = sorted(glob.glob(os.path.join(frame_dir, "frame_*.jpg")))
    if not frames:
        return np.empty((0, 768))

    class FrameDataset(Dataset):
        def __init__(self, paths):
            self.paths = paths
        def __len__(self):
            return len(self.paths)
        def __getitem__(self, i):
            return tfm(Image.open(self.paths[i]).convert("RGB"))

    dl = DataLoader(FrameDataset(frames), batch_size=batch_size, shuffle=False, 
                    num_workers=0, pin_memory=False)  # Optimized for CPU
    feats = []
    vit_model.eval()
    with torch.no_grad():
        for batch in dl:
            batch = batch.to(device)
            out = vit_model(batch).last_hidden_state.mean(dim=1).cpu().numpy()
            feats.append(out)
    return np.vstack(feats)

# ===== CONTEXT LOADING =====
# Cache context arrays globally (load once, reuse)
_cached_audio_ctx = None
_cached_visual_ctx = None

def load_context_avg_arrays(kind):
    global _cached_audio_ctx, _cached_visual_ctx
    
    # Return cached if available
    if kind == "audio" and _cached_audio_ctx is not None:
        return _cached_audio_ctx
    if kind == "visual" and _cached_visual_ctx is not None:
        return _cached_visual_ctx
    
    # Check both subdirectory and root directory for feature files
    d = AUDIO_DIR if kind == "audio" else VISUAL_DIR
    out = {}
    print(f"Loading {kind} context features...")
    for show in CONTEXT_SHOWS:
        pattern = f"{show}_*_features.npy" if kind == "audio" else f"{show}_*_sampled_960ms_features.npy"
        
        # Try subdirectory first
        segs = sorted(glob.glob(os.path.join(d, pattern)))
        
        # If not found in subdirectory, try root directory
        if not segs:
            segs = sorted(glob.glob(os.path.join(DATA_ROOT, pattern)))
        
        arrs = [np.load(p) for p in segs]
        if arrs:
            out[show] = arrs
    
    if not out:
        raise RuntimeError(f"No context features for {kind}. Searched in {d} and {DATA_ROOT}")
    
    # Cache for future use
    if kind == "audio":
        _cached_audio_ctx = out
    else:
        _cached_visual_ctx = out
    
    print(f"✓ Loaded {len(out)} {kind} contexts")
    return out

# ===== CONGRUENCE CALCULATION =====
def sliding_congruence(tv_arrays, ad_array, window_sec, step_sec, chunk_dur):
    W = max(int(round(window_sec / chunk_dur)), 1)
    S = max(int(round(step_sec / chunk_dur)), 1)
    
    # Precompute ad windows ONCE (flattened and normalized)
    ad_indices = list(sliding_indices(ad_array.shape[0], W, S))
    n_ad_windows = len(ad_indices)
    
    # Create matrix of all ad windows at once (vectorized!)
    ad_windows_matrix = np.zeros((n_ad_windows, W * ad_array.shape[1]))
    for i, a0 in enumerate(ad_indices):
        ad_windows_matrix[i] = ad_array[a0:a0 + W].flatten()
    
    # Precompute norms for all ad windows at once
    ad_norms = np.linalg.norm(ad_windows_matrix, axis=1, keepdims=True)
    ad_windows_normalized = ad_windows_matrix / (ad_norms + 1e-8)
    
    # Store results efficiently
    results = []
    
    for tv in tv_arrays:
        n_tv = tv.shape[0]
        tv_indices = list(sliding_indices(n_tv, W, S))
        n_tv_windows = len(tv_indices)
        
        # Create matrix of all TV windows at once
        tv_windows_matrix = np.zeros((n_tv_windows, W * tv.shape[1]))
        for i, t0 in enumerate(tv_indices):
            tv_windows_matrix[i] = tv[t0:t0 + W].flatten()
        
        # Compute norms for TV windows
        tv_norms = np.linalg.norm(tv_windows_matrix, axis=1, keepdims=True)
        tv_windows_normalized = tv_windows_matrix / (tv_norms + 1e-8)
        
        # VECTORIZED: Compute ALL correlations at once (matrix multiplication)
        # Shape: (n_tv_windows, n_ad_windows)
        correlations = tv_windows_normalized @ ad_windows_normalized.T
        
        # Apply Fisher-z transformation (vectorized)
        correlations_clipped = np.clip(correlations, -0.999999, 0.999999)
        z_scores = 0.5 * np.log((1 + correlations_clipped) / (1 - correlations_clipped))
        
        # Store results for this TV segment
        for i, t0 in enumerate(tv_indices):
            for j, a0 in enumerate(ad_indices):
                results.append([t0 * chunk_dur, a0 * chunk_dur, z_scores[i, j]])
    
    # Average z-scores across multiple TV segments if needed
    if len(tv_arrays) > 1:
        result_dict = {}
        for t_time, a_time, z in results:
            key = (t_time, a_time)
            if key not in result_dict:
                result_dict[key] = []
            result_dict[key].append(z)
        
        final_results = [[t, a, float(np.mean(zs))] for (t, a), zs in result_dict.items()]
        return np.array(final_results, float)
    
    return np.array(results, float)

# ===== PLOTTING =====
def create_plot(rows, show, agg, modality):
    tv_t0, ad_t0, z = rows[:, 0], rows[:, 1], rows[:, 2]
    if agg == "ad_time":
        xs = sorted(set(ad_t0))
        xlabel = "Ad time (s)"
        vals = [z[ad_t0 == t] for t in xs]
    else:
        xs = sorted(set(tv_t0))
        xlabel = "TV context time (s)"
        vals = [z[tv_t0 == t] for t in xs]
    
    means = [float(np.mean(v)) for v in vals]
    cis = [1.96 * float(np.std(v, ddof=1) / max(np.sqrt(len(v)), 1)) for v in vals]
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(xs, means, linewidth=2, label=f"{modality} congruence")
    ax.fill_between(xs, np.array(means) - np.array(cis), 
                     np.array(means) + np.array(cis), alpha=0.2)
    ax.set_title(f"{show} – {modality} congruence ({agg})")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Fisher-z score")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    return fig

def create_combined_plot(rows_a, rows_v, show, agg):
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=False)
    
    for ax, rows, modality in [(axes[0], rows_a, "Audio"), (axes[1], rows_v, "Visual")]:
        tv_t0, ad_t0, z = rows[:, 0], rows[:, 1], rows[:, 2]
        if agg == "ad_time":
            xs = sorted(set(ad_t0))
            xlabel = "Ad time (s)"
            vals = [z[ad_t0 == t] for t in xs]
        else:
            xs = sorted(set(tv_t0))
            xlabel = "TV context time (s)"
            vals = [z[tv_t0 == t] for t in xs]
        
        means = [float(np.mean(v)) for v in vals]
        cis = [1.96 * float(np.std(v, ddof=1) / max(np.sqrt(len(v)), 1)) for v in vals]
        
        ax.plot(xs, means, linewidth=2)
        ax.fill_between(xs, np.array(means) - np.array(cis), 
                         np.array(means) + np.array(cis), alpha=0.2)
        ax.set_title(f"{show} – {modality} congruence ({agg})")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Fisher-z score")
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

# ===== MAIN PROCESSING FUNCTION =====
def analyze_ad(video_file, show, agg, combine_metric):
    if video_file is None:
        return None, None, "Please upload a video file.", None
    
    import time
    start_time = time.time()
    
    try:
        # Create unique ID and directory
        ad_id = str(uuid.uuid4())
        ad_dir = os.path.join(UPLOAD_ROOT, ad_id)
        os.makedirs(ad_dir, exist_ok=True)
        
        # Save uploaded file
        video_path = os.path.join(ad_dir, "ad.mp4")
        shutil.copy(video_file, video_path)
        
        # Extract audio (~20s)
        yield None, None, "⏳ [1/6] Extracting audio... (est. 20s)", None
        t1 = time.time()
        wav_path = os.path.join(ad_dir, "audio.wav")
        extract_audio_wav(video_path, wav_path)
        ad_audio = vggish_embeddings(wav_path)
        audio_time = time.time() - t1
        print(f"✓ Audio: {audio_time:.1f}s")
        
        # Extract visual frames (~90s - biggest bottleneck)
        yield None, None, f"⏳ [2/6] Processing video frames... (est. 90s)", None
        t2 = time.time()
        frames_dir = os.path.join(ad_dir, "frames")
        sample_frames(video_path, frames_dir, SAMPLE_SEC)
        ad_vis = vit_embeddings_from_frames(frames_dir)
        visual_time = time.time() - t2
        print(f"✓ Visual: {visual_time:.1f}s")
        
        # Load contexts (cached, fast)
        yield None, None, "⏳ [3/6] Loading TV contexts...", None
        ctx_audio = load_context_avg_arrays("audio")
        ctx_vis = load_context_avg_arrays("visual")
        
        # Calculate congruence (~30s with vectorization)
        yield None, None, f"⏳ [4/6] Computing congruence for {show}... (est. 30s)", None
        t3 = time.time()
        rows_a = sliding_congruence(ctx_audio[show], ad_audio, WINDOW_SEC, STEP_SEC, SAMPLE_SEC)
        rows_v = sliding_congruence(ctx_vis[show], ad_vis, WINDOW_SEC, STEP_SEC, SAMPLE_SEC)
        selected_time = time.time() - t3
        print(f"✓ Selected show: {selected_time:.1f}s")
        
        # Create plot
        yield None, None, "⏳ [5/6] Creating visualization...", None
        plot = create_combined_plot(rows_a, rows_v, show, agg)
        
        # Calculate rankings for all shows (~150s for 6 remaining shows)
        # OPTIMIZATION: Reuse already computed results for selected show!
        yield None, None, "⏳ [6/6] Ranking all shows... (est. 150s)", None
        t4 = time.time()
        rankings = []
        
        # Create a cache to store computed results
        congruence_cache = {
            show: {'audio': rows_a, 'visual': rows_v}  # Reuse selected show results!
        }
        
        remaining_shows = [s for s in CONTEXT_SHOWS if s != show and s in ctx_audio and s in ctx_vis]
        for idx, s in enumerate(CONTEXT_SHOWS):
            if s not in ctx_audio or s not in ctx_vis:
                continue
            
            # Show progress for each show
            if s != show:
                progress_pct = int((idx / len(CONTEXT_SHOWS)) * 100)
                yield None, None, f"⏳ [6/6] Ranking shows... {progress_pct}% ({s})", None
            
            # Check if we already computed this show
            if s in congruence_cache:
                Ra = congruence_cache[s]['audio']
                Rv = congruence_cache[s]['visual']
            else:
                # Only compute if not already done
                Ra = sliding_congruence(ctx_audio[s], ad_audio, WINDOW_SEC, STEP_SEC, SAMPLE_SEC)
                Rv = sliding_congruence(ctx_vis[s], ad_vis, WINDOW_SEC, STEP_SEC, SAMPLE_SEC)
            
            mean_a = float(np.mean(Ra[:, 2])) if Ra.size else float("nan")
            mean_v = float(np.mean(Rv[:, 2])) if Rv.size else float("nan")
            
            if combine_metric == "Audio only":
                score = mean_a
            elif combine_metric == "Visual only":
                score = mean_v
            else:  # Combined (mean of A/V)
                score = float(np.nanmean([mean_a, mean_v]))
            
            rankings.append({
                "Show": s,
                "Combined": round(float(np.nanmean([mean_a, mean_v])), 3),
                "Audio": round(mean_a, 3),
                "Visual": round(mean_v, 3)
            })
        
        ranking_time = time.time() - t4
        print(f"✓ Ranking: {ranking_time:.1f}s")
        
        df = pd.DataFrame(rankings)
        df = df.sort_values(by="Audio" if combine_metric == "Audio only" 
                            else "Visual" if combine_metric == "Visual only" 
                            else "Combined", ascending=False)
        
        total_time = time.time() - start_time
        
        info_text = f"""
### Analysis Complete ✓
**Ad ID:** `{ad_id}`
**TV Show:** {show}
**Aggregation:** {agg}
**Total Time:** {total_time:.1f}s ({total_time/60:.1f} min)
**Breakdown:**
- Audio processing: {audio_time:.1f}s
- Visual processing: {visual_time:.1f}s
- Congruence (selected): {selected_time:.1f}s
- Ranking all shows: {ranking_time:.1f}s
**Analyzed at:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

Higher Fisher-z scores indicate better congruence between your ad and the TV context.
        """
        
        yield plot, df, info_text, ad_dir
        
    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()
        print(f"❌ Error:\n{error_msg}")
        yield None, None, f"❌ Error: {str(e)}", None

# ===== GRADIO INTERFACE =====
with gr.Blocks(title="Ad–Context Congruence", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎬 Ad–Context Congruence Analysis")
    gr.Markdown("Analyze how well your advertisement matches different TV show contexts using audio and visual AI models.")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 1️⃣ Upload Your Ad")
            video_input = gr.Video(label="Upload video (.mp4, .mov)", sources=["upload"])
            
            gr.Markdown("### 2️⃣ Choose TV Context")
            show_select = gr.Dropdown(
                choices=CONTEXT_SHOWS,
                value="NFL",
                label="TV Show"
            )
            agg_select = gr.Radio(
                choices=["ad_time", "tv_time"],
                value="ad_time",
                label="Aggregation Method"
            )
            
            gr.Markdown("### 3️⃣ Ranking Options")
            combine_select = gr.Radio(
                choices=["Combined (mean of A/V)", "Audio only", "Visual only"],
                value="Combined (mean of A/V)",
                label="Ranking Metric"
            )
            
            analyze_btn = gr.Button("🚀 Analyze", variant="primary", size="lg")
        
        with gr.Column(scale=2):
            status_text = gr.Markdown("Upload a video and click Analyze to begin.")
            plot_output = gr.Plot(label="Congruence Plot")
            info_output = gr.Markdown()
    
    gr.Markdown("### 📊 Show Rankings")
    ranking_output = gr.Dataframe(label="Top Matches (by Fisher-z score)")
    
    # Hidden state for cleanup
    ad_dir_state = gr.State(None)
    
    # Event handlers
    analyze_btn.click(
        fn=analyze_ad,
        inputs=[video_input, show_select, agg_select, combine_select],
        outputs=[plot_output, ranking_output, status_text, ad_dir_state]
    )
    
    gr.Markdown("""
    ---
    ### 📖 How it works
    1. **Upload** your ad video (MP4/MOV format)
    2. **Select** a TV show context to compare against
    3. The system extracts:
       - **Audio features** using VGGish (Google's audio embedding model)
       - **Visual features** using ViT (Vision Transformer)
    4. Computes **sliding-window cosine similarity** and converts to Fisher-z scores
    5. Shows **time-resolved congruence** and ranks all shows
    
    **Higher scores = Better match between ad and TV context**
    """)

# ===== PRELOAD MODELS & DATA (Speed up first request) =====
def preload_models_and_data():
    """Warm up models and load context data at startup"""
    print("\n" + "="*60)
    print("🚀 Preloading models and data for faster response...")
    print("="*60)
    
    try:
        # Preload models
        print("📥 Loading VGGish model...")
        get_vggish()
        
        print("📥 Loading ViT model...")
        get_vit()
        
        # Preload context arrays
        print("📥 Loading audio context features...")
        load_context_avg_arrays("audio")
        
        print("📥 Loading visual context features...")
        load_context_avg_arrays("visual")
        
        print("="*60)
        print("✅ All models and data loaded! Ready for requests.")
        print("="*60 + "\n")
    except Exception as e:
        print(f"⚠️ Warning: Could not preload all resources: {e}")

if __name__ == "__main__":
    # Preload everything before starting server
    preload_models_and_data()
    
    # Get port from environment variable (for Railway deployment)
    import os
    port = int(os.getenv("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port, share=False)

