# 🚀 Deploy to Hugging Face Spaces (RECOMMENDED)

## Why Hugging Face Spaces?

✅ **No image size limits** (Railway free = 500MB, TensorFlow+PyTorch = 2-3GB)  
✅ **No build timeout** (Railway free = 10 min, HF = unlimited)  
✅ **Free GPU available** (optional, makes app much faster!)  
✅ **Built for ML apps** (TensorFlow, PyTorch, Gradio native support)  
✅ **100% FREE forever**  

---

## 📝 Step-by-Step Deployment

### 1️⃣ Create Hugging Face Account
- Go to https://huggingface.co/join
- Sign up (free)

### 2️⃣ Create Access Token
- Go to https://huggingface.co/settings/tokens
- Click "New token"
- Name: `ad-congruence-deploy`
- Type: **Write** access
- Click "Generate"
- **Copy the token** (you'll need it once)

### 3️⃣ Push to Hugging Face

**Option A: Via Git (Recommended)**

```bash
# Login to Hugging Face
git config --global credential.helper store

# Add Hugging Face remote
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/ad-congruence

# Push to Hugging Face
git push hf master:main
```

When prompted for password, **paste your access token** (not your password).

**Option B: Via Web Interface**

1. Go to https://huggingface.co/spaces
2. Click "Create new Space"
3. Settings:
   - **Name:** `ad-congruence`
   - **SDK:** Gradio
   - **Visibility:** Public (or Private)
   - **Hardware:** CPU Basic (free) or upgrade to T4 GPU
4. Click "Create Space"
5. Upload files:
   - `app.py`
   - `requirements.txt`
   - `README.md`
   - `packages.txt`
   - `audio/` folder (with .npy files)
   - `visual/` folder (with .npy files)

### 4️⃣ Wait for Build (5-10 minutes)

Hugging Face will automatically:
- Install system packages from `packages.txt`
- Install Python packages from `requirements.txt`
- Run `app.py`
- Deploy your Gradio interface

### 5️⃣ Access Your App

Your app will be live at:
```
https://huggingface.co/spaces/YOUR_USERNAME/ad-congruence
```

---

## 🎛️ Configuration Files

All required files are already in your repo:

| File | Purpose | Status |
|------|---------|--------|
| `README.md` | Space metadata | ✅ |
| `packages.txt` | System dependencies (ffmpeg) | ✅ |
| `requirements.txt` | Python packages | ✅ |
| `app.py` | Main Gradio app | ✅ |
| `audio/*.npy` | Audio embeddings | ✅ |
| `visual/*.npy` | Visual embeddings | ✅ |

---

## ⚡ Optional: Enable GPU (Faster Processing)

Free tier: CPU Basic (slow but free)  
Upgrade: T4 GPU (~$0.60/hour, only when running)

**With GPU:**
- Video processing: **2-3x faster**
- Cold start: **Faster model loading**
- Worth it for production!

To enable GPU:
1. Go to your Space settings
2. Click "Change hardware"
3. Select "T4 small" or "T4 medium"
4. Click "Update"

---

## 🐛 Troubleshooting

### Build fails with "out of memory"
**Solution:** The build will still succeed, just takes longer. Hugging Face has unlimited build time.

### "Application startup error"
**Solution:** Check that all `.npy` files are uploaded correctly. They should be in:
- `audio/show1_audio.npy`
- `audio/show2_audio.npy`
- etc.

### "No module named 'tensorflow'"
**Solution:** Make sure `requirements.txt` includes all dependencies. It should have:
```
tensorflow==2.15.0
torch==2.2.2
torchvision==0.17.2
transformers==4.44.2
tensorflow-hub==0.16.1
gradio==4.44.0
librosa==0.10.2.post1
...
```

---

## 📊 Comparison: Railway vs Hugging Face

| Feature | Railway Free | Hugging Face Spaces |
|---------|--------------|---------------------|
| **Image size limit** | 500MB ❌ | Unlimited ✅ |
| **Build timeout** | ~10 min ⚠️ | Unlimited ✅ |
| **ML framework support** | Generic | Optimized ⭐ |
| **GPU** | No | Yes (paid) ✅ |
| **Cost** | Free | Free ✅ |
| **Best for** | Web apps | ML apps ⭐ |

**Verdict:** Hugging Face Spaces is **perfect** for your TensorFlow + PyTorch app! 🎉

---

## 🎉 Summary

1. Create HF account
2. Create access token
3. Push code to Hugging Face
4. Wait 5-10 minutes
5. **App is LIVE!** 🚀

**Your app will be at:**  
`https://huggingface.co/spaces/YOUR_USERNAME/ad-congruence`

