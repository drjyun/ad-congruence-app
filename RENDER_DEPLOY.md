# 🚀 Deploy to Render.com (Private Code + Public App)

## ✅ Why Render.com?

- ✅ **FREE tier** with private repos
- ✅ **Private GitHub repo** + Public deployment
- ✅ 15-minute build timeout (vs Railway 10 min)
- ✅ Similar to Railway but more generous limits
- ✅ **Protects your intellectual property**

---

## 📝 Deployment Steps

### 1️⃣ Make GitHub Repo Private (Protect IP)

**Option A: GitHub.com Website**
1. Go to your repo: https://github.com/Techotkxy/ad-congruence-app
2. Click "Settings" (top right)
3. Scroll to "Danger Zone"
4. Click "Change visibility"
5. Select "Make private"
6. Confirm

**Option B: Via Command Line**
```bash
# Use GitHub CLI (if installed)
gh repo edit --visibility private
```

### 2️⃣ Sign Up for Render.com

1. Go to https://render.com
2. Click "Get Started for Free"
3. Sign up with **GitHub account** (easiest)
4. Authorize Render to access your private repos

### 3️⃣ Create Web Service

1. Click "New +" → "Web Service"
2. Connect your **private GitHub repo**: `Techotkxy/ad-congruence-app`
3. Configure:
   - **Name:** `ad-congruence-app`
   - **Region:** Choose closest to you
   - **Branch:** `main`
   - **Runtime:** Docker
   - **Dockerfile Path:** `Dockerfile.minimal`
   - **Instance Type:** Free

4. Click "Create Web Service"

### 4️⃣ Wait for Build (10-15 minutes)

Render will:
- Build Docker image from your **private repo**
- Install all dependencies
- Deploy your app
- Give you a public URL

### 5️⃣ Access Your App

Your app will be live at:
```
https://ad-congruence-app.onrender.com
```

**Code remains private!** ✅ Only the web interface is public.

---

## ⚙️ Configuration

### Using Dockerfile.minimal

Already configured in your repo. Render will auto-detect it.

If build times out, try splitting Docker layers:

```dockerfile
# Install TensorFlow CPU (smaller)
RUN pip install --no-cache-dir tensorflow-cpu==2.15.0

# Install PyTorch CPU-only
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

---

## 🐛 Troubleshooting

### Build timeout (>15 min)

**Solution 1: Use minimal Dockerfile**
- Already created: `Dockerfile.minimal`
- Uses `tensorflow-cpu` (smaller)
- Strips test files and caches

**Solution 2: Pre-build and push to Docker Hub**
```bash
# Build locally
docker build -t yourusername/ad-congruence:latest -f Dockerfile.minimal .

# Push to Docker Hub
docker push yourusername/ad-congruence:latest

# Then in Render, use: yourusername/ad-congruence:latest
```

### "Image too large"

Render free tier is more generous than Railway, but if it still fails:

**Solution:** Strip more from Docker image
```dockerfile
# Remove test files, docs, and caches
RUN find /usr/local/lib/python3.10 -type d -name "tests" -exec rm -rf {} + \
    && find /usr/local/lib/python3.10 -type d -name "__pycache__" -exec rm -rf {} + \
    && pip cache purge
```

---

## 💰 Cost Comparison

| Tier | Cost | RAM | Build Time | Image Size |
|------|------|-----|------------|------------|
| **Free** | $0 | 512MB | 15 min | ~1GB |
| **Starter** | $7/mo | 512MB | 30 min | Unlimited |

**Free tier should work!** If not, Starter is $7/mo (similar to Railway Hobby $5/mo)

---

## 🔒 IP Protection Verification

**Your code is protected because:**
1. ✅ GitHub repo is **private**
2. ✅ Render deploys from private repo (authorized access only)
3. ✅ Only **compiled Python bytecode** runs on server
4. ✅ Public can only access **web interface**, not code
5. ✅ Source code never exposed

**To verify:**
- Try accessing: `https://ad-congruence-app.onrender.com/app.py` → 404 Error ✅
- Your GitHub repo is private → Only you can see code ✅

---

## 📊 Render.com vs Railway vs HuggingFace

| Feature | Render Free | Railway Free | Railway Hobby | HuggingFace |
|---------|-------------|--------------|---------------|-------------|
| **Private code** | ✅ | ✅ | ✅ | ❌ |
| **Image size** | ~1GB | 500MB ❌ | Unlimited ✅ | Unlimited ✅ |
| **Build timeout** | 15 min | 10 min ⚠️ | 30 min ✅ | Unlimited ✅ |
| **Cost** | FREE ✅ | FREE | $5/mo | FREE |
| **IP Protection** | ✅ | ✅ | ✅ | ❌ (public) |

**Winner for FREE + Private:** Render.com ⭐

---

## 🎯 Alternative: Railway Hobby (If Render Fails)

If Render free tier doesn't work (unlikely):

**Railway Hobby ($5/mo):**
- Unlimited image size
- 30-minute builds
- 8GB RAM
- Worth it for production/research

Upgrade at: https://railway.app/pricing

---

## 📖 Summary

**Deployment flow:**
1. Make GitHub repo private (protect IP)
2. Connect Render to private repo
3. Deploy from `Dockerfile.minimal`
4. Get public URL
5. Code stays private! ✅

**Expected timeline:**
- Setup: 5 minutes
- Build: 10-15 minutes
- **Total: ~20 minutes to live deployment!** 🚀

**Your app:** `https://ad-congruence-app.onrender.com`  
**Your code:** Private GitHub repo (protected) ✅

