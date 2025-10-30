# 🏗️ Deployment Architecture Options

You have **3 different deployment architectures** in this repo. Choose based on your needs:

---

## ✅ **Option 1: Single Gradio App (CURRENT - RECOMMENDED)**

**Files Used:**
- ✅ `app.py` (Gradio interface + all ML logic)
- ✅ `Dockerfile.gradio`
- ✅ `railway.toml`

**Architecture:**
```
User → Gradio App (app.py) → [VGGish + ViT + Analysis] → Results
```

**Pros:**
- ✅ Simplest setup
- ✅ All-in-one container
- ✅ No API calls needed
- ✅ Best for Railway/HF Spaces

**Cons:**
- ❌ Can't separate frontend/backend scaling
- ❌ Single point of failure

**Current Railway Deployment:** Uses this!

**URL:** Just the Railway domain (e.g., `your-app.up.railway.app`)

---

## Option 2: Split Architecture (FastAPI + Streamlit)

**Files Used:**
- ⚙️ `ad_context_api.py` (FastAPI backend)
- 🎨 `streamlit_app.py` (Streamlit frontend)
- 🐳 `Dockerfile.api` (backend)
- 🐳 `Dockerfile` (frontend)

**Architecture:**
```
User → Streamlit Frontend → API calls → FastAPI Backend → [ML Processing] → Results
```

**Pros:**
- ✅ Separate frontend/backend
- ✅ Can scale independently
- ✅ API can serve multiple frontends
- ✅ Better for production

**Cons:**
- ❌ More complex setup
- ❌ Requires 2 services
- ❌ Network latency between services
- ❌ More expensive

**Deployment:**
- Backend: Railway Service 1 (Dockerfile.api)
- Frontend: Railway Service 2 (Dockerfile)
- Set `BASE_URL` env var in frontend to backend URL

---

## Option 3: Local Self-Hosting

**Files Used:**
- ✅ `app.py` or split architecture
- ✅ `run-local.bat` (local only)
- ✅ `run-public.bat` (local + Cloudflare Tunnel)

**Architecture:**
```
User → Cloudflare Tunnel → Your PC → Gradio App → Results
```

**Pros:**
- ✅ FREE
- ✅ Use your own hardware
- ✅ Full control
- ✅ Can be faster than cloud

**Cons:**
- ❌ Requires PC to be on
- ❌ Not 24/7 (unless dedicated hardware)

---

## 🎯 **Which One Are You Using?**

### **Currently on Railway:**

You're using **Option 1 (Single Gradio App)** ✅

**How to verify:**
1. Go to Railway → Settings → Build
2. Check which Dockerfile: Should be `Dockerfile.gradio`
3. No API URL needed - it's all-in-one!

---

## 🔧 **How to Switch Between Options:**

### **Switch to Split Architecture (Option 2):**

1. **Deploy Backend:**
   ```
   Railway → New Service → GitHub repo
   Settings → Build → Dockerfile: Dockerfile.api
   Generate Domain → Copy URL (e.g., api.railway.app)
   ```

2. **Deploy Frontend:**
   ```
   Railway → New Service → Same repo
   Settings → Build → Dockerfile: Dockerfile
   Variables → Add: BASE_URL = <backend-url>
   Generate Domain → This is your public URL
   ```

### **Switch to Local (Option 3):**

Just double-click `run-public.bat` on your PC!

---

## 📊 **Comparison Table:**

| Feature | Single Gradio | Split (API+UI) | Local |
|---------|---------------|----------------|-------|
| **Setup Time** | 10 min | 30 min | 5 min |
| **Cost** | $5/mo | $10/mo | $0 |
| **Services** | 1 | 2 | 1 |
| **Scalability** | Limited | Excellent | N/A |
| **Complexity** | Low | High | Low |
| **Best For** | MVP/Demo | Production | Dev/Testing |

---

## 🚨 **Common Issue: "Host is still to render"**

This error happens when:

❌ **Wrong:** Using Streamlit frontend (streamlit_app.py) with hardcoded Render.com URL

✅ **Fix:** You're using Gradio (app.py) - no external API needed!

**If using split architecture:**
1. Make sure backend is deployed first
2. Get backend URL from Railway
3. Set `BASE_URL` env var in frontend service
4. Restart frontend service

---

## 🎯 **Recommendation:**

**For your use case:** Stick with **Option 1 (Single Gradio App)** ✅

**Why?**
- Simpler
- Cheaper
- Faster (no network calls)
- Already working on Railway!

**Only use Option 2 if:**
- Need to scale frontend/backend separately
- Want to add other frontends (mobile app, etc.)
- Need API for programmatic access

---

## 📝 **Current Railway Setup Checklist:**

✅ Using `app.py` (Gradio)
✅ Using `Dockerfile.gradio`
✅ Models preloaded at startup
✅ Context arrays cached
✅ No external API calls needed

**Your app is optimized and ready!** 🚀

