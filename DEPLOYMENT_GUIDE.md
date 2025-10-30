# 🚀 Deployment Guide for Hugging Face Spaces

## 📦 Files Created

I've created the following files for your Hugging Face Space:

1. **app.py** - Main Gradio application (converted from FastAPI + Streamlit)
2. **requirements.txt** - Python dependencies
3. **README.md** - Space description and metadata
4. **.gitattributes** - Git LFS configuration for large files
5. **packages.txt** - System dependencies (ffmpeg, libsndfile)

## 📋 Deployment Steps

### Option 1: Using Git (Recommended)

1. **Clone your Space repository:**
```bash
git clone https://huggingface.co/spaces/jiangxy/ad-congruence-app
cd ad-congruence-app
```

2. **Copy all files from your local project:**
```bash
# Copy the new files
cp /path/to/your/project/app.py .
cp /path/to/your/project/requirements.txt .
cp /path/to/your/project/README.md .
cp /path/to/your/project/.gitattributes .
cp /path/to/your/project/packages.txt .

# Copy the audio and visual feature directories
cp -r /path/to/your/project/audio .
cp -r /path/to/your/project/visual .
```

3. **Initialize Git LFS (for large .npy files):**
```bash
git lfs install
git lfs track "*.npy"
```

4. **Commit and push:**
```bash
git add .
git commit -m "Add ad-congruence application"
git push
```

### Option 2: Using Hugging Face Web Interface

1. **Go to your Space:** https://huggingface.co/spaces/jiangxy/ad-congruence-app
2. **Click "Files" tab**
3. **Upload files one by one:**
   - Click "Add file" → "Upload files"
   - Upload: `app.py`, `requirements.txt`, `README.md`, `.gitattributes`, `packages.txt`
4. **Create directories and upload .npy files:**
   - Click "Add file" → "Create a new file"
   - Enter path: `audio/BIGMOOD_1_features.npy`
   - Upload the file
   - Repeat for all .npy files in audio/ and visual/ folders

### Option 3: Using Hugging Face CLI

1. **Install the CLI:**
```bash
pip install huggingface_hub
```

2. **Login:**
```bash
huggingface-cli login
```

3. **Upload files:**
```bash
huggingface-cli upload jiangxy/ad-congruence-app app.py app.py
huggingface-cli upload jiangxy/ad-congruence-app requirements.txt requirements.txt
huggingface-cli upload jiangxy/ad-congruence-app README.md README.md
huggingface-cli upload jiangxy/ad-congruence-app packages.txt packages.txt
huggingface-cli upload jiangxy/ad-congruence-app audio/ audio/ --repo-type space
huggingface-cli upload jiangxy/ad-congruence-app visual/ visual/ --repo-type space
```

## 🔧 After Deployment

1. **Wait for build** (5-10 minutes for first build)
2. **Check logs** in the "Logs" tab if there are errors
3. **Test your Space** by uploading a sample video
4. **Share your Space URL:** https://huggingface.co/spaces/jiangxy/ad-congruence-app

## 🐛 Troubleshooting

### If the Space fails to build:

**Error: "Out of memory"**
- Upgrade to ZeroGPU hardware (still free, just requires authentication)
- Settings → Hardware → Select "ZeroGPU"

**Error: "ffmpeg not found"**
- Make sure `packages.txt` is uploaded correctly
- Check build logs to ensure system packages are installed

**Error: "Cannot load .npy files"**
- Verify all .npy files are uploaded to correct directories
- Check that Git LFS is tracking the files

**Error: "Module not found"**
- Check `requirements.txt` is present and complete
- Look at build logs for failed pip installs

### If models are slow to load:

- **First run is always slow** (downloading VGGish + ViT models)
- Models are cached after first load
- Consider upgrading to persistent storage if needed

## 📊 Monitoring

- **Usage**: Check "Analytics" tab
- **Logs**: View real-time logs in "Logs" tab
- **Settings**: Adjust hardware, visibility, etc.

## 🎉 Next Steps

1. **Test thoroughly** with different videos
2. **Share** your Space URL with users
3. **Upgrade hardware** if you need faster inference (ZeroGPU is still free!)
4. **Add examples** by including sample videos in your repo
5. **Monitor usage** and adjust as needed

---

Need help? Check:
- [Hugging Face Spaces Docs](https://huggingface.co/docs/hub/spaces)
- [Gradio Docs](https://gradio.app/docs/)

