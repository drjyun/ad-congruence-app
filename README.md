---
title: Ad-Context Congruence
emoji: 🎬
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# 🎬 Ad–Context Congruence Analysis

This Hugging Face Space analyzes how well your advertisement matches different TV show contexts using **audio and visual AI models**.

## 🎯 What it does

Upload a video advertisement, and the system will:
1. Extract **audio features** using VGGish (Google's audio embedding model)
2. Extract **visual features** using ViT (Vision Transformer)
3. Compare against 7 pre-recorded TV show contexts:
   - CNN
   - Fox News
   - NFL
   - LEGO
   - Pickers
   - Continental
   - Big Mood

4. Compute **sliding-window congruence scores** (Fisher-z transformed correlations)
5. Rank all shows by how well they match your ad

## 📊 Output

- **Time-resolved plots**: See how congruence changes throughout your ad
- **Rankings table**: Compare your ad across all TV contexts
- **Audio + Visual analysis**: Get separate insights for each modality

## 🚀 How to use

1. Upload your ad video (MP4/MOV format)
2. Select a TV show to visualize
3. Choose aggregation method (ad_time or tv_time)
4. Click "Analyze"
5. View results: plots, rankings, and downloadable data

## 🔬 Technical Details

- **Window size**: 10 seconds
- **Step size**: 5 seconds  
- **Sample rate**: 0.96 seconds per frame
- **Audio model**: VGGish (128-dim embeddings)
- **Visual model**: ViT-base-patch16-224 (768-dim embeddings)
- **Similarity metric**: Cosine similarity → Fisher-z transformation

## 📝 Citation

If you use this tool in your research, please cite:

```
@software{ad_context_congruence,
  title = {Ad-Context Congruence Analysis Tool},
  year = {2024},
  url = {https://huggingface.co/spaces/YOUR_USERNAME/ad-congruence-app}
}
```

## 🔧 Local Development

```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/ad-congruence-app
cd ad-congruence-app
pip install -r requirements.txt
python app.py
```

## 📄 License

MIT License - Feel free to use and modify!

