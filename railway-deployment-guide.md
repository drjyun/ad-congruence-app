# 🚂 Railway.app Deployment Guide

## Why Railway?
- **$5 free credit/month** (sufficient for testing/moderate usage)
- **Easy GitHub integration** 
- **Better for ML models** (TensorFlow + PyTorch)
- **No complex token management**
- **Simple environment variable setup**

## Prerequisites
1. GitHub account (you have: https://github.com/drjyun/ad-congruence-app)
2. Railway account (sign up with GitHub): https://railway.app

## Deployment Steps

### 1. Sign Up for Railway
- Go to: https://railway.app
- Click "Start a New Project"
- Sign in with GitHub

### 2. Create New Project from GitHub
- Click "Deploy from GitHub repo"
- Select: `drjyun/ad-congruence-app`
- Railway will auto-detect your Dockerfile

### 3. Configure Service
Railway will create TWO services (API + Frontend):

#### Service 1: API Backend
- **Dockerfile:** Select `Dockerfile.api`
- **Environment Variables:**
  - `PORT=8000`
  - `DATA_ROOT=/app`

#### Service 2: Streamlit Frontend
- **Dockerfile:** Select `Dockerfile`
- **Environment Variables:**
  - `PORT=8080`
  - `BASE_URL=<API_SERVICE_URL>` (Railway will provide this)

### 4. Deploy!
- Click "Deploy"
- Wait 5-10 minutes for build
- Railway will give you public URLs

## Cost Estimate
- **Free tier:** $5 credit = ~500 hours
- **Your usage:** Probably 100-200 hours/month = $1-2
- **Well within free tier!**

## Alternative: Single Service Deployment

If you want simpler setup, we can combine into ONE service:
- Use the Gradio app (app.py) instead
- Single container, easier to manage
- Still uses same ML models

