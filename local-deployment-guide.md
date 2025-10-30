# 🏠 Self-Hosted Deployment Guide

## Option 1: Local Machine + Cloudflare Tunnel (100% FREE)

### What You Get:
- ✅ Run on your own computer (Windows/Mac/Linux)
- ✅ Public URL (e.g., `your-app.trycloudflare.com`)
- ✅ No port forwarding needed
- ✅ Secure HTTPS automatically
- ✅ Can turn off when not needed
- ✅ Completely FREE

### Prerequisites:
- Your Windows PC
- Internet connection
- That's it!

---

## 🚀 Quick Start (15 minutes)

### Step 1: Install Cloudflared

**Windows:**
```powershell
# Download cloudflared
Invoke-WebRequest -Uri "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe" -OutFile "cloudflared.exe"

# Move to a permanent location
Move-Item cloudflared.exe C:\Windows\System32\cloudflared.exe
```

### Step 2: Run Your App Locally

```powershell
cd "D:\TV-ad Congruence\ad-congruence-app-main"

# Install dependencies (one-time)
pip install -r requirements.txt

# Run the Gradio app
python app.py
```

Your app will start on: http://localhost:7860

### Step 3: Expose to Internet with Cloudflare

**Open a NEW terminal** and run:
```powershell
cloudflared tunnel --url http://localhost:7860
```

You'll get a public URL like:
```
https://your-random-name.trycloudflare.com
```

**Share this URL with anyone!** 🎉

---

## 📊 Comparison: Platforms vs Self-Hosted

| Feature | Railway/HF | Self-Hosted (Local) | Self-Hosted (VPS) |
|---------|-----------|---------------------|-------------------|
| **Cost** | $0-5/mo | $0 | $4-6/mo |
| **Setup Time** | 10 min | 5 min | 30 min |
| **Uptime** | 99.9% | When PC on | 99.9% |
| **Performance** | Good | Excellent | Excellent |
| **Control** | Limited | Full | Full |
| **RAM Limits** | 512MB-4GB | Your PC RAM | VPS RAM |

---

## 🔧 Option 2: Self-Hosted on VPS

If you want 24/7 uptime, here's the setup:

### Recommended VPS Providers:
1. **Hetzner Cloud** - €4/mo (4GB RAM, 2 vCPU) - BEST VALUE
2. **DigitalOcean** - $6/mo (1GB RAM) - Most popular
3. **Vultr** - $6/mo (1GB RAM)
4. **Linode** - $5/mo (1GB RAM)

### Quick VPS Setup:

```bash
# 1. SSH into your VPS
ssh root@your-vps-ip

# 2. Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh

# 3. Clone your repo
git clone https://github.com/drjyun/ad-congruence-app.git
cd ad-congruence-app

# 4. Run with Docker
docker build -f Dockerfile.gradio -t ad-app .
docker run -d -p 80:8080 --restart unless-stopped ad-app

# 5. Access at: http://your-vps-ip
```

### Add Domain (Optional):
- Point your domain to VPS IP
- Install Caddy/Nginx for HTTPS

---

## 🏡 Option 3: Home Server (24/7)

Use a spare computer or Raspberry Pi:

```bash
# Same as VPS setup, but on your home machine
# Use Cloudflare Tunnel for public access (no port forwarding)
```

---

## 🎯 My Recommendation

**For you right now:**
👉 **Start with Option 1 (Local + Cloudflare Tunnel)**

**Why?**
- ✅ Works in 5 minutes
- ✅ Completely FREE
- ✅ Test everything before committing to VPS
- ✅ Can upgrade to VPS later if needed

**Then if you need 24/7:**
👉 **Upgrade to Hetzner VPS** (€4/mo, best value)

---

## 🔐 Security Notes

**Local Deployment:**
- ✅ Cloudflare Tunnel is secure (encrypted)
- ✅ No ports opened on your router
- ✅ Can stop anytime

**VPS Deployment:**
- 🔒 Use firewall (ufw)
- 🔒 Use HTTPS (Caddy/Let's Encrypt)
- 🔒 Keep system updated

---

## 💡 Pro Tips

1. **For development:** Local + Cloudflare Tunnel
2. **For small audience:** Same as above
3. **For production:** VPS with domain + HTTPS
4. **For scale:** Use Railway/Cloud Run with auto-scaling

