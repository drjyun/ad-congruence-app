# 🏠 Self-Hosting Guide

Deploy your Ad-Context Congruence app on your own infrastructure - **FREE or cheap!**

---

## 🚀 Quick Start (5 Minutes - LOCAL)

### Option 1: Run Locally (Private Access Only)

**Simply double-click:** `run-local.bat`

Your app will be available at: http://localhost:7860

✅ **Use this for:** Testing, personal use on your own machine

---

### Option 2: Run Locally + Public Access (FREE)

**Step 1:** Double-click: `install-cloudflare-tunnel.bat`  
**Step 2:** Double-click: `run-public.bat`

You'll get a public URL like: `https://your-name.trycloudflare.com`

✅ **Use this for:** Sharing with others, demos, public access  
✅ **Cost:** $0  
✅ **Uptime:** When your computer is on

---

## 💰 Cost Comparison

| Deployment Method | Monthly Cost | Setup Time | Uptime |
|-------------------|--------------|------------|--------|
| **Local Only** | $0 | 1 min | When PC on |
| **Local + Cloudflare** | $0 | 5 min | When PC on |
| **Cheap VPS** | $4-6 | 30 min | 99.9% |
| **Premium VPS** | $12+ | 30 min | 99.99% |
| **Railway/Render** | $0-5 | 10 min | 99.9% |

---

## 🖥️ VPS Deployment (24/7 Uptime)

If you need always-on hosting, rent a VPS:

### Recommended Providers:

1. **Hetzner Cloud** ⭐ BEST VALUE
   - €3.79/mo (2GB RAM, 1 vCPU)
   - €7.59/mo (4GB RAM, 2 vCPU)
   - Germany/US locations
   - Sign up: https://www.hetzner.com/cloud

2. **DigitalOcean** ⭐ MOST POPULAR
   - $6/mo (1GB RAM)
   - $12/mo (2GB RAM)
   - Global locations
   - Sign up: https://www.digitalocean.com

3. **Vultr**
   - $6/mo (1GB RAM)
   - Similar to DigitalOcean

4. **Oracle Cloud** ⭐ FREE FOREVER
   - 4 ARM CPUs, 24GB RAM - **FREE**
   - Best free option but harder setup
   - Sign up: https://www.oracle.com/cloud/free/

### VPS Setup Instructions:

```bash
# 1. SSH into your VPS
ssh root@your-vps-ip

# 2. Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh

# 3. Install Git
apt update && apt install -y git

# 4. Clone your repository
git clone https://github.com/drjyun/ad-congruence-app.git
cd ad-congruence-app

# 5. Build and run
docker build -f Dockerfile.gradio -t ad-app .
docker run -d -p 80:8080 --restart unless-stopped --name ad-app ad-app

# 6. Check it's running
docker logs ad-app
```

**Access at:** `http://your-vps-ip`

### Add HTTPS (Recommended):

```bash
# Install Caddy (auto HTTPS)
apt install -y debian-keyring debian-archive-keyring apt-transport-https
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | tee /etc/apt/sources.list.d/caddy-stable.list
apt update && apt install caddy

# Create Caddyfile
cat > /etc/caddy/Caddyfile <<EOF
your-domain.com {
    reverse_proxy localhost:8080
}
EOF

# Restart Caddy
systemctl restart caddy
```

---

## 🏡 Home Server (Raspberry Pi / Old PC)

Use spare hardware for 24/7 hosting:

### Hardware Options:

1. **Raspberry Pi 4** (4GB+ RAM recommended)
2. **Old laptop/desktop**
3. **NAS device**

### Setup:

```bash
# Same as VPS setup above
# Use Cloudflare Tunnel for public access (no port forwarding needed!)
```

---

## 🔐 Security Best Practices

### For Local Deployment:
- ✅ Use Cloudflare Tunnel (no port forwarding)
- ✅ Don't expose ports directly to internet
- ✅ Can turn off when not needed

### For VPS Deployment:
```bash
# Enable firewall
ufw allow 22
ufw allow 80
ufw allow 443
ufw enable

# Auto-update security patches
apt install unattended-upgrades
dpkg-reconfigure -plow unattended-upgrades

# Use SSH keys (not passwords)
ssh-keygen -t ed25519
# Add public key to VPS
```

---

## 📊 Performance Tips

### Optimize for Your Hardware:

**Low RAM (2GB or less):**
- Use CPU-only mode (already default)
- Reduce batch sizes in app.py
- Consider using smaller models

**High RAM (8GB+):**
- Can handle multiple concurrent users
- No optimizations needed

---

## 🔄 Updating Your App

### Local Deployment:
```powershell
cd "D:\TV-ad Congruence\ad-congruence-app-main"
git pull origin master
# Restart run-local.bat or run-public.bat
```

### VPS Deployment:
```bash
ssh root@your-vps-ip
cd ad-congruence-app
git pull origin master
docker build -f Dockerfile.gradio -t ad-app .
docker stop ad-app && docker rm ad-app
docker run -d -p 80:8080 --restart unless-stopped --name ad-app ad-app
```

---

## 🆘 Troubleshooting

### App won't start:
```powershell
# Check Python version (need 3.10+)
python --version

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### Cloudflare Tunnel not working:
```powershell
# Re-download cloudflared
del cloudflared.exe
# Run install-cloudflare-tunnel.bat again
```

### Out of memory:
- Close other applications
- Restart your computer
- Consider VPS with more RAM

---

## 💡 Which Option Should You Choose?

**Choose LOCAL (free) if:**
- 👤 Personal use or small audience
- 💰 Want zero cost
- 🏠 Have reliable internet at home
- 📱 Don't need 24/7 availability

**Choose VPS ($4-6/mo) if:**
- 🌍 Need 24/7 availability
- 👥 Serving many users
- 🚀 Want professional setup
- 📊 Need better performance

**Choose Cloud Platform ($0-5/mo) if:**
- 🎯 Want easiest setup
- 📈 Need auto-scaling
- 🔧 Don't want to manage servers
- 💳 Okay with platform lock-in

---

## 🎯 My Recommendation

**For you right now:**

1. **Test locally:** Run `run-local.bat` (1 minute)
2. **Share publicly:** Run `run-public.bat` (5 minutes)
3. **If you like it:** Get Hetzner VPS (€4/mo) for 24/7 hosting

**Start free, upgrade only if needed!** 🚀

