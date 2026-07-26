# 🔥 Bordeaux Wildfire Risk Monitor — Deployment Guide

## VPS Deployment

### Option 1: Docker (Recommended)

```bash
# Clone repository (on your VPS)
git clone https://github.com/FelixChop/bordeaux-wildfire-monitor.git
cd bordeaux-wildfire-monitor/wildfire_risk

# Build Docker image
docker build -t wildfire-monitor .

# Run container (port 5000)
docker run -d \
  --name wildfire \
  -p 5000:8000 \
  -e NASA_FIRMS_MAP_KEY="your_api_key_here" \
  wildfire-monitor
```

### Option 2: Direct Python (No Docker)

```bash
# Install dependencies
pip install -r requirements.txt

# Get NASA FIRMS MAP_KEY from: https://firms.modaps.eosdis.nasa.gov/map_key/
export NASA_FIRMS_MAP_KEY="your_api_key_here"

# Run Flask app
python app.py
```

### Option 3: Systemd Service + Gunicorn (Production)

**Create `/etc/systemd/system/wildfire.service`:**

```ini
[Unit]
Description=Bordeaux Wildfire Risk Monitor
After=network.target

[Service]
Type=notify
User=www-data
WorkingDirectory=/opt/wildfire-monitor/wildfire_risk
ExecStart=/usr/bin/gunicorn --bind 0.0.0.0:5000 --workers 4 app:app
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

**Enable and start:**

```bash
sudo systemctl daemon-reload
sudo systemctl enable wildfire
sudo systemctl start wildfire
sudo systemctl status wildfire
```

## Nginx Reverse Proxy (Production)

**Create `/etc/nginx/sites-available/wildfire`:**

```nginx
server {
    listen 80;
    server_name wildfire.yourdomain.com;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

**Enable:**

```bash
sudo ln -s /etc/nginx/sites-available/wildfire /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

## Environment Variables

**Set on VPS:**

```bash
export NASA_FIRMS_MAP_KEY="your_free_map_key"  # Get from https://firms.modaps.eosdis.nasa.gov/map_key/
export PORT=5000
```

## Features

- ✅ Real-time fire hotspots (NASA FIRMS)
- ✅ 10-day wind forecast (Météo-France ARPEGE + GFS)
- ✅ Interactive Leaflet map (zoom/pan)
- ✅ Fire propagation zones
- ✅ Temperature overlay
- ✅ Auto-refresh every hour
- ✅ JSON API (`/api/data`)

## Logs

```bash
# Docker logs
docker logs -f wildfire

# Systemd logs
sudo journalctl -u wildfire -f

# Flask development
python app.py  # Runs on localhost:5000
```

## Performance

- **Single instance:** ~50 concurrent users
- **Data update:** Every 1 hour (configurable)
- **Memory:** ~150-200 MB
- **CPU:** < 1% idle

## Cost

- **VPS:** $5-10/month
- **NASA FIRMS:** Free (API key required)
- **Météo-France:** Free (Open-Meteo proxy)
- **Total:** ~$5-10/month

## Troubleshooting

### Map not loading
- Check browser console (F12)
- Verify Folium installed: `pip list | grep folium`
- Check Flask running: `curl localhost:5000`

### No fire data
- Verify NASA_FIRMS_MAP_KEY: https://firms.modaps.eosdis.nasa.gov/map_key/
- Check logs for fetch errors

### High memory usage
- Restart container: `docker restart wildfire`
- Check data cache size: `du -sh cache/`
- Clear old cache: `rm cache/*.json`

## Support

For issues, open GitHub issue or check logs with:
```bash
docker logs wildfire 2>&1 | tail -50
```
