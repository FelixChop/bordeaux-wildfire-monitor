#!/usr/bin/env python3
"""
Flask web app: Bordeaux wildfire risk interactive map.
Auto-fetches real-time data, serves Windy-style map.
"""

import os
import io
import csv
import json
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from threading import Thread
import time

from flask import Flask, render_template, jsonify, Response
import requests

# Imports from local modules
import sys
sys.path.insert(0, os.path.dirname(__file__))
from src.montecarlo import MonteCarloEngine

app = Flask(__name__)

# Configuration
DATA_CACHE_DIR = Path('cache')
DATA_CACHE_DIR.mkdir(exist_ok=True)
UPDATE_INTERVAL = 3600  # Refresh every hour (seconds)

# Global state
latest_data = None
last_update = None

# Multiple NRT products => more satellite passes => fresher & denser coverage.
_FIRMS_PRODUCTS = ['VIIRS_SNPP_NRT', 'VIIRS_NOAA20_NRT', 'VIIRS_NOAA21_NRT', 'MODIS_NRT']


def _firms_confident(raw):
    """VIIRS confidence is a letter (l/n/h); MODIS is 0-100. Keep nominal+high."""
    raw = (raw or '').strip()
    if raw in ('n', 'h'):
        return True
    if raw == 'l':
        return False
    try:
        return float(raw) >= 70
    except ValueError:
        return False


def _fetch_firms_product(map_key, product, bbox, days=5):
    """Fetch one FIRMS product; return list of hotspot dicts."""
    url = (f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{map_key}/"
           f"{product}/{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}/{days}")
    out = []
    try:
        r = requests.get(url, timeout=15)
        if r.status_code != 200 or not r.text.strip():
            return out
        for row in csv.DictReader(io.StringIO(r.text)):
            try:
                if not _firms_confident(row.get('confidence')):
                    continue
                t = (row.get('acq_time') or '0').strip().zfill(4)
                ts = f"{row.get('acq_date', '').strip()}T{t[:2]}:{t[2:]}:00Z"
                out.append({
                    'lat': float(row['latitude']),
                    'lon': float(row['longitude']),
                    'confidence': row.get('confidence', '').strip(),
                    'frp': float(row.get('frp') or 0.0),
                    'timestamp': ts,
                    'sat': product.split('_')[0],
                })
            except (ValueError, KeyError, TypeError):
                pass
    except Exception as e:
        print(f"FIRMS {product} error: {e}")
    return out


def fetch_nasa_firms():
    """Fetch real-time fire hotspots from several NRT satellites and merge them."""
    map_key = os.getenv('NASA_FIRMS_MAP_KEY', 'DEMO_KEY')
    bbox = (-1.5, 44.5, -0.5, 45.2)

    hotspots = []
    for product in _FIRMS_PRODUCTS:
        hotspots.extend(_fetch_firms_product(map_key, product, bbox))

    if hotspots:
        last_ts = max((h['timestamp'] for h in hotspots), default=None)
        sats = sorted({h.get('sat') for h in hotspots})
        return {'hotspots': hotspots, 'source': 'NASA FIRMS',
                'satellites': sats, 'last_detection': last_ts}

    # Fallback mock data
    return {
        'hotspots': [
            {'lat': 44.85, 'lon': -1.205, 'confidence': 95, 'timestamp': None},
            {'lat': 44.86, 'lon': -1.22, 'confidence': 93, 'timestamp': None},
            {'lat': 44.87, 'lon': -1.23, 'confidence': 90, 'timestamp': None},
        ],
        'source': 'MOCK (from real perimeter)', 'satellites': [], 'last_detection': None
    }

def fetch_extended_wind_forecast():
    """Fetch 10-day wind forecast (GFS via Open-Meteo)."""
    try:
        # Open-Meteo provides GFS data (up to 16 days)
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            'latitude': 45.0,
            'longitude': -1.0,
            'hourly': 'wind_speed_10m,wind_direction_10m,relative_humidity_2m,temperature_2m',
            'forecast_days': 10,
            'timezone': 'UTC'
        }

        response = requests.get(url, params=params, timeout=15)
        if response.status_code == 200:
            data = response.json()
            hourly = data.get('hourly', {})
            times = hourly.get('time', [])
            speeds = hourly.get('wind_speed_10m', [])
            directions = hourly.get('wind_direction_10m', [])
            humidity = hourly.get('relative_humidity_2m', [])
            temperature = hourly.get('temperature_2m', [])

            def _at(arr, i, default):
                return float(arr[i]) if i < len(arr) and arr[i] is not None else default

            wind_records = []
            for i, time_str in enumerate(times[:240]):  # 10 days = 240 hours
                if i < len(speeds) and i < len(directions):
                    wind_records.append({
                        'timestamp': time_str,
                        'wind_speed_10m_ms': _at(speeds, i, 0),
                        'wind_direction_10m_deg': _at(directions, i, 270),
                        'relative_humidity_pct': _at(humidity, i, 50),
                        'temperature_c': _at(temperature, i, 25),
                    })

            return {
                'hourly_wind': wind_records,
                'n_records': len(wind_records),
                'source': 'Open-Meteo GFS'
            }
    except Exception as e:
        print(f"Wind fetch error: {e}")

    return {
        'hourly_wind': [
            {'timestamp': '2026-07-26T00:00', 'wind_speed_10m_ms': 10.0, 'wind_direction_10m_deg': 280}
            for _ in range(240)
        ],
        'source': 'MOCK'
    }

# Simulation domain (also used for the vegetation raster).
SIM_BBOX = (-1.5, 44.5, -0.5, 45.2)  # lon_min, lat_min, lon_max, lat_max

# Vegetation fuel map cached across refreshes (numpy, not JSON-serialisable).
_veg_cache = {'date': None, 'fuel': None, 'bbox': SIM_BBOX}


def fetch_wind_field(n=6, hours=192):
    """Grid of Open-Meteo wind+humidity forecasts for a spatial wind field."""
    lon0, lat0, lon1, lat1 = SIM_BBOX
    grid_lats = list(np.linspace(lat0, lat1, n))
    grid_lons = list(np.linspace(lon0, lon1, n))
    lats_q, lons_q = [], []
    for la in grid_lats:
        for lo in grid_lons:
            lats_q.append(round(float(la), 4))
            lons_q.append(round(float(lo), 4))
    try:
        r = requests.get("https://api.open-meteo.com/v1/forecast", params={
            'latitude': ','.join(map(str, lats_q)),
            'longitude': ','.join(map(str, lons_q)),
            'hourly': 'wind_speed_10m,wind_direction_10m,relative_humidity_2m',
            'forecast_days': 8, 'timezone': 'UTC',
        }, timeout=20)
        if r.status_code != 200:
            return None
        results = r.json()
        if isinstance(results, dict):
            results = [results]
        times = results[0]['hourly']['time'][:hours]
        # reshape into n x n grids per hour
        speed = np.zeros((len(times), n, n))
        wdir = np.zeros((len(times), n, n))
        rh = np.zeros((len(times), n, n))
        for idx, res in enumerate(results):
            i, j = idx // n, idx % n
            h = res.get('hourly', {})
            for t in range(len(times)):
                speed[t, i, j] = (h.get('wind_speed_10m') or [0])[t] or 0
                wdir[t, i, j] = (h.get('wind_direction_10m') or [270])[t] or 270
                rh[t, i, j] = (h.get('relative_humidity_2m') or [50])[t] or 50
        return {
            'grid_lats': [round(float(x), 4) for x in grid_lats],
            'grid_lons': [round(float(x), 4) for x in grid_lons],
            'times': times,
            'speed': speed, 'dir': wdir, 'rh': rh,
        }
    except Exception as e:
        print(f"Wind field fetch error: {e}")
        return None


def fetch_vegetation():
    """NDVI (NASA GIBS MODIS) -> fuel map: water/urban low, forest high."""
    lon0, lat0, lon1, lat1 = SIM_BBOX
    day = (datetime.utcnow() - timedelta(days=16)).strftime('%Y-%m-%d')
    if _veg_cache['date'] == day and _veg_cache['fuel'] is not None:
        return _veg_cache['fuel']
    try:
        url = ("https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi"
               "?SERVICE=WMS&REQUEST=GetMap&VERSION=1.3.0"
               "&LAYERS=MODIS_Terra_NDVI_8Day&STYLES=&CRS=EPSG:4326"
               f"&BBOX={lat0},{lon0},{lat1},{lon1}&WIDTH=360&HEIGHT=280"
               f"&FORMAT=image/png&TIME={day}")
        r = requests.get(url, timeout=25)
        if r.status_code != 200:
            return None
        import matplotlib.image as mpimg
        img = mpimg.imread(io.BytesIO(r.content), format='png')  # HxWx4, 0..1
        rgb = (img[:, :, :3] * 255.0)
        R, G, B = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        bright = R + G + B
        veg = (G > R) & (G > 40)
        fuel = np.where(veg, np.clip(G / 150.0, 0.25, 1.0), 0.12)
        fuel = np.where(bright < 30, 0.0, fuel)  # water / nodata
        # sanity: if almost everything is nodata, the tile failed -> uniform fuel
        if (fuel > 0).mean() < 0.15:
            fuel = np.full_like(fuel, 0.6)
        _veg_cache.update(date=day, fuel=fuel)
        print(f"✓ Vegetation loaded ({day}), veg-cover {(fuel>0.25).mean():.0%}")
        return fuel
    except Exception as e:
        print(f"Vegetation fetch error: {e}")
        return None


_wind_field = None


def fetch_air_quality():
    """Air quality at Bordeaux (Open-Meteo) — smoke particles & gases."""
    try:
        r = requests.get("https://air-quality-api.open-meteo.com/v1/air-quality", params={
            'latitude': 44.84, 'longitude': -0.58, 'timezone': 'UTC',
            'current': 'pm2_5,pm10,carbon_monoxide,nitrogen_dioxide,ozone,'
                       'sulphur_dioxide,european_aqi',
        }, timeout=15)
        if r.status_code != 200:
            return None
        c = r.json().get('current', {})
        return {
            'aqi': c.get('european_aqi'),
            'pm2_5': c.get('pm2_5'), 'pm10': c.get('pm10'),
            'co': c.get('carbon_monoxide'), 'no2': c.get('nitrogen_dioxide'),
            'o3': c.get('ozone'), 'so2': c.get('sulphur_dioxide'),
            'time': c.get('time'),
        }
    except Exception as e:
        print(f"Air quality error: {e}")
        return None


def update_data():
    """Background thread: fetch latest data periodically."""
    global latest_data, last_update, _wind_field

    while True:
        try:
            print(f"[{datetime.utcnow().isoformat()}] Fetching real-time data...")

            firms = fetch_nasa_firms()
            wind = fetch_extended_wind_forecast()
            _wind_field = fetch_wind_field()
            fetch_vegetation()  # populates _veg_cache

            # Compute fire perimeter
            if firms['hotspots']:
                hotspots = firms['hotspots']
                lats = [h['lat'] for h in hotspots]
                lons = [h['lon'] for h in hotspots]
                centroid_lat = np.mean(lats)
                centroid_lon = np.mean(lons)

                # Distance to Bordeaux
                bordeaux_lat, bordeaux_lon = 44.837, -0.579
                dist_km = np.sqrt(
                    ((centroid_lat - bordeaux_lat) * 111)**2 +
                    ((centroid_lon - bordeaux_lon) * 111 * np.cos(np.radians(centroid_lat)))**2
                )
            else:
                centroid_lat, centroid_lon = 44.85, -1.205
                dist_km = 49.4

            latest_data = {
                'timestamp': datetime.utcnow().isoformat(),
                'fire_perimeter': {
                    'centroid': {'lat': centroid_lat, 'lon': centroid_lon},
                    'distance_to_bordeaux_km': dist_km,
                    'n_hotspots': len(firms['hotspots'])
                },
                'firms': firms,
                'wind': wind,
                'air': fetch_air_quality(),
            }

            last_update = datetime.utcnow()
            print(f"✓ Data updated. Fire distance: {dist_km:.1f} km")

            try:
                compute_simulation()  # pre-warm the 7-day sim so requests are instant
                print("✓ Simulation pre-computed")
            except Exception as e:
                print(f"sim precompute error: {e}")

        except Exception as e:
            print(f"❌ Data fetch error: {e}")

        time.sleep(UPDATE_INTERVAL)

def generate_map_html():
    """Generate interactive Folium map HTML."""
    if not latest_data:
        return "<p>No data available yet. Please wait...</p>"

    try:
        import folium
    except ImportError:
        return "<p>Folium not installed</p>"

    m = folium.Map(
        location=[44.837, -0.579],
        zoom_start=8,
        tiles='OpenStreetMap'
    )

    # Fire hotspots
    fire_group = folium.FeatureGroup(name='🔥 Fire Hotspots', show=True)
    for spot in latest_data['firms']['hotspots']:
        folium.Circle(
            location=[spot['lat'], spot['lon']],
            radius=300,
            color='#ff0000',
            fill=True,
            fillColor='#ff6600',
            fillOpacity=0.6,
            weight=2
        ).add_to(fire_group)
    fire_group.add_to(m)

    # Wind vectors
    wind_group = folium.FeatureGroup(name='💨 Wind Forecast', show=True)
    lat_range = np.linspace(43.8, 45.8, 8)
    lon_range = np.linspace(-1.8, -0.3, 10)

    wind_records = latest_data['wind']['hourly_wind'][:24]
    if wind_records:
        avg_speed = np.mean([w['wind_speed_10m_ms'] for w in wind_records])
        avg_dir = np.mean([w['wind_direction_10m_deg'] for w in wind_records])

        for lat in lat_range:
            for lon in lon_range:
                direction_rad = np.radians(avg_dir)
                arrow_len = 0.05 + avg_speed * 0.01
                end_lat = lat + arrow_len * np.cos(direction_rad)
                end_lon = lon + arrow_len * np.sin(direction_rad) / np.cos(np.radians(lat))

                color = '#00cc00' if avg_speed < 5 else '#ffcc00' if avg_speed < 10 else '#ff9900' if avg_speed < 15 else '#ff0000'

                folium.PolyLine(
                    locations=[[lat, lon], [end_lat, end_lon]],
                    color=color,
                    weight=2,
                    opacity=0.7
                ).add_to(wind_group)

    wind_group.add_to(m)

    # Risk zones
    risk_group = folium.FeatureGroup(name='⚠️ Risk Zones', show=True)
    fire_lat = latest_data['fire_perimeter']['centroid']['lat']
    fire_lon = latest_data['fire_perimeter']['centroid']['lon']

    for radius_km, label, color in [(10000, 'HIGH', '#cc0000'), (30000, 'MEDIUM', '#ff6600'), (60000, 'LOW', '#ffcc00')]:
        folium.Circle(
            location=[fire_lat, fire_lon],
            radius=radius_km,
            color=color,
            fill=True,
            fillOpacity=0.2,
            weight=1,
            dashArray='5,5'
        ).add_to(risk_group)

    risk_group.add_to(m)

    # Cities
    cities = {
        'Bordeaux': (44.837, -0.579, 'blue'),
        'Saumos': (44.85, -1.205, 'red'),
        'Lacanau': (44.984, -1.272, 'orange'),
    }

    for city, (lat, lon, color) in cities.items():
        folium.Marker(
            location=[lat, lon],
            popup=city,
            icon=folium.Icon(color=color)
        ).add_to(m)

    folium.LayerControl().add_to(m)

    return m._repr_html_()

@app.route('/')
def index():
    """Main page: responsive Leaflet map + timeline (served from template)."""
    return render_template('index.html')


def _hour_bucket(ts):
    """'2026-07-26T14:37:00Z' -> '2026-07-26T14:00Z' (hour bucket key)."""
    try:
        return ts[:13] + ':00Z'
    except (TypeError, IndexError):
        return None


@app.route('/api/fire-history')
def api_fire_history():
    """Real FIRMS hotspots grouped into cumulative hourly frames (past fire)."""
    if not latest_data:
        return jsonify({'error': 'No data available'}), 503
    hotspots = latest_data['firms'].get('hotspots', [])
    # bucket by hour
    buckets = {}
    for h in hotspots:
        key = _hour_bucket(h.get('timestamp'))
        if key:
            buckets.setdefault(key, []).append({
                'lat': h['lat'], 'lon': h['lon'], 'frp': h.get('frp', 0.0),
            })
    hours = sorted(buckets.keys())
    frames = []
    cumulative = []
    for hkey in hours:
        cumulative = cumulative + buckets[hkey]
        frames.append({
            'timestamp': hkey,
            'new': len(buckets[hkey]),
            'total': len(cumulative),
            'points': list(cumulative),
        })
    return jsonify({
        'source': latest_data['firms'].get('source'),
        'n_frames': len(frames),
        'frames': frames,
    })


_sim_cache = {'key': None, 'data': None}


def compute_simulation():
    """Build (and cache) the 7-day propagation from the current data."""
    if not latest_data:
        return None
    from src.fire_front import simulate_fire_front
    hotspots = latest_data['firms'].get('hotspots', [])
    wind = latest_data['wind'].get('hourly_wind', [])
    now_key = datetime.utcnow().strftime('%Y-%m-%dT%H:00')
    future = [w for w in wind if (w.get('timestamp') or '') >= now_key]
    wind = future if future else wind

    wf = None
    if _wind_field and _wind_field.get('times'):
        idx = [k for k, t in enumerate(_wind_field['times']) if (t or '') >= now_key]
        if idx:
            wf = {
                'grid_lats': _wind_field['grid_lats'],
                'grid_lons': _wind_field['grid_lons'],
                'times': [_wind_field['times'][k] for k in idx],
                'speed': _wind_field['speed'][idx],
                'dir': _wind_field['dir'][idx],
                'rh': _wind_field['rh'][idx],
            }
        else:
            wf = _wind_field
    fuel = _veg_cache.get('fuel')

    key = f"{last_update}:{len(hotspots)}:{now_key}:{wf is not None}:{fuel is not None}:168"
    if _sim_cache['key'] != key:
        _sim_cache['data'] = simulate_fire_front(
            hotspots, wind, wind_field=wf, veg_fuel=fuel, veg_bbox=SIM_BBOX,
            max_hours=168, emit_every=3)  # 7 days, one frame every 3 h
        _sim_cache['key'] = key
    return _sim_cache['data']


@app.route('/api/simulation')
def api_simulation():
    """Hour-by-hour multi-source fire propagation from every real hotspot."""
    if not latest_data:
        return jsonify({'error': 'No data available'}), 503
    data = compute_simulation()
    return jsonify(data if data is not None else {'error': 'No data'}), (200 if data else 503)


@app.route('/api/vegetation.png')
def api_vegetation_png():
    """Fuel map as a translucent PNG overlay (green=forest, blue=water)."""
    fuel = _veg_cache.get('fuel')
    if fuel is None:
        return jsonify({'error': 'No vegetation'}), 503
    import matplotlib.image as mpimg
    h, w = fuel.shape
    rgba = np.zeros((h, w, 4))
    veg = fuel > 0.25
    urban = (fuel > 0) & (fuel <= 0.25)
    water = fuel <= 0
    rgba[veg] = [0.20, 0.62, 0.17, 0.0]
    rgba[veg, 3] = np.clip(fuel[veg] * 0.7, 0.18, 0.7)
    rgba[urban] = [0.60, 0.58, 0.52, 0.28]
    rgba[water] = [0.10, 0.42, 0.62, 0.30]
    buf = io.BytesIO()
    mpimg.imsave(buf, rgba, format='png')
    buf.seek(0)
    return Response(buf.read(), mimetype='image/png',
                    headers={'Cache-Control': 'public, max-age=3600'})


@app.route('/api/windfield')
def api_windfield():
    """Spatial wind field (grid of arrows) for the current & forecast hours."""
    if not _wind_field:
        return jsonify({'error': 'No wind field'}), 503
    now_key = datetime.utcnow().strftime('%Y-%m-%dT%H:00')
    times = _wind_field['times']
    idx = [k for k, t in enumerate(times) if (t or '') >= now_key] or list(range(len(times)))
    return jsonify({
        'grid_lats': _wind_field['grid_lats'],
        'grid_lons': _wind_field['grid_lons'],
        'times': [times[k] for k in idx],
        'speed': np.asarray(_wind_field['speed'])[idx].round(1).tolist(),
        'dir': np.asarray(_wind_field['dir'])[idx].round(0).tolist(),
    })


def _unused_legacy_index():
    fire_perim = latest_data['fire_perimeter']
    map_html = generate_map_html()

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>🔥 Bordeaux Wildfire Risk Monitor</title>
        <style>
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            body {{ font-family: Arial, sans-serif; background: #1a1a1a; color: #fff; }}
            .container {{ display: flex; height: 100vh; }}
            .map {{ flex: 1; }}
            .panel {{
                width: 350px;
                background: #222;
                padding: 20px;
                overflow-y: auto;
                border-left: 2px solid #cc0000;
            }}
            .panel h2 {{ color: #ff6600; margin-bottom: 15px; font-size: 18px; }}
            .stat {{ margin: 12px 0; padding: 8px; background: #333; border-left: 3px solid #ff9900; }}
            .stat-label {{ font-size: 12px; color: #aaa; }}
            .stat-value {{ font-size: 16px; font-weight: bold; color: #fff; }}
            .warning {{ background: #442200; border-left-color: #ff6600; }}
            .safe {{ background: #224400; border-left-color: #00cc00; }}
            .time {{ font-size: 11px; color: #666; margin-top: 20px; padding-top: 10px; border-top: 1px solid #444; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="map">
                {map_html}
            </div>
            <div class="panel">
                <h2>📊 CURRENT STATUS</h2>

                <div class="stat warning">
                    <div class="stat-label">🔥 Fire Distance</div>
                    <div class="stat-value">{fire_perim['distance_to_bordeaux_km']:.1f} km</div>
                </div>

                <div class="stat">
                    <div class="stat-label">🔴 Active Hotspots</div>
                    <div class="stat-value">{fire_perim['n_hotspots']}</div>
                </div>

                <div class="stat">
                    <div class="stat-label">💨 Avg Wind (24h)</div>
                    <div class="stat-value">10.9 m/s from W</div>
                </div>

                <div class="stat safe">
                    <div class="stat-label">⚠️ Bordeaux Risk</div>
                    <div class="stat-value">&lt; 1%</div>
                </div>

                <h2 style="margin-top: 25px;">🎯 FORECAST (10 DAYS)</h2>

                <div class="stat">
                    <div class="stat-label">🌪️ Wind Trend</div>
                    <div class="stat-value">Stable W (coastal)</div>
                </div>

                <div class="stat safe">
                    <div class="stat-label">✅ Trend</div>
                    <div class="stat-value">Fire away from city</div>
                </div>

                <h2 style="margin-top: 25px;">📍 REAL IMPACT</h2>

                <div class="stat warning">
                    <div class="stat-label">🏠 Le Porge/Lacanau</div>
                    <div class="stat-value">~250 buildings burned</div>
                </div>

                <div class="stat warning">
                    <div class="stat-label">📈 Perimeter</div>
                    <div class="stat-value">4,800 hectares</div>
                </div>

                <div class="stat">
                    <div class="stat-label">🚒 Suppression</div>
                    <div class="stat-value">2,500 firefighters</div>
                </div>

                <div class="time">
                    Last updated: {last_update.strftime('%Y-%m-%d %H:%M UTC') if last_update else 'Never'}
                    <br>
                    Data sources: NASA FIRMS, Météo-France ARPEGE, Gironde Préfecture
                </div>
            </div>
        </div>
    </body>
    </html>
    """

    return html

@app.route('/api/data')
def api_data():
    """JSON API endpoint for real-time data."""
    if latest_data:
        return jsonify(latest_data)
    return jsonify({'error': 'No data available'}), 503

# ---------------------------------------------------------------------------
# Ground / response layers: fire stations, water bombers, road traffic.
# ---------------------------------------------------------------------------
_layer_cache = {}


def _cached(key, ttl, producer):
    """Tiny TTL cache; on producer failure, serve the last good value if any."""
    now = time.time()
    ent = _layer_cache.get(key)
    if ent and now - ent[0] < ttl:
        return ent[1]
    try:
        val = producer()
    except Exception as e:
        print(f"layer {key} error: {e}")
        val = None
    if val is not None:
        _layer_cache[key] = (now, val)
        return val
    return ent[1] if ent else None


@app.route('/api/stations')
def api_stations():
    """Fire stations (fixed) from OpenStreetMap — response means nearby."""
    def prod():
        q = ('[out:json][timeout:25];('
             'node["amenity"="fire_station"](44.4,-1.6,45.4,-0.2);'
             'way["amenity"="fire_station"](44.4,-1.6,45.4,-0.2);'
             ');out center;')
        r = requests.get("https://overpass-api.de/api/interpreter",
                         params={'data': q},
                         headers={'User-Agent': 'wildfire-app/1.0 (felixrevert@gmail.com)'},
                         timeout=35)
        if r.status_code != 200:
            return None
        out = []
        for e in r.json().get('elements', []):
            lat = e.get('lat') or (e.get('center') or {}).get('lat')
            lon = e.get('lon') or (e.get('center') or {}).get('lon')
            if lat and lon:
                out.append({'lat': lat, 'lon': lon,
                            'name': e.get('tags', {}).get('name', 'Caserne')})
        return out
    return jsonify({'stations': _cached('stations', 86400, prod) or []})


@app.route('/api/traffic')
def api_traffic():
    """Live road traffic (Bordeaux Métropole) — congestion & blockages."""
    def prod():
        r = requests.get(
            "https://opendata.bordeaux-metropole.fr/api/records/1.0/search/",
            params={'dataset': 'ci_trafi_l', 'rows': 1000}, timeout=20)
        if r.status_code != 200:
            return None
        out, updated = [], None
        for rec in r.json().get('records', []):
            f = rec.get('fields', {})
            etat = f.get('etat')
            if etat in (None, 'INCONNU'):
                continue
            gs = f.get('geo_shape', {})
            if gs.get('type') != 'LineString':
                continue
            out.append({'etat': etat,
                        'coords': [[c[1], c[0]] for c in gs['coordinates']],
                        'voie': f.get('ident')})
            updated = f.get('mdate') or updated
        return {'segments': out, 'updated': updated}
    data = _cached('traffic', 120, prod) or {'segments': [], 'updated': None}
    return jsonify(data)


@app.route('/api/aircraft')
def api_aircraft():
    """Low-flying aircraft (OpenSky ADS-B) — water bombers when a fire is active."""
    def prod():
        r = requests.get("https://opensky-network.org/api/states/all",
                         params={'lamin': 44.2, 'lomin': -1.7,
                                 'lamax': 45.5, 'lomax': -0.2}, timeout=15)
        if r.status_code != 200:
            return None
        fire_kw = ('PELICAN', 'PELIC', 'MILAN', 'FIRE', 'DRAGON', 'CANADAIR', 'BOMB')
        out = []
        for s in r.json().get('states') or []:
            cs = (s[1] or '').strip()
            lon, lat, baro, vel, trk, geo = s[5], s[6], s[7], s[9], s[10], s[13]
            if lat is None or lon is None:
                continue
            alt = baro if baro is not None else geo
            is_fire = any(k in cs.upper() for k in fire_kw)
            # exclude airliners around Bordeaux-Mérignac airport (44.83, -0.70)
            near_airport = ((lat - 44.83) ** 2 + (lon + 0.70) ** 2) ** 0.5 < 0.14
            low_working = (alt is not None and alt < 1800 and not near_airport)
            if is_fire or low_working:
                out.append({'callsign': cs or '—', 'lat': lat, 'lon': lon,
                            'alt': round(alt) if alt is not None else None,
                            'heading': trk, 'fire': is_fire})
        return out
    return jsonify({'aircraft': _cached('aircraft', 45, prod) or []})


# Start background data fetch thread at import time so it also runs under
# gunicorn (which never executes the __main__ block). update_data() does its
# first fetch immediately then loops with an hourly sleep.
data_thread = Thread(target=update_data, daemon=True)
data_thread.start()

if __name__ == '__main__':
    # Run Flask (change host/port as needed for VPS)
    app.run(
        host='0.0.0.0',  # Listen on all interfaces
        port=int(os.getenv('PORT', 5000)),
        debug=False
    )
