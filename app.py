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

from flask import Flask, render_template, jsonify, Response, request, redirect
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

# Warm-cache on disk: served instantly after a restart while the background
# thread refreshes — visitors never wait for a recompute.
WARM_DIR = Path(os.getenv('WARM_DIR', '/data/warm'))


def _warm_save(name, obj):
    try:
        WARM_DIR.mkdir(parents=True, exist_ok=True)
        tmp = WARM_DIR / (name + '.tmp')
        with open(tmp, 'w') as f:
            json.dump(obj, f)
        tmp.rename(WARM_DIR / name)
    except OSError:
        pass


def _warm_load(name):
    try:
        with open(WARM_DIR / name) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None

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


# Debut officiel de l'evenement : l'incendie bordelais s'est declare le
# mercredi 22 juillet. La timeline reste ANCREE a cette date (les detections
# sont archivees sur disque pour survivre a la fenetre glissante des APIs).
FIRE_T0 = datetime(2026, 7, 22)


def _firms_days():
    """Fenetre FIRMS : l'API NRT accepte AU MAX 5 jours par requete.
    L'historique anterieur vient des archives disque + backfill par date."""
    return 5


_backfilled = set()


def _backfill_archive(name, bbox):
    """Recupere une fois les detections depuis FIRE_T0 (tranches de 5 j
    avec date de depart) pour ancrer l'historique au debut de l'incendie."""
    if name in _backfilled:
        return []
    _backfilled.add(name)
    arch = _warm_load(name) or []
    t0s = FIRE_T0.strftime('%Y-%m-%d')
    have_start = min((h.get('timestamp') or '9') for h in arch) if arch else '9'
    if have_start <= t0s + 'T23':
        return []          # l'archive couvre deja le debut
    map_key = os.getenv('NASA_FIRMS_MAP_KEY', 'DEMO_KEY')
    got = []
    t = FIRE_T0
    while t < datetime.utcnow() - timedelta(days=4):
        ds = t.strftime('%Y-%m-%d')
        for product in _FIRMS_PRODUCTS:
            try:
                got.extend(_fetch_firms_product(map_key, product, bbox,
                                                days=5, date=ds))
            except Exception as e:
                print(f"backfill {product} {ds} err: {e}")
        t += timedelta(days=5)
    if got:
        print(f"✓ Backfill {name} : {len(got)} detections depuis {t0s}")
    return got


def _merge_archive(name, hotspots):
    """Fusionne les detections avec l'archive disque (dedupe, >= FIRE_T0)."""
    arch = _warm_load(name) or []
    seen = {(round(h['lat'], 4), round(h['lon'], 4), h.get('timestamp'), h.get('sat'))
            for h in arch}
    added = 0
    for h in hotspots:
        k = (round(h['lat'], 4), round(h['lon'], 4), h.get('timestamp'), h.get('sat'))
        if k not in seen:
            arch.append(h)
            seen.add(k)
            added += 1
    t0s = FIRE_T0.strftime('%Y-%m-%d')
    arch = [h for h in arch if (h.get('timestamp') or '') >= t0s]
    if added:
        _warm_save(name, arch)
    return arch


def _fetch_firms_product(map_key, product, bbox, days=None, date=None):
    """Fetch one FIRMS product; return list of hotspot dicts.
    date='YYYY-MM-DD' (optionnel) = jour de DEPART de la fenetre (backfill)."""
    if days is None:
        days = _firms_days()
    url = (f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{map_key}/"
           f"{product}/{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}/{days}"
           + (f"/{date}" if date else ''))
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
    bbox = (-1.6, 44.2, -0.4, 45.3)   # inclut Biscarrosse au sud

    hotspots = []
    for product in _FIRMS_PRODUCTS:
        hotspots.extend(_fetch_firms_product(map_key, product, bbox))
    hotspots = _merge_archive('fire_archive_gironde.json',
                              hotspots + _backfill_archive(
                                  'fire_archive_gironde.json', bbox))

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
SIM_BBOX = (-1.8, 44.2, -0.2, 45.6)  # lon_min, lat_min, lon_max, lat_max (large : couvre écrans hauts)

# Vegetation fuel map cached across refreshes (numpy, not JSON-serialisable).
_veg_cache = {'date': None, 'fuel': None, 'bbox': SIM_BBOX}


def fetch_wind_field(n=6, hours=432):  # 10 j passes + 8 j de prevision
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
        global _wx_block_until
        if time.time() < _wx_block_until:
            return None
        r = requests.get("https://api.open-meteo.com/v1/forecast", params={
            'latitude': ','.join(map(str, lats_q)),
            'longitude': ','.join(map(str, lons_q)),
            'hourly': 'wind_speed_10m,wind_direction_10m,relative_humidity_2m,'
                      'temperature_2m,soil_moisture_0_to_7cm,'
                      'weather_code,precipitation',
            'past_days': 10, 'forecast_days': 8, 'timezone': 'UTC',
        }, timeout=20)
        if r.status_code == 429:
            _wx_block_until = time.time() + 1800
            print("vent 429 -> pause 30 min")
            return None
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
        temp = np.zeros((len(times), n, n))
        soil = np.zeros((len(times), n, n))
        wmo = np.zeros((len(times), n, n))
        rain = np.zeros((len(times), n, n))
        for idx, res in enumerate(results):
            i, j = idx // n, idx % n
            h = res.get('hourly', {})

            def _g(key, t, d):
                arr = h.get(key) or []
                v = arr[t] if t < len(arr) else None
                return d if v is None else v
            for t in range(len(times)):
                speed[t, i, j] = _g('wind_speed_10m', t, 0)
                wdir[t, i, j] = _g('wind_direction_10m', t, 270)
                rh[t, i, j] = _g('relative_humidity_2m', t, 50)
                temp[t, i, j] = _g('temperature_2m', t, 25)
                soil[t, i, j] = _g('soil_moisture_0_to_7cm', t, 0.2)
                wmo[t, i, j] = _g('weather_code', t, 0)
                rain[t, i, j] = _g('precipitation', t, 0)
        return {
            'grid_lats': [round(float(x), 4) for x in grid_lats],
            'grid_lons': [round(float(x), 4) for x in grid_lons],
            'times': times,
            'speed': speed, 'dir': wdir, 'rh': rh, 'temp': temp, 'soil': soil,
            'wmo': wmo, 'rain': rain,
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
    # keep serving the previous fuel map if today's fetch fails (GIBS 504s)
    stale = _veg_cache.get('fuel')
    try:
        r = None
        for back in (16, 24, 32):  # try older 8-day composites if GIBS chokes
            d2 = (datetime.utcnow() - timedelta(days=back)).strftime('%Y-%m-%d')
            url = ("https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi"
                   "?SERVICE=WMS&REQUEST=GetMap&VERSION=1.3.0"
                   "&LAYERS=MODIS_Terra_NDVI_8Day&STYLES=&CRS=EPSG:4326"
                   f"&BBOX={lat0},{lon0},{lat1},{lon1}"
                   f"&WIDTH={min(1400, max(420, int((lon1 - lon0) * 90)))}"
                   f"&HEIGHT={min(1400, max(370, int((lat1 - lat0) * 90)))}"
                   f"&FORMAT=image/png&TIME={d2}")
            r = requests.get(url, timeout=40)
            if r.status_code == 200:
                break
        if r is None or r.status_code != 200:
            return stale
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
        return stale


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


# --------------------------------------------------------------------------
# Mode FRANCE : détection nationale + clustering des incendies actifs
# --------------------------------------------------------------------------
FR_BBOX = (-5.2, 41.2, 9.8, 51.3)
_fr_fires = {'ts': None, 'clusters': []}
_geo_names = {}   # cache reverse-geocoding


def _cluster_name(lat, lon):
    key = (round(lat, 1), round(lon, 1))
    if key in _geo_names:
        return _geo_names[key]
    name = ''
    try:
        r = requests.get('https://api-adresse.data.gouv.fr/reverse/',
                         params={'lat': lat, 'lon': lon}, timeout=8)
        feats = r.json().get('features', [])
        if feats:
            p = feats[0].get('properties', {})
            name = p.get('city') or p.get('municipality') or ''
            ctx = (p.get('context') or '').split(',')
            if len(ctx) > 1:
                name = f"{name} ({ctx[1].strip()})" if name else ctx[1].strip()
    except Exception:
        pass
    if not name:
        # centroïde en pleine forêt : Nominatim (zoom commune) trouve toujours
        try:
            time.sleep(1.1)   # politique d'usage Nominatim
            r = requests.get('https://nominatim.openstreetmap.org/reverse',
                             params={'lat': lat, 'lon': lon, 'format': 'jsonv2',
                                     'zoom': 10, 'accept-language': 'fr'},
                             headers={'User-Agent': 'feux-de-foret.fr (felixrevert@gmail.com)'},
                             timeout=10)
            a = r.json().get('address', {})
            if a.get('country_code') and a['country_code'] != 'fr':
                _geo_names[key] = None      # hors de France
                return None
            name = (a.get('village') or a.get('town') or a.get('city')
                    or a.get('municipality') or '')
            dep = a.get('county') or a.get('state_district') or ''
            if dep:
                name = f"{name} ({dep})" if name else dep
        except Exception:
            pass
    _geo_names[key] = name
    return name


def fetch_france_fires():
    """France-wide hotspots (3 days) grouped into fire clusters."""
    map_key = os.getenv('NASA_FIRMS_MAP_KEY', 'DEMO_KEY')
    hotspots = []
    for product in _FIRMS_PRODUCTS:
        hotspots.extend(_fetch_firms_product(map_key, product, FR_BBOX))
    hotspots = _merge_archive('fire_archive_france.json',
                              hotspots + _backfill_archive(
                                  'fire_archive_france.json', FR_BBOX))
    # clustering par buckets 0.22° (~20 km) fusionnés en 8-connexité
    from collections import defaultdict
    cell = 0.22
    buckets = defaultdict(list)
    for h in hotspots:
        buckets[(int(h['lat'] / cell), int(h['lon'] / cell))].append(h)
    seen, clusters = set(), []
    for key in list(buckets):
        if key in seen:
            continue
        stack, members = [key], []
        seen.add(key)
        while stack:
            k = stack.pop()
            members.extend(buckets[k])
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    nk = (k[0] + di, k[1] + dj)
                    if nk in buckets and nk not in seen:
                        seen.add(nk)
                        stack.append(nk)
        clusters.append(members)
    now = datetime.utcnow()
    out = []
    for m in clusters:
        if len(m) < 3:
            continue
        frp24 = n24 = 0
        last = ''
        for h in m:
            dt = _parse_ts(h.get('timestamp'))
            if dt and (now - dt) <= timedelta(hours=24):
                frp24 += h.get('frp', 0)
                n24 += 1
            if h.get('timestamp') and h['timestamp'] > last:
                last = h['timestamp']
        first = min((h['timestamp'] for h in m if h.get('timestamp')),
                    default=None)
        w = [max(h.get('frp', 1), 1) for h in m]
        sw = sum(w)
        lat = sum(h['lat'] * wi for h, wi in zip(m, w)) / sw
        lon = sum(h['lon'] * wi for h, wi in zip(m, w)) / sw
        out.append({'zone': f'z{round(lat, 2)}_{round(lon, 2)}',
                    'lat': round(lat, 3), 'lon': round(lon, 3),
                    'n_total': len(m), 'n_24h': n24,
                    'frp_24h': round(frp24), 'last_detection': last,
                    'first_detection': first,
                    'active': n24 > 0})
    out.sort(key=lambda c: -c['frp_24h'])
    kept = []
    for i, c in enumerate(out):
        if i < 30:
            nm = _cluster_name(c['lat'], c['lon'])
            if nm is None:
                continue                     # cluster hors de France
            c['name'] = nm
        kept.append(c)
    for c in kept:
        if c.get('active'):
            c['response'] = _fire_response(c)
            sc_z = (ZONES.get(c['zone']) or {}).get('sim_score')
            if sc_z:
                c['sim_score'] = sc_z
    return {'ts': now.strftime('%Y-%m-%dT%H:00Z'), 'clusters': kept,
            'hotspots': hotspots}


@app.route('/api/france-air.png')
def api_france_air_png():
    """Champ qualité de l'air France entière (EAQI, lissé façon Windy)."""
    def prod():
        n_lat, n_lon = 11, 10
        lat0, lat1, lon0, lon1 = 41.2, 51.3, -5.2, 9.8
        lats = np.linspace(lat0, lat1, n_lat)
        lons = np.linspace(lon0, lon1, n_lon)
        laq, loq = [], []
        for la in lats:
            for lo in lons:
                laq.append(round(float(la), 2))
                loq.append(round(float(lo), 2))
        r = requests.get('https://air-quality-api.open-meteo.com/v1/air-quality', params={
            'latitude': ','.join(map(str, laq)), 'longitude': ','.join(map(str, loq)),
            'current': 'european_aqi', 'timezone': 'UTC'}, timeout=25)
        if r.status_code != 200:
            return None
        res = r.json()
        if isinstance(res, dict):
            res = [res]
        grid = np.zeros((n_lat, n_lon))
        for idx, x in enumerate(res[:n_lat * n_lon]):
            v = x.get('current', {}).get('european_aqi')
            grid[idx // n_lon, idx % n_lon] = v if v is not None else 0
        from scipy.interpolate import RegularGridInterpolator
        from scipy.ndimage import gaussian_filter
        f = RegularGridInterpolator((lats, lons), grid, method='cubic')
        fy = np.linspace(lat0, lat1, 300)
        fx = np.linspace(lon0, lon1, 300)
        YY, XX = np.meshgrid(fy, fx, indexing='ij')
        fine = gaussian_filter(f(np.column_stack([YY.ravel(), XX.ravel()])).reshape(300, 300), 2.0)
        import matplotlib.colors as mcolors
        cmap = mcolors.LinearSegmentedColormap.from_list('aqi', [
            (0.00, '#2e7d32'), (0.18, '#8bc34a'), (0.34, '#cddc39'),
            (0.50, '#ffb300'), (0.66, '#fb8c00'), (0.82, '#e53935'), (1.00, '#8e24aa')])
        norm = np.clip(fine / 120.0, 0, 1)
        rgba = cmap(norm)
        rgba[..., 3] = np.clip(0.15 + norm * 0.55, 0.15, 0.65)
        import matplotlib.image as mpimg
        buf = io.BytesIO()
        mpimg.imsave(buf, np.flipud(rgba), format='png')
        buf.seek(0)
        return buf.read()
    png = _cached('franceair', 1800, prod)
    if png is None:
        return jsonify({'error': 'No data'}), 503
    return Response(png, mimetype='image/png',
                    headers={'Cache-Control': 'public, max-age=900'})


@app.route('/france')
def france():
    return redirect('/?zone=france', code=302)


@app.route('/zone')
def zone_page():
    return _html('index.html')


def _met_series(wf, lat, lon):
    """Serie horaire vent/direction/temperature INTERPOLEE a un point,
    calculee serveur : les deux sites lisent exactement les memes valeurs."""
    try:
        gl = np.asarray(wf['grid_lats']); gn = np.asarray(wf['grid_lons'])
        gy = float(np.clip((lat - gl[0]) / (gl[-1] - gl[0]) * (len(gl) - 1),
                           0, len(gl) - 1.001))
        gx = float(np.clip((lon - gn[0]) / (gn[-1] - gn[0]) * (len(gn) - 1),
                           0, len(gn) - 1.001))
        i, j = int(gy), int(gx)
        ty, tx = gy - i, gx - j

        def bl(M):
            M = np.asarray(M)
            return (M[:, i, j] * (1-ty) * (1-tx) + M[:, i+1, j] * ty * (1-tx)
                    + M[:, i, j+1] * (1-ty) * tx + M[:, i+1, j+1] * ty * tx)
        sp = bl(wf['speed']); tp = bl(wf['temp'])
        W = np.asarray(wf['wmo']) if wf.get('wmo') is not None else None
        R = np.asarray(wf['rain']) if wf.get('rain') is not None else None
        i_n = i + (1 if ty > 0.5 else 0)
        j_n = j + (1 if tx > 0.5 else 0)
        d2r = np.pi / 180.0
        D = np.asarray(wf['dir']) * d2r
        ue = (np.sin(D[:, i, j]) * (1-ty) * (1-tx)
              + np.sin(D[:, i+1, j]) * ty * (1-tx)
              + np.sin(D[:, i, j+1]) * (1-ty) * tx
              + np.sin(D[:, i+1, j+1]) * ty * tx)
        vn = (np.cos(D[:, i, j]) * (1-ty) * (1-tx)
              + np.cos(D[:, i+1, j]) * ty * (1-tx)
              + np.cos(D[:, i, j+1]) * (1-ty) * tx
              + np.cos(D[:, i+1, j+1]) * ty * tx)
        dr = (np.degrees(np.arctan2(ue, vn)) + 360) % 360
        out = {'times': wf['times'],
               'speed': np.round(sp, 1).tolist(),
               'dir': np.round(dr, 0).tolist(),
               'temp': np.round(tp, 1).tolist()}
        if W is not None:
            out['wmo'] = W[:, i_n, j_n].astype(int).tolist()
        if R is not None:
            out['rain'] = np.round(bl(R), 1).tolist()
        return out
    except Exception:
        return None


def _attach_scores(clusters):
    fr_wf = (ZONES.get('france') or {}).get('wind_field')
    for c in clusters:
        z_c = ZONES.get(c.get('zone')) or {}
        sc_z = z_c.get('sim_score')
        if sc_z:
            c['sim_score'] = sc_z
        if not c.get('active'):
            continue
        # meteo du feu : le girondin lit le champ de la fiche officielle,
        # les autres leur champ local de zone, sinon le champ national
        if 44.0 <= c['lat'] <= 45.4 and -1.6 <= c['lon'] <= -0.2 \
                and _wind_field is not None:
            wf_c = _wind_field
        else:
            wf_c = z_c.get('wind_field') or fr_wf
        if wf_c is not None:
            ms = _met_series(wf_c, c['lat'], c['lon'])
            if ms:
                c['met'] = ms
    return clusters


@app.route('/api/jslog', methods=['POST'])
def api_jslog():
    """Les erreurs JS des clients remontent ici (diagnostic mobile)."""
    try:
        msg = (request.get_data(as_text=True) or '')[:600]
        ua = (request.headers.get('User-Agent') or '')[:80]
        print(f"⚠ JSERR [{ua}] {msg}")
    except Exception:
        pass
    return '', 204


@app.route('/api/fires')
def api_fires():
    """Incendies actifs détectés sur toute la France (clusters)."""
    data = _cached('france_fires', 3600, fetch_france_fires)
    if data is None:
        return jsonify({'clusters': []}), 503
    return jsonify({'ts': data['ts'],
                    'clusters': _attach_scores(list(data['clusters']))})


# ---------------------------------------------------------------------------
# ZONES dynamiques (Phase 2 France) : chaque incendie cliqué devient une zone
# avec ses données locales (FIRMS, vent, végétation, air, communes, ensemble).
# ---------------------------------------------------------------------------
ZONES = {}


def _zone_bboxes(lat, lon, span=1.0):
    """span<1 : petit feu -> petit domaine (simulation en secondes)."""
    dx, dy = 0.8 * span, 0.7 * span
    sim = (round(lon - dx, 2), round(lat - dy, 2),
           round(lon + dx, 2), round(lat + dy, 2))
    view = [[round(lat - 0.37, 2), round(lon - 0.45, 2)],
            [round(lat + 0.36, 2), round(lon + 0.43, 2)]]
    return sim, view


_veg_zone_cache = {}


def fetch_vegetation_bbox(bbox):
    """NDVI fuel map pour une bbox arbitraire (cache par bbox+date)."""
    lon0, lat0, lon1, lat1 = bbox
    day = (datetime.utcnow() - timedelta(days=16)).strftime('%Y-%m-%d')
    key = (bbox, day)
    if key in _veg_zone_cache:
        return _veg_zone_cache[key]
    try:
        r = None
        for back in (16, 24, 32):
            d2 = (datetime.utcnow() - timedelta(days=back)).strftime('%Y-%m-%d')
            url = ("https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi"
                   "?SERVICE=WMS&REQUEST=GetMap&VERSION=1.3.0"
                   "&LAYERS=MODIS_Terra_NDVI_8Day&STYLES=&CRS=EPSG:4326"
                   f"&BBOX={lat0},{lon0},{lat1},{lon1}"
                   f"&WIDTH={min(1400, max(420, int((lon1 - lon0) * 90)))}"
                   f"&HEIGHT={min(1400, max(370, int((lat1 - lat0) * 90)))}"
                   f"&FORMAT=image/png&TIME={d2}")
            r = requests.get(url, timeout=40)
            if r.status_code == 200:
                break
        if r is None or r.status_code != 200:
            return None
        import matplotlib.image as mpimg
        img = mpimg.imread(io.BytesIO(r.content), format='png')
        rgb = (img[:, :, :3] * 255.0)
        R, G, B = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        veg = (G > R) & (G > 40)
        fuel = np.where(veg, np.clip(G / 150.0, 0.25, 1.0), 0.12)
        fuel = np.where((R + G + B) < 30, 0.0, fuel)
        if (fuel > 0).mean() < 0.15:
            fuel = np.full_like(fuel, 0.6)
        if len(_veg_zone_cache) > 12:
            _veg_zone_cache.clear()
        _veg_zone_cache[key] = fuel
        return fuel
    except Exception as e:
        print(f"veg zone error: {e}")
        return None


_air_block_until = 0.0

# Series nationales APPEND-ONLY : le passe est telecharge UNE fois puis
# conserve sur disque ; chaque cycle ne rafraichit que l'heure nouvelle et
# la prevision. L'historique reste ancre a FIRE_T0 meme quand la fenetre
# des APIs (10 j) aura glisse au-dela.
_WX_KEYS = ('speed', 'dir', 'rh', 'temp', 'soil', 'wmo', 'rain')
_AIR_KEYS = ('aqi', 'pm25', 'pm10', 'co', 'no2', 'o3')


def _merge_series(store, new, keys):
    """Fusionne une serie temporelle {times, k: [par heure]} dans le store :
    les heures connues sont mises a jour (prevision -> reanalyse), les
    nouvelles ajoutees ; tri chronologique ; coupe avant FIRE_T0."""
    if not new:
        return store
    if not store or store.get('grid_lats') != new.get('grid_lats')             or store.get('grid_lons') != new.get('grid_lons')             or store.get('bbox') != new.get('bbox'):
        return new                       # grille changee -> reset propre
    idx = {t: i for i, t in enumerate(store['times'])}
    for j, t in enumerate(new['times']):
        i = idx.get(t)
        if i is None:
            store['times'].append(t)
            for k in keys:
                if store.get(k) is not None and new.get(k) is not None:
                    store[k].append(new[k][j])
        else:
            for k in keys:
                if store.get(k) is not None and new.get(k) is not None:
                    store[k][i] = new[k][j]
    order = sorted(range(len(store['times'])), key=lambda q: store['times'][q])
    t0s = FIRE_T0.strftime('%Y-%m-%dT%H:%M')
    keep = [q for q in order if store['times'][q] >= t0s]
    store['times'] = [store['times'][q] for q in keep]
    for k in keys:
        if store.get(k) is not None:
            store[k] = [store[k][q] for q in keep]
    return store


def fetch_air_sensors_fallback():
    """Qualite de l'air OBSERVEE via les capteurs sensor.community (reseau
    ouvert, sans cle) — secours quand l'API CAMS est rate-limitee.
    Produit une frame 'maintenant' sur la grille France."""
    try:
        r = requests.get('https://data.sensor.community/airrohr/v1/filter/'
                         'box=41.2,-5.2,51.3,9.8', timeout=60)
        if r.status_code != 200:
            return None
        n_lat, n_lon = 10, 14
        lon0, lat0, lon1, lat1 = FR_SIM_BBOX
        s = np.zeros((n_lat, n_lon))
        c = np.zeros((n_lat, n_lon))
        for e in r.json():
            try:
                la = float(e['location']['latitude'])
                lo = float(e['location']['longitude'])
                for v in e.get('sensordatavalues', []):
                    if v.get('value_type') == 'P2':
                        pm = float(v['value'])
                        if 0 <= pm < 500:
                            i = int((la - lat0) / (lat1 - lat0) * n_lat)
                            j = int((lo - lon0) / (lon1 - lon0) * n_lon)
                            if 0 <= i < n_lat and 0 <= j < n_lon:
                                s[i, j] += pm
                                c[i, j] += 1
            except (KeyError, ValueError, TypeError):
                continue
        if not c.any():
            return None
        pm = np.where(c > 0, s / np.maximum(c, 1), np.nan)
        pm = np.where(np.isnan(pm), np.nanmean(pm), pm)

        def _eaqi(p):
            bp = [(0, 0), (10, 20), (20, 40), (25, 60), (50, 80),
                  (75, 100), (150, 150)]
            for (x0, y0), (x1, y1) in zip(bp, bp[1:]):
                if p <= x1:
                    return y0 + (p - x0) / (x1 - x0) * (y1 - y0)
            return 150.0
        aqi = np.vectorize(_eaqi)(pm)
        now = datetime.utcnow().strftime('%Y-%m-%dT%H:00')
        print(f"✓ Air capteurs sensor.community : {int(c.sum())} mesures")
        return {'times': [now], 'aqi': [np.round(aqi, 1).tolist()],
                'pm25': [np.round(pm, 1).tolist()],
                'bbox': [FR_SIM_BBOX[1], FR_SIM_BBOX[0],
                         FR_SIM_BBOX[3], FR_SIM_BBOX[2]],
                'source': 'sensor.community (mesures capteurs)'}
    except Exception as e:
        print(f"sensors fallback err: {e}")
        return None


_wx_block_until = 0.0


def _wx_subset(bbox, n=6):
    """Champ vent d'une zone DECOUPE du store national (0 appel API)."""
    fr = (ZONES.get('france') or {}).get('wind_field')
    if not fr or fr.get('speed') is None:
        return None
    try:
        gl = np.asarray(fr['grid_lats']); gn = np.asarray(fr['grid_lons'])
        lon0, lat0, lon1, lat1 = bbox
        zl = np.linspace(lat0, lat1, n); zn = np.linspace(lon0, lon1, n)
        gy = np.clip((zl - gl[0]) / (gl[-1] - gl[0]) * (len(gl) - 1), 0, len(gl) - 1.001)
        gx = np.clip((zn - gn[0]) / (gn[-1] - gn[0]) * (len(gn) - 1), 0, len(gn) - 1.001)
        i0 = gy.astype(int); j0 = gx.astype(int)
        wy = (gy - i0)[None, :, None]; wx = (gx - j0)[None, None, :]
        i1 = np.minimum(i0 + 1, len(gl) - 1); j1 = np.minimum(j0 + 1, len(gn) - 1)

        def sub(M, vector_deg=False):
            M = np.asarray(M, dtype=float)
            if vector_deg:
                d = np.radians(M)
                ue = sub(np.sin(d)); vn = sub(np.cos(d))
                return ((np.degrees(np.arctan2(ue, vn)) + 360) % 360)
            return (M[:, i0][:, :, j0] * (1 - wy) * (1 - wx)
                    + M[:, i1][:, :, j0] * wy * (1 - wx)
                    + M[:, i0][:, :, j1] * (1 - wy) * wx
                    + M[:, i1][:, :, j1] * wy * wx)
        out = {'grid_lats': [round(float(x), 4) for x in zl],
               'grid_lons': [round(float(x), 4) for x in zn],
               'times': fr['times'], 'g': f'sub{n}'}
        for k in _WX_KEYS:
            if fr.get(k) is None:
                out[k] = None
            elif k == 'dir':
                out[k] = np.round(sub(fr[k], vector_deg=True), 0).tolist()
            else:
                out[k] = np.round(sub(fr[k]), 2).tolist()
        return out
    except Exception as e:
        print(f"wx subset err: {e}")
        return None


def _air_subset(bbox):
    """Champ air d'une zone DECOUPE dans le champ France stocke (0 appel API)."""
    fr = (ZONES.get('france') or {}).get('air_field')
    if not fr or not fr.get('aqi'):
        return None
    la0, lo0, la1, lo1 = fr['bbox']
    A = np.asarray(fr['aqi'], dtype=float)
    P = np.asarray(fr.get('pm25') or fr['aqi'], dtype=float)
    T, nl, nc = A.shape
    zlon0, zlat0, zlon1, zlat1 = bbox
    lats = np.linspace(zlat0, zlat1, 6)
    lons = np.linspace(zlon0, zlon1, 7)
    gy = np.clip((lats - la0) / (la1 - la0) * (nl - 1), 0, nl - 1)
    gx = np.clip((lons - lo0) / (lo1 - lo0) * (nc - 1), 0, nc - 1)
    i0 = np.floor(gy).astype(int); j0 = np.floor(gx).astype(int)
    i1 = np.minimum(i0 + 1, nl - 1); j1 = np.minimum(j0 + 1, nc - 1)
    wy = (gy - i0)[:, None]; wx = (gx - j0)[None, :]

    def sub(M):
        out = (M[:, i0][:, :, j0] * (1 - wy) * (1 - wx)
               + M[:, i1][:, :, j0] * wy * (1 - wx)
               + M[:, i0][:, :, j1] * (1 - wy) * wx
               + M[:, i1][:, :, j1] * wy * wx)
        return np.round(out, 1).tolist()
    return {'times': fr['times'], 'aqi': sub(A), 'pm25': sub(P),
            'bbox': [zlat0, zlon0, zlat1, zlon1]}


def fetch_air_field_bbox(bbox, n_lat=6, n_lon=7, past_days=10):
    """Champ air horaire pour une bbox : EAQI, PM2.5, PM10, CO, NO2, O3."""
    global _air_block_until
    if time.time() < _air_block_until:
        return None                      # quota API en cours de refroidissement
    lon0, lat0, lon1, lat1 = bbox
    lats = np.linspace(lat0, lat1, n_lat)
    lons = np.linspace(lon0, lon1, n_lon)
    laq, loq = [], []
    for la in lats:
        for lo in lons:
            laq.append(round(float(la), 3))
            loq.append(round(float(lo), 3))
    r = None
    for att in range(3):
        try:
            r = requests.get('https://air-quality-api.open-meteo.com/v1/air-quality', params={
                'latitude': ','.join(map(str, laq)), 'longitude': ','.join(map(str, loq)),
                'hourly': 'european_aqi,pm2_5,pm10,carbon_monoxide,'
                          'nitrogen_dioxide,ozone',
                'past_days': past_days, 'forecast_days': 7,
                'timezone': 'UTC'}, timeout=60)
            if r.status_code == 200:
                break
            if r.status_code == 429:     # rate-limite : on n'insiste PAS
                _air_block_until = time.time() + 900
                print("air bbox 429 -> pause 15 min")
                return None
        except requests.RequestException as e:
            print(f"air bbox fetch essai {att + 1}: {e}")
        time.sleep(4)
    if r is None or r.status_code != 200:
        print(f"air bbox KO ({'HTTP ' + str(r.status_code) if r is not None else 'reseau'})")
        return None
    res = r.json()
    if isinstance(res, dict):
        res = [res]
    times = res[0].get('hourly', {}).get('time', [])
    T = len(times)
    src_keys = {'aqi': 'european_aqi', 'pm25': 'pm2_5', 'pm10': 'pm10',
                'co': 'carbon_monoxide', 'no2': 'nitrogen_dioxide',
                'o3': 'ozone'}
    grids = {k: np.zeros((T, n_lat, n_lon)) for k in src_keys}
    for idx, x in enumerate(res[:n_lat * n_lon]):
        h = x.get('hourly', {})
        i, j = idx // n_lon, idx % n_lon
        for k, sk in src_keys.items():
            arr = h.get(sk) or []
            last = 0.0
            g = grids[k]
            for t in range(T):
                v = arr[t] if t < len(arr) else None
                if v is not None:
                    last = v
                g[t, i, j] = last        # persistance au-dela de l'horizon
    out = {'times': times, 'bbox': [lat0, lon0, lat1, lon1],
           'g': f'{n_lat}x{n_lon}'}
    for k in src_keys:
        out[k] = grids[k].round(1).tolist()
    return out


def fetch_wind_field_bbox(bbox, n=6, past_days=10):
    """Champ de vent n x n pour une bbox (passe parametrable + prev 8 j)."""
    lon0, lat0, lon1, lat1 = bbox
    grid_lats = list(np.linspace(lat0, lat1, n))
    grid_lons = list(np.linspace(lon0, lon1, n))
    laq, loq = [], []
    for la in grid_lats:
        for lo in grid_lons:
            laq.append(round(float(la), 4))
            loq.append(round(float(lo), 4))
    try:
        global _wx_block_until
        if time.time() < _wx_block_until:
            return None
        r = requests.get("https://api.open-meteo.com/v1/forecast", params={
            'latitude': ','.join(map(str, laq)),
            'longitude': ','.join(map(str, loq)),
            'hourly': 'wind_speed_10m,wind_direction_10m,relative_humidity_2m,'
                      'temperature_2m,soil_moisture_0_to_7cm,'
                      'weather_code,precipitation',
            'past_days': past_days, 'forecast_days': 8,
            'timezone': 'UTC'}, timeout=40)
        if r.status_code == 429:
            _wx_block_until = time.time() + 1800
            print("vent bbox 429 -> pause 30 min")
            return None
        if r.status_code != 200:
            return None
        results = r.json()
        if isinstance(results, dict):
            results = [results]
        times = results[0]['hourly']['time'][:(past_days + 8) * 24]
        arrs = {k: np.zeros((len(times), n, n)) for k in
                ('speed', 'dir', 'rh', 'temp', 'soil', 'wmo', 'rain')}
        keys = {'speed': 'wind_speed_10m', 'dir': 'wind_direction_10m',
                'rh': 'relative_humidity_2m', 'temp': 'temperature_2m',
                'soil': 'soil_moisture_0_to_7cm', 'wmo': 'weather_code',
                'rain': 'precipitation'}
        defaults = {'speed': 0, 'dir': 270, 'rh': 50, 'temp': 25, 'soil': 0.2,
                    'wmo': 0, 'rain': 0}
        for idx, res in enumerate(results):
            i, j = idx // n, idx % n
            hh = res.get('hourly', {})
            for k, ak in keys.items():
                arr = hh.get(ak) or []
                for t in range(len(times)):
                    v = arr[t] if t < len(arr) else None
                    arrs[k][t, i, j] = defaults[k] if v is None else v
        return {'grid_lats': [round(float(x), 4) for x in grid_lats],
                'grid_lons': [round(float(x), 4) for x in grid_lons],
                'times': times, 'g': f'{n}x{n}', 'bbox': list(bbox),
                **{k: v.round(2).tolist() for k, v in arrs.items()}}
    except Exception as e:
        print(f"wind zone error: {e}")
        return None


def _zone_refresh(z):
    """Charge/rafraîchit toutes les données locales d'une zone."""
    try:
        bbox = z['sim_bbox']
        map_key = os.getenv('NASA_FIRMS_MAP_KEY', 'DEMO_KEY')
        hotspots = []
        for product in _FIRMS_PRODUCTS:
            hotspots.extend(_fetch_firms_product(map_key, product, bbox))
        last_ts = max((h['timestamp'] for h in hotspots), default=None)
        sats = sorted({h.get('sat') for h in hotspots})
        clat, clon = z['lat'], z['lon']
        z['wind_field'] = (_wx_subset(bbox) or fetch_wind_field_bbox(bbox)
                           or z.get('wind_field'))
        wind_series = []
        wf = z['wind_field']
        if wf:
            sp = np.asarray(wf['speed']).mean(axis=(1, 2))
            dr = np.asarray(wf['dir']).mean(axis=(1, 2))
            rh = np.asarray(wf['rh']).mean(axis=(1, 2))
            tp = np.asarray(wf['temp']).mean(axis=(1, 2))
            for i, t in enumerate(wf['times']):
                wind_series.append({'timestamp': t,
                                    'wind_speed_10m_ms': float(sp[i]),
                                    'wind_direction_10m_deg': float(dr[i]),
                                    'relative_humidity_pct': float(rh[i]),
                                    'temperature_c': float(tp[i])})
        z['latest'] = {
            'timestamp': datetime.utcnow().isoformat(),
            'zone': {'id': z['id'], 'name': z.get('name', ''),
                     'view': z['view'], 'sim_bbox': list(bbox)},
            'fire_perimeter': {'centroid': {'lat': clat, 'lon': clon},
                               'distance_to_bordeaux_km': None,
                               'n_hotspots': len(hotspots)},
            'firms': {'hotspots': hotspots, 'source': 'NASA FIRMS',
                      'satellites': sats, 'last_detection': last_ts},
            'wind': {'hourly_wind': wind_series},
            'air': None,
            'pyro_watch': fetch_pyro_watch(clat, clon, hotspots),
            'response': _fire_response({'lat': clat, 'lon': clon,
                                        'frp_24h': 3000}),
        }
        z['air_field'] = (_air_subset(bbox) or fetch_air_field_bbox(bbox)
                          or z.get('air_field'))
        z['last_update'] = datetime.utcnow()
        z['ready'] = True
        # NDVI en dernier : GIBS peut prendre >1 min, la fiche est deja servie
        if z.get('veg') is None:
            z['veg'] = fetch_vegetation_bbox(bbox)
        z['smoke_past'] = _smoke_past_zone(z)
        print(f"✓ Zone {z['id']} ({z.get('name','')}) prête : {len(hotspots)} foyers")
    except Exception as e:
        print(f"zone refresh {z.get('id')} error: {e}")
        z['error'] = str(e)


def _get_zone(zid):
    """Récupère/crée une zone à partir de son id 'zLAT_LON'."""
    z = ZONES.get(zid)
    if z:
        z['last_access'] = time.time()
        return z
    try:
        la, lo = zid[1:].split('_')
        lat, lon = float(la), float(lo)
    except (ValueError, IndexError):
        return None
    fires = _cached('france_fires', 3600, fetch_france_fires) or {'clusters': []}
    best, bd = None, 0.35
    for c in fires['clusters']:
        d = abs(c['lat'] - lat) + abs(c['lon'] - lon)
        if d < bd:
            best, bd = c, d
    span = 1.0 if (best or {}).get('frp_24h', 0) >= 500 else 0.45
    sim, view = _zone_bboxes(lat, lon, span)
    z = {'id': zid, 'lat': lat, 'lon': lon,
         'name': (best or {}).get('name', ''),
         'sim_bbox': sim, 'view': view, 'ready': False,
         'ens': {}, 'last_access': time.time()}
    ZONES[zid] = z
    Thread(target=_zone_refresh, args=(z,), daemon=True).start()
    return z


FR_SIM_BBOX = (-5.2, 41.2, 9.8, 51.3)
FRANCE_VIEW = [[41.2, -5.2], [51.3, 9.8]]


def _refresh_france_zone(fr):
    """Construit/rafraîchit la pseudo-zone nationale."""
    z = ZONES.get('france') or {'id': 'france', 'ens': {}}
    ZONES['france'] = z
    try:
        hotspots = fr.get('hotspots', [])
        clusters = fr.get('clusters', [])
        last_ts = max((h['timestamp'] for h in hotspots), default=None)
        sats = sorted({h.get('sat') for h in hotspots if h.get('sat')})
        z['lat'], z['lon'] = 46.5, 2.0
        z['sim_bbox'] = FR_SIM_BBOX
        z['view'] = FRANCE_VIEW
        z['name'] = 'France'
        # STORE meteo incremental : passe telecharge une fois (ancre FIRE_T0),
        # puis seulement les heures recentes + la prevision a chaque cycle
        wx_prev = z.get('wind_field') or _warm_load('france_wx.json')
        first_wx = not (wx_prev and wx_prev.get('g') == '10x10'
                        and wx_prev.get('wmo') is not None)
        wf_new = fetch_wind_field_bbox(FR_SIM_BBOX, n=10,
                                       past_days=10 if first_wx else 2)
        z['wind_field'] = _merge_series(wx_prev, wf_new, _WX_KEYS) \
            or wx_prev or wf_new
        if wf_new and z['wind_field']:
            _warm_save('france_wx.json', z['wind_field'])
        # STORE air incremental (grille densifiee 10x14, polluants complets)
        air_prev = z.get('air_field') or _warm_load('france_air.json')
        first_air = not (air_prev and air_prev.get('g') == '10x14')
        af_new = fetch_air_field_bbox(FR_SIM_BBOX, n_lat=10, n_lon=14,
                                      past_days=10 if first_air else 2)
        af_new = _merge_series(air_prev if not first_air else None,
                               af_new, _AIR_KEYS) or af_new
        if af_new is not None:
            try:
                sens = fetch_air_sensors_fallback()
                if sens:
                    now_k = datetime.utcnow().strftime('%Y-%m-%dT%H:00')
                    times_af = af_new['times']
                    kk = next((i for i, t in enumerate(times_af)
                               if t >= now_k), len(times_af) - 1)
                    A = np.asarray(af_new['aqi'][kk])
                    P = np.asarray(af_new['pm25'][kk])
                    As = np.asarray(sens['aqi'][0])
                    Ps = np.asarray(sens['pm25'][0])
                    af_new['aqi'][kk] = np.maximum(A, As).round(0).tolist()
                    af_new['pm25'][kk] = np.maximum(P, Ps).round(1).tolist()
                    af_new['assim'] = now_k
                    print(f"✓ Assimilation capteurs : max PM2.5 mesure "
                          f"{float(Ps.max()):.0f} vs CAMS {float(P.max()):.0f} µg")
            except Exception as e_as:
                print(f"assimilation capteurs err: {e_as}")
            z['air_field'] = af_new
            _warm_save('france_air.json', af_new)
        elif z.get('air_field') is None:
            z['air_field'] = (_warm_load('france_air.json')
                              or fetch_air_sensors_fallback())
        top = clusters[0] if clusters else None
        z['latest'] = {
            'timestamp': datetime.utcnow().isoformat(),
            'zone': {'id': 'france', 'name': 'France', 'view': FRANCE_VIEW,
                     'sim_bbox': list(FR_SIM_BBOX)},
            'clusters': clusters,
            'fire_perimeter': {'centroid': {'lat': 46.5, 'lon': 2.0},
                               'distance_to_bordeaux_km': None,
                               'n_hotspots': len(hotspots)},
            'firms': {'hotspots': hotspots, 'source': 'NASA FIRMS',
                      'satellites': sats, 'last_detection': last_ts},
            'wind': {'hourly_wind': []},
            'air': None,
            'pyro_watch': (fetch_pyro_watch(top['lat'], top['lon'], hotspots)
                           if top else None),
        }
        wf = z['wind_field']
        if wf:
            sp = np.asarray(wf['speed']).mean(axis=(1, 2))
            dr = np.asarray(wf['dir']).mean(axis=(1, 2))
            rh = np.asarray(wf['rh']).mean(axis=(1, 2))
            tp = np.asarray(wf['temp']).mean(axis=(1, 2))
            z['latest']['wind']['hourly_wind'] = [
                {'timestamp': t, 'wind_speed_10m_ms': float(sp[i]),
                 'wind_direction_10m_deg': float(dr[i]),
                 'relative_humidity_pct': float(rh[i]),
                 'temperature_c': float(tp[i])}
                for i, t in enumerate(wf['times'])]
        z['smoke_past'] = _smoke_past_zone(z)
        z['last_update'] = datetime.utcnow()
        z['ready'] = True
        # NDVI en dernier : GIBS peut prendre >1 min, la zone est deja servie
        if z.get('veg') is None:
            z['veg'] = fetch_vegetation_bbox(FR_SIM_BBOX)
        print(f"✓ Zone FRANCE prête : {len(hotspots)} foyers, {len(clusters)} incendies")
    except Exception as e:
        print(f"zone france error: {e}")


def _france_top_clusters(n=4):
    fr = _cached('france_fires', 3600, fetch_france_fires) or {}
    return [c for c in fr.get('clusters', [])
            if c.get('active') and c.get('frp_24h', 0) >= 150][:n]


def compute_france_scenario(view, lutte):
    """Simulation France = fusion des ensembles locaux des principaux feux."""
    from src.fire_front import derive_view
    tops = _france_top_clusters()
    merged, missing = {}, []
    meta0, runs_used, pyro_sum = None, [], 0
    for k, c in enumerate(tops):
        zid = c['zone']
        z = ZONES.get(zid)
        if not z or not z.get('ready'):
            missing.append(c.get('name') or zid)
            if not z:
                _get_zone(zid)
            continue
        ent = z['ens'].get(lutte)
        ver_ok = ent and ent.get('views')
        if not ver_ok:
            missing.append(c.get('name') or zid)
            Thread(target=compute_zone_ensemble, args=(zid, lutte),
                   daemon=True).start()
            continue
        if view.startswith('m') and view[1:].isdigit() and ent.get('store') is not None:
            vk = f"m{int(view[1:]) % max(ent['views'].get('ref', {}).get('n_runs', 12), 1)}"
            d = ent['views'].get(vk)
            if d is None:
                d = _normalize_ts(derive_view(ent['store'], vk))
                ent['views'][vk] = d
        else:
            d = ent['views'].get(view if view in _VIEWS else 'ref')
        if not d:
            continue
        runs_used.append(d.get('n_runs', 0))
        pyro_sum += d.get('pyro_runs', 0)
        for f in d.get('frames', []):
            h = f['hour']
            m = merged.setdefault(h, {'hour': h, 'timestamp': f['timestamp'],
                                      'area_ha': 0, 'area_p10': 0, 'area_p90': 0,
                                      'new_points': [],
                                      'wind_speed_ms': f.get('wind_speed_ms'),
                                      'humidity_pct': f.get('humidity_pct'),
                                      'temp_c': f.get('temp_c')})
            m['area_ha'] += f.get('area_ha', 0)
            m['area_p10'] += f.get('area_p10', 0)
            m['area_p90'] += f.get('area_p90', 0)
            m['new_points'].extend(f.get('new_points') or [])
    frames = [merged[h] for h in sorted(merged)]
    return {'n_frames': len(frames), 'frames': frames,
            'n_runs': (min(runs_used) if runs_used else 0),
            'pyro_runs': pyro_sum, 'view': view, 'lutte': lutte,
            'n_fires': len(tops) - len(missing), 'computing': missing,
            'n_seeds': sum(c.get('n_24h', 0) for c in tops)}


def _zone_from_req():
    zid = request.args.get('zone', '')
    if not zid or zid == 'gironde':
        return None
    if zid == 'france':
        return ZONES.setdefault('france', {'id': 'france', 'ready': False, 'ens': {}})
    return _get_zone(zid)


def compute_zone_ensemble(zid, lutte='med', wait=False):
    """Ensemble Monte Carlo local d'une zone (à la demande, puis caché).

    wait=True (pipeline de fond) : attend le verrou au lieu de servir
    l'ancienne version — garantit un résultat frais.
    """
    z = ZONES.get(zid)
    if not z or not z.get('ready'):
        return None
    from src.fire_front import simulate_ensemble, derive_view
    ver = f"v8z:{z.get('last_update')}:{datetime.utcnow().strftime('%dT%H')}:{_sim_params_stamp()}"
    ent = z['ens'].get(lutte)
    if ent and ent.get('ver') == ver:
        return ent['views']
    _lk = _fr_lock if zid == 'france' else _sim_lock
    if wait:
        _lk.acquire()
    elif not _lk.acquire(blocking=False):
        return (ent or {}).get('views')
    try:
        ent = z['ens'].get(lutte)
        if ent and ent.get('ver') == ver:
            return ent['views']
        now_dt = datetime.utcnow()
        active = [h for h in z['latest']['firms']['hotspots']
                  if _parse_ts(h.get('timestamp'))
                  and (now_dt - _parse_ts(h['timestamp'])) <= timedelta(hours=ACTIVE_WINDOW_H)]
        now_key = now_dt.strftime('%Y-%m-%dT%H:00')
        wf = z.get('wind_field')
        wfx = None
        if wf and wf.get('times'):
            idx = [k for k, t in enumerate(wf['times']) if (t or '') >= now_key]
            if idx:
                wfx = {k: (wf[k] if k in ('grid_lats', 'grid_lons')
                           else [wf['times'][i] for i in idx] if k == 'times'
                           else (np.asarray(wf[k])[idx] if wf.get(k) is not None else None))
                       for k in ('grid_lats', 'grid_lons', 'times', 'speed',
                                 'dir', 'rh', 'temp', 'soil')}
        wind = [w for w in z['latest']['wind']['hourly_wind']
                if (w.get('timestamp') or '') >= now_key]
        lat0, lon0 = z['sim_bbox'][1], z['sim_bbox'][0]
        lat1, lon1 = z['sim_bbox'][3], z['sim_bbox'][2]
        is_fr = (zid == 'france')
        n_fires = len([c for c in (z['latest'].get('clusters') or []) if c.get('active')]) if is_fr else 1
        ign_rate = 0.0
        if is_fr:
            cutoff48 = (now_dt - timedelta(hours=48)).strftime('%Y-%m-%dT%H')
            n_new48 = len([c for c in (z['latest'].get('clusters') or [])
                           if (c.get('first_detection') or '') >= cutoff48])
            ign_rate = float(np.clip(n_new48 / 2.0, 4, 30))
        cutoff_b = now_dt - timedelta(hours=ACTIVE_WINDOW_H)
        burned_prev = [(h['lat'], h['lon'])
                       for h in z['latest']['firms']['hotspots']
                       if (_parse_ts(h.get('timestamp')) or now_dt) < cutoff_b]
        store = simulate_ensemble(
            active, wind, wind_field=wfx, veg_fuel=z.get('veg'),
            veg_bbox=z['sim_bbox'], max_hours=168, burned_pts=burned_prev,
            n_runs=int(os.getenv('FR_RUNS', '8')) if is_fr else int(os.getenv('ZONE_RUNS', '12')),
            scenario={'supp_level': _LUTTE.get(lutte, 1.0),
                      'cap_mult': float(np.clip(n_fires, 1, 15) ** 1.5),
                      'pyro_mult': 0.5 if is_fr else 1.0,
                      'ignition_rate': ign_rate,
                      **_sim_params()},
            smk_bbox=(lat0, lon0, lat1, lon1))
        if store is None:
            z['ens'][lutte] = {'ver': ver, 'views': {}}
            return {}
        store['smoke_k'] = calibrate_smoke_k()   # même ancrage CAMS que la Gironde
        views = {}
        for v in _VIEWS:
            views[v] = _normalize_ts(derive_view(store, v))
        z['ens'][lutte] = {'ver': ver, 'views': views, 'store': store}
        print(f"✓ Ensemble zone {zid} lutte={lutte} prêt")
        return views
    finally:
        _lk.release()


def _fire_response(c):
    """Moyens engages sur un feu : chiffres CURES (communiques officiels,
    warm deployments.json) si disponibles, sinon estimation d'echelle
    coherente avec le modele de lutte (calibree : Gironde 750 pompiers
    pour ~3300 MW de FRP 24 h)."""
    dep = _warm_load('deployments.json') or {}
    if 44.0 <= c.get('lat', 0) <= 45.4 and -1.6 <= c.get('lon', 0) <= -0.2 \
            and dep.get('gironde'):
        return {**dep['gironde'], 'curated': True}
    frp = float(c.get('frp_24h') or 0)
    if frp < 40:
        return None
    pompiers = int(np.clip(50 + frp * 0.22, 30, 2500))
    first = c.get('first_detection')
    dep_sol = dep_air = None
    if first:
        try:
            f0 = _parse_ts(first)
            dep_sol = (f0 + timedelta(hours=6)).strftime('%Y-%m-%dT%H:00')
            dep_air = (f0 + timedelta(hours=20)).strftime('%Y-%m-%dT%H:00')
        except (TypeError, ValueError):
            pass
    return {'pompiers': pompiers, 'camions': max(4, pompiers // 4),
            'aeronefs': (int(min(6, frp // 1200 + 1)) if frp > 700 else 0),
            'tracteurs': max(0, pompiers // 12),
            'depuis_sol': dep_sol, 'depuis_air': dep_air,
            'source': 'estimation (échelle du feu)', 'curated': False}


def fetch_pyro_watch(lat, lon, hotspots):
    """Real-time pyroCumulonimbus watch over the fire zone.

    Combines OBSERVED data: WMO weather code at the fire (95/96/99 = orage
    détecté), CAPE (convective energy available), and the fire's real
    intensity (sum of FRP over the last 24 h). A pyroCb is an episodic
    thunderstorm lasting hours — it needs an intense convective column."""
    try:
        r = requests.get('https://api.open-meteo.com/v1/forecast', params={
            'latitude': lat, 'longitude': lon,
            'current': 'cape,weather_code', 'timezone': 'UTC'}, timeout=15)
        cur = r.json().get('current', {}) if r.status_code == 200 else {}
        cape = float(cur.get('cape') or 0)
        wcode = int(cur.get('weather_code') or 0)
        now = datetime.utcnow()
        frp24 = sum(h.get('frp', 0) for h in hotspots
                    if _parse_ts(h.get('timestamp'))
                    and (now - _parse_ts(h['timestamp'])) <= timedelta(hours=24))
        storm_here = wcode in (95, 96, 99)
        if storm_here and frp24 > 500:
            status, lvl = 'ORAGE OBSERVÉ sur la zone du feu', 'fire'
        elif frp24 < 300:
            status, lvl = 'improbable — feu en fort déclin', 'ok'
        elif cape > 1000 and frp24 > 2000:
            status, lvl = 'conditions favorables (convection + feu intense)', 'warn'
        else:
            status, lvl = 'peu probable actuellement', 'ok'
        return {'status': status, 'level': lvl, 'cape': round(cape),
                'storm_observed': storm_here, 'frp24': round(frp24)}
    except Exception as e:
        print(f"pyro watch error: {e}")
        return None


def update_data():
    """Background thread: fetch latest data periodically."""
    global latest_data, last_update, _wind_field

    while True:
        try:
            print(f"[{datetime.utcnow().isoformat()}] Fetching real-time data...")

            firms = fetch_nasa_firms()
            wind = fetch_extended_wind_forecast()
            _wind_field = (fetch_wind_field()
                           or _wx_subset((-1.75, 44.20, -0.25, 45.55))
                           or _wind_field or _warm_load('gironde_wx.json'))
            if _wind_field is not None and isinstance(
                    _wind_field.get('speed'), list):
                _warm_save('gironde_wx.json', _wind_field)
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
                'pyro_watch': fetch_pyro_watch(centroid_lat, centroid_lon,
                                               firms['hotspots']),
                'response': _fire_response({'lat': centroid_lat,
                                            'lon': centroid_lon,
                                            'frp_24h': 3000}),
            }

            last_update = datetime.utcnow()
            _warm_save('latest.json', {'latest': latest_data,
                                       'last_update': last_update.isoformat()})
            print(f"✓ Data updated. Fire distance: {dist_km:.1f} km")

            try:
                compute_simulation()  # pre-warm the 7-day sim so requests are instant
                print("✓ Simulation pre-computed")
            except Exception as e:
                print(f"sim precompute error: {e}")

            # ---- mode France : zone nationale + ensembles des top incendies ----
            try:
                fr = fetch_france_fires()
                if fr.get('hotspots'):
                    _warm_save('france_fires.json', fr)
                _layer_cache['france_fires'] = (time.time(), fr)
                _refresh_france_zone(fr)
                # les ensembles nationaux tournent dans leur propre thread
                # (_france_ens_loop) pour ne pas retarder le cycle horaire
            except Exception as e:
                print(f"france precompute error: {e}")

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

_SEO_GIRONDE = {
    'title': "Incendie Bordeaux — carte temps réel du feu de forêt en Gironde et simulation 7 jours",
    'desc': ("Carte en direct de l'incendie près de Bordeaux : foyers détectés par satellite "
             "(NASA FIRMS, Meteosat ~10 min), qualité de l'air, vent, et simulation Monte Carlo "
             "de la propagation à 7 jours (vent, sécheresse, végétation, pompiers, pyrocumulonimbus)."),
    'url': 'https://incendiebordeaux.fr/',
    'og_title': "Incendie Bordeaux — carte temps réel & simulation",
    'og_desc': ("Foyers satellites en direct, qualité de l'air, et prévision de propagation "
                "à 7 jours par ensemble Monte Carlo."),
    'h1': "Feu à Bordeaux : carte en temps réel et simulation de l'incendie de forêt en Gironde",
}
_SEO_FRANCE = {
    'title': "Feux de forêt en France — carte temps réel des incendies et simulation 7 jours",
    'desc': ("Tous les feux de forêt actifs en France sur une carte en temps réel : foyers détectés "
             "par satellite (NASA FIRMS, Meteosat ~10 min), qualité de l'air, vent, et simulation "
             "Monte Carlo de la propagation de chaque incendie à 7 jours (météo, sécheresse, "
             "végétation, pompiers, pyrocumulonimbus)."),
    'url': 'https://feux-de-foret.fr/',
    'og_title': "Feux de forêt en France — carte temps réel & simulation des incendies",
    'og_desc': ("Incendies actifs détectés par satellite dans toute la France, qualité de l'air "
                "et prévision de propagation à 7 jours par ensemble Monte Carlo."),
    'h1': ("Feux de forêt en France : carte en temps réel des incendies actifs, "
           "qualité de l'air et simulation de propagation"),
}


def _site_for_host():
    """Site dedie a un incendie (dijon.feux-de-foret.fr…) ? (warm, extensible
    sans redeploiement)."""
    host = (request.host or '').split(':')[0]
    return host, (_warm_load('fire_sites.json') or {}).get(host)


def _html(t):
    host, site = _site_for_host()
    if site:
        seo = {'title': site['title'], 'desc': site['desc'],
               'url': f'https://{host}/', 'og_title': site['title'],
               'og_desc': site['desc'], 'h1': site['h1'],
               'zone': site['zone']}
    else:
        is_fr = 'feux-de-foret' in host
        seo = dict(_SEO_FRANCE if is_fr else _SEO_GIRONDE)
        seo['zone'] = ''
    resp = Response(render_template(t, seo=seo), mimetype='text/html')
    resp.headers['Cache-Control'] = 'no-cache'
    return resp


@app.route('/')
def index():
    """Bordeaux map by default; national view on feux-de-foret.fr."""
    return _html('index.html')


def _hour_bucket(ts):
    """'2026-07-26T14:37:00Z' -> '2026-07-26T14:00Z' (hour bucket key)."""
    try:
        return ts[:13] + ':00Z'
    except (TypeError, IndexError):
        return None


# A hotspot counts as "active" for this many hours. 24 h guarantees the window
# always spans at least one daytime AND one nighttime satellite pass — night
# passes detect far fewer pixels (fires calm down at night), so a shorter
# window made the current activity look artificially tiny.
ACTIVE_WINDOW_H = 24


def _parse_ts(ts):
    try:
        return datetime.strptime(ts[:19], '%Y-%m-%dT%H:%M:%S')
    except (ValueError, TypeError):
        return None


@app.route('/api/fire-history')
def api_fire_history():
    """Compact hotspot timeline: each detection once with its hour offset.

    The client renders any hour by filtering detections whose age is within
    the active window, and fades them with age (fresh = bright, dying = dark
    red). Much smaller than per-frame point lists.
    """
    z = _zone_from_req()
    if z is not None:
        if not z.get('ready'):
            return jsonify({'error': 'Zone loading'}), 503
        src_data = z['latest']
    elif latest_data:
        src_data = latest_data
    else:
        return jsonify({'error': 'No data available'}), 503
    hotspots = src_data['firms'].get('hotspots', [])
    now = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    start = FIRE_T0
    pts = []
    for h in hotspots:
        dt = _parse_ts(h.get('timestamp'))
        if not dt:
            continue
        off = (dt - start).total_seconds() / 3600.0
        if off <= -ACTIVE_WINDOW_H:
            continue  # never visible on the timeline
        pts.append({'lat': h['lat'], 'lon': h['lon'],
                    'frp': round(h.get('frp', 0.0), 1), 'h': round(off, 1)})
    return jsonify({
        'source': src_data['firms'].get('source'),
        'start': start.strftime('%Y-%m-%dT%H:00Z'),
        'now': now.strftime('%Y-%m-%dT%H:00Z'),
        'n_hours': int((now - start).total_seconds() // 3600) + 1,
        'window_h': ACTIVE_WINDOW_H,
        'points': pts,
    })


_sim_cache = {'key': None, 'data': None}
def _sim_params():
    """Paramètres de propagation calibrés par backtest (persistés)."""
    saved = _warm_load('sim_params.json') or {}
    return saved.get('params') or {}


def _sim_params_stamp():
    saved = _warm_load('sim_params.json') or {}
    return saved.get('date', 'none')


_sim_lock = __import__('threading').Lock()
_fr_lock = __import__('threading').Lock()   # ensembles nationaux, independants

# ---- interactive scenario dashboard ---------------------------------------
_LVL = ('low', 'med', 'high')
_SCEN_FACTORS = ('pompiers', 'vent', 'temp', 'secheresse', 'pyro')
_scenario_cache = {}   # levels-key -> {'ver':…, 'data':…}


def _scenario_mods(p, v, t, s, y):
    return {
        'supp_base':   {'low': 0.0, 'med': 0.75, 'high': 0.95}[p],
        'speed_mult_g': {'low': 0.6, 'med': 1.0, 'high': 1.5}[v],
        'temp_off_g':  {'low': -5.0, 'med': 0.0, 'high': 6.0}[t],
        'soil_off_g':  {'low': 0.12, 'med': 0.0, 'high': -0.10}[s],
        'pyro_cal':    {'low': 1.0, 'med': 1.15, 'high': 1.35}[y],
        'pyro_dir_std': {'low': 0.0, 'med': 20.0, 'high': 40.0}[y],
        'pyro_spot':   {'low': 0.0, 'med': 0.6, 'high': 1.8}[y],
    }


def _sim_inputs():
    """(active hotspots, wind series, sliced wind field, fuel) for a sim run."""
    now_dt = datetime.utcnow()
    active = []
    for h in latest_data['firms'].get('hotspots', []):
        dt = _parse_ts(h.get('timestamp'))
        if dt and (now_dt - dt) <= timedelta(hours=ACTIVE_WINDOW_H):
            active.append(h)
    wind = latest_data['wind'].get('hourly_wind', [])
    now_key = now_dt.strftime('%Y-%m-%dT%H:00')
    future = [w for w in wind if (w.get('timestamp') or '') >= now_key]
    wind = future if future else wind
    wf = None
    if _wind_field and _wind_field.get('times'):
        idx = [k for k, t in enumerate(_wind_field['times']) if (t or '') >= now_key]
        if idx:
            wf = {k: (_wind_field[k] if k in ('grid_lats', 'grid_lons')
                      else [_wind_field['times'][i] for i in idx] if k == 'times'
                      else (_wind_field[k][idx] if _wind_field.get(k) is not None else None))
                  for k in ('grid_lats', 'grid_lons', 'times', 'speed', 'dir',
                            'rh', 'temp', 'soil')}
    return active, wind, wf, _veg_cache.get('fuel'), now_key


_LUTTE = {'low': 0.5, 'med': 1.0, 'high': 1.6}
_ens_store = {l: {'ver': None, 'views': {}, 'store': None} for l in _LUTTE}
_VIEWS = ('ref', 'opt', 'pess', 'pyro')


def compute_ensemble(lutte='med'):
    """One honest ensemble per firefighting level; views derived from it."""
    if not latest_data or lutte not in _LUTTE:
        return None
    from src.fire_front import simulate_ensemble, derive_view
    ent = _ens_store[lutte]
    ver = f"v8:{last_update}:{datetime.utcnow().strftime('%dT%H')}:{_sim_params_stamp()}"
    if ent['ver'] == ver:
        return ent['views']
    if not _sim_lock.acquire(blocking=False):
        return ent['views'] or None
    try:
        if ent['ver'] == ver:
            return ent['views']
        hotspots, wind, wf, fuel, _ = _sim_inputs()
        _now_g = datetime.utcnow()
        burned_g = [(h['lat'], h['lon'])
                    for h in (latest_data or {}).get('firms', {}).get('hotspots', [])
                    if (_parse_ts(h.get('timestamp')) or _now_g)
                    < _now_g - timedelta(hours=ACTIVE_WINDOW_H)]
        store = simulate_ensemble(
            hotspots, wind, wind_field=wf, veg_fuel=fuel, veg_bbox=SIM_BBOX,
            max_hours=168, n_runs=int(os.getenv('ENS_RUNS', '16')),
            burned_pts=burned_g,
            scenario={'supp_level': _LUTTE[lutte], **_sim_params()})
        store['smoke_k'] = calibrate_smoke_k()   # ancré sur le panache RÉEL
        views = {}
        for v in _VIEWS:
            views[v] = _normalize_ts(derive_view(store, v))
        ent['views'] = views
        ent['store'] = store   # gardé pour dériver les tirages 'mK'
        ent['ver'] = ver
        _warm_save(f'views_{lutte}.json', {'ver': ver, 'views': views})
        print(f"✓ Ensemble lutte={lutte} {ver}: {len(views)} vues prêtes")
    finally:
        _sim_lock.release()
    return ent['views']


def _normalize_ts(d):
    for f in d.get('frames', []):
        ts = f.get('timestamp')
        if ts and not ts.endswith('Z'):
            f['timestamp'] = ts + 'Z'
    return d


@app.route('/api/scenario')
def api_scenario():
    """Ensemble views: ?view=ref (défaut) | opt | pess | pyro."""
    if not latest_data:
        return jsonify({'error': 'No data available'}), 503
    lutte = request.args.get('lutte', 'med')
    if lutte not in _LUTTE:
        lutte = 'med'
    z = _zone_from_req()
    if z is not None and z.get('id') == 'france':
        # LECTURE SEULE : le calcul national vit dans _france_ens_loop ;
        # on sert toujours la dernière version disponible, jamais de calcul
        # déclenché par une requête utilisateur.
        view_f = request.args.get('view', 'ref')
        if not (view_f in _VIEWS or (view_f.startswith('m') and view_f[1:].isdigit())):
            view_f = 'ref'
        ent_f = (z.get('ens') or {}).get(lutte) or {}
        hr_ok = ent_f.get('hr_ver') == ent_f.get('ver')
        data_f = (((ent_f.get('views_hr') or {}).get(view_f) if hr_ok else None)
                  or (ent_f.get('views') or {}).get(view_f))
        if data_f is None and ent_f.get('store') is not None:
            from src.fire_front import derive_view
            data_f = _normalize_ts(derive_view(ent_f['store'], view_f))
            ent_f['views'][view_f] = data_f
        if data_f is not None:
            data_f = dict(data_f); data_f['lutte'] = lutte
        return (jsonify(data_f if data_f else {'error': 'Computing'}),
                (200 if data_f else 503))
    if z is not None:
        if not z.get('ready'):
            return jsonify({'error': 'Zone loading'}), 503
        view_z = request.args.get('view', 'ref')
        if not (view_z in _VIEWS or (view_z.startswith('m') and view_z[1:].isdigit())):
            view_z = 'ref'
        views_z = compute_zone_ensemble(z['id'], lutte)
        data_z = (views_z or {}).get(view_z)
        if data_z is None and views_z is not None and z['ens'].get(lutte, {}).get('store') is not None:
            from src.fire_front import derive_view
            data_z = _normalize_ts(derive_view(z['ens'][lutte]['store'], view_z))
            views_z[view_z] = data_z
        if data_z is not None:
            data_z = dict(data_z)
            data_z['lutte'] = lutte
        return (jsonify(data_z if data_z else {'error': 'Computing'}),
                (200 if data_z else 503))
    ent = _ens_store[lutte]
    views = ent['views'] or compute_ensemble(lutte)
    view = request.args.get('view', 'ref')
    if not (view in _VIEWS or (view.startswith('m') and view[1:].isdigit())):
        view = 'ref'
    data = (views or {}).get(view)
    if data is None and views is not None and ent.get('store') is not None:
        # tirage individuel 'mK' : dérivé à la demande puis mis en cache
        from src.fire_front import derive_view
        data = _normalize_ts(derive_view(ent['store'], view))
        views[view] = data
    if data is not None:
        data = dict(data); data['lutte'] = lutte
    return jsonify(data if data else {'error': 'Computing'}), (200 if data else 503)


def compute_simulation():
    """Background precompute: one ensemble per firefighting level."""
    out = compute_ensemble('med')
    for l in ('low', 'high'):
        try:
            compute_ensemble(l)
        except Exception as e:
            print(f"ensemble lutte={l} erreur: {e}")
    return out


def _obsolete_compute_simulation():
    if not latest_data:
        return None
    from src.fire_front import simulate_monte_carlo
    now_dt = datetime.utcnow()
    active = []
    for h in latest_data['firms'].get('hotspots', []):
        dt = _parse_ts(h.get('timestamp'))
        if dt and (now_dt - dt) <= timedelta(hours=ACTIVE_WINDOW_H):
            active.append(h)
    hotspots = active
    wind = latest_data['wind'].get('hourly_wind', [])
    now_key = now_dt.strftime('%Y-%m-%dT%H:00')
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
                'temp': _wind_field['temp'][idx],
                'soil': (_wind_field['soil'][idx]
                         if _wind_field.get('soil') is not None else None),
            }
        else:
            wf = _wind_field
    fuel = _veg_cache.get('fuel')

    key = f"{last_update}:{len(hotspots)}:{now_key}:{wf is not None}:{fuel is not None}:mc2"
    if _sim_cache['key'] == key:
        return _sim_cache['data']
    # Only ONE thread computes the ensembles; others get the previous result.
    if not _sim_lock.acquire(blocking=False):
        return _sim_cache['data']
    try:
        n_runs = int(os.getenv('MC_RUNS', '8'))
        both = {}
        for mode, supp in (('lutte', True), ('libre', False)):
            data = simulate_monte_carlo(
                hotspots, wind, wind_field=wf, veg_fuel=fuel, veg_bbox=SIM_BBOX,
                max_hours=168, emit_every=3, n_runs=n_runs, suppression=supp)
            # normalise timestamps to explicit UTC so the client renders the
            # same local time as past frames (fixes the 12h/14h mismatch)
            for f in data.get('frames', []):
                ts = f.get('timestamp')
                if ts and not ts.endswith('Z'):
                    f['timestamp'] = ts + 'Z'
            both[mode] = data
        _sim_cache['data'] = both
        _sim_cache['key'] = key
    finally:
        _sim_lock.release()
    return _sim_cache['data']


@app.route('/api/simulation')
def api_simulation():
    """Compat: serves the reference view of the ensemble."""
    if not latest_data:
        return jsonify({'error': 'No data available'}), 503
    views = _ens_store['med']['views'] or compute_ensemble('med')
    data = (views or {}).get('ref')
    return jsonify(data if data else {'error': 'Computing'}), (200 if data else 503)


@app.route('/api/vegetation.png')
def api_vegetation_png():
    """Fuel map as a translucent PNG overlay (green=forest, blue=water)."""
    z = _zone_from_req()
    if z is not None:
        fuel = z.get('veg')
        if fuel is None and z.get('ready'):
            fuel = fetch_vegetation_bbox(z['sim_bbox'])
            z['veg'] = fuel
    else:
        fuel = _veg_cache.get('fuel')
        if fuel is None:
            fuel = fetch_vegetation()   # lazy load if the boot fetch failed
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
                    headers={'Cache-Control': 'public, max-age=300'})


@app.route('/api/windfield')
def api_windfield():
    """Spatial wind field (grid of arrows): past 5 days AND forecast hours."""
    z = _zone_from_req()
    wf = z.get('wind_field') if (z and z.get('ready')) else (_wind_field if z is None else None)
    if not wf:
        return jsonify({'error': 'No wind field'}), 503
    return jsonify({
        'grid_lats': wf['grid_lats'],
        'grid_lons': wf['grid_lons'],
        'times': wf['times'],
        'speed': np.asarray(wf['speed']).round(1).tolist(),
        'dir': np.asarray(wf['dir']).round(0).tolist(),
        'temp': np.asarray(wf['temp']).round(0).tolist(),
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
    """JSON API endpoint for real-time data (zone-aware)."""
    z = _zone_from_req()
    if z is not None:
        if not z.get('ready'):
            return jsonify({'error': 'Zone loading'}), 503
        dz = dict(z['latest'])
        if z.get('sim_score'):
            dz['sim_score'] = z['sim_score']
        if z.get('wind_field') is not None:
            ms = _met_series(z['wind_field'], z['lat'], z['lon'])
            if ms:
                dz['met'] = ms
        return jsonify(dz)
    if latest_data:
        d = dict(latest_data)
        d['zone'] = {'id': 'gironde', 'name': 'Gironde',
                     'view': [[44.33, -1.35], [45.06, -0.45]],
                     'sim_bbox': list(SIM_BBOX)}
        sc_g = _warm_load('sim_score_gironde.json')
        if sc_g:
            d['sim_score'] = sc_g
        ct = (d.get('fire_perimeter') or {}).get('centroid') or {}
        if _wind_field is not None and ct:
            ms = _met_series(_wind_field, ct.get('lat', 44.6), ct.get('lon', -0.9))
            if ms:
                d['met'] = ms
        return jsonify(d)
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


# TomTom traffic incidents across the whole region (2 tiles, each < 10 000 km²).
_TT_TILES = ["-1.40,44.35,-0.78,45.20", "-0.86,44.45,-0.30,45.10"]
_TT_FIELDS = ('{incidents{geometry{type,coordinates},properties{iconCategory,'
              'magnitudeOfDelay,events{description},delay,length,roadNumbers}}}')
_TT_CAT = {0: 'inconnu', 1: 'accident', 6: 'bouchon', 7: 'voie fermée',
           8: 'route coupée', 9: 'travaux', 14: 'panne'}


@app.route('/api/traffic')
def api_traffic():
    """Region-wide road traffic (TomTom): jams, accidents, impactful closures."""
    def prod():
        key = os.getenv('TOMTOM_KEY')
        if not key:
            return None
        out, seen = [], set()
        for bbox in _TT_TILES:
            r = requests.get(
                "https://api.tomtom.com/traffic/services/5/incidentDetails",
                params={'key': key, 'bbox': bbox, 'fields': _TT_FIELDS,
                        'language': 'fr-FR'}, timeout=25)
            if r.status_code != 200:
                continue
            for inc in r.json().get('incidents', []):
                p = inc.get('properties', {})
                g = inc.get('geometry', {})
                cat = p.get('iconCategory')
                delay = p.get('delay') or 0
                # Keep only incidents with a real traffic impact (delay) or an
                # accident. This drops the hundreds of permanent/forest-track
                # "road closed" markers that carry no delay.
                if not (cat == 1 or delay > 0):
                    continue
                coords = g.get('coordinates') or []
                if g.get('type') == 'LineString' and coords:
                    line = [[c[1], c[0]] for c in coords[::max(1, len(coords) // 20)]]
                    pt = line[0]
                elif g.get('type') == 'Point' and coords:
                    line, pt = None, [coords[1], coords[0]]
                else:
                    continue
                road = (p.get('roadNumbers') or [None])[0]
                keyid = (round(pt[0], 4), round(pt[1], 4), cat)
                if keyid in seen:
                    continue
                seen.add(keyid)
                out.append({
                    'cat': _TT_CAT.get(cat, 'incident'),
                    'closed': cat in (7, 8),
                    'road': road,
                    'delay': delay,
                    'desc': (p.get('events') or [{}])[0].get('description'),
                    'line': line, 'pt': pt,
                })
        return {'incidents': out}
    data = _cached('traffic', 120, prod) or {'incidents': []}
    return jsonify(data)


# French aerial firefighting call signs (Sécurité Civile + military reinforcement)
_FIRE_KW = ('PELICAN', 'PELIC', 'MILAN', 'DRAGON', 'CANADAIR', 'FIRE', 'BOMB',
            'COTAM', 'FRAF', 'A400', 'FENNEC')


def _opensky_track(icao24):
    """Recent trajectory [[lat,lon],...] of an aircraft (OpenSky tracks API)."""
    try:
        r = requests.get("https://opensky-network.org/api/tracks/all",
                         params={'icao24': icao24, 'time': 0}, timeout=15)
        if r.status_code != 200:
            return []
        return [[round(p[1], 4), round(p[2], 4)] for p in (r.json().get('path') or [])
                if p[1] is not None and p[2] is not None]
    except Exception:
        return []


@app.route('/api/aircraft')
def api_aircraft():
    """Firefighting aircraft over France (OpenSky ADS-B), with recent tracks."""
    def prod():
        # whole of France so bombers are caught wherever the fire is
        r = requests.get("https://opensky-network.org/api/states/all",
                         params={'lamin': 41.0, 'lomin': -5.5,
                                 'lamax': 51.5, 'lomax': 10.0}, timeout=20)
        if r.status_code != 200:
            return None
        out = []
        for s in r.json().get('states') or []:
            cs = (s[1] or '').strip()
            icao = s[0]
            lon, lat, baro, trk, geo = s[5], s[6], s[7], s[10], s[13]
            if lat is None or lon is None:
                continue
            alt = baro if baro is not None else geo
            is_fire = any(k in cs.upper() for k in _FIRE_KW)
            if not is_fire:
                continue  # only confirmed firefighting call signs
            out.append({'callsign': cs or '—', 'icao24': icao, 'lat': lat, 'lon': lon,
                        'alt': round(alt) if alt is not None else None,
                        'heading': trk, 'fire': True,
                        'track': _opensky_track(icao)})
        return out
    return jsonify({'aircraft': _cached('aircraft', 60, prod) or []})


def _eum_token():
    key = os.getenv('EUM_CONSUMER_KEY')
    sec = os.getenv('EUM_CONSUMER_SECRET')
    if not key:
        return None
    r = requests.post('https://api.eumetsat.int/token', auth=(key, sec),
                      data={'grant_type': 'client_credentials'}, timeout=15)
    return r.json().get('access_token') if r.status_code == 200 else None


def fetch_meteosat_fire():
    """Geostationary fire detection (MTG FCI, every ~10 min) over the Gironde."""
    import io as _io
    import zipfile
    import tempfile
    import urllib.parse
    tok = _eum_token()
    if not tok:
        return None
    h = {'Authorization': 'Bearer ' + tok}
    coll = 'EO:EUM:DAT:0682'
    s = requests.get('https://api.eumetsat.int/data/search-products/1.0.0/os',
                     params={'format': 'json', 'pi': coll, 'c': 1, 'si': 0},
                     headers=h, timeout=20)
    feats = s.json().get('features', []) if s.status_code == 200 else []
    if not feats:
        return None
    pid = feats[0]['id']
    tstr = (feats[0].get('properties', {}).get('date') or '')
    url = ('https://api.eumetsat.int/data/download/1.0.0/collections/'
           'EO%3AEUM%3ADAT%3A0682/products/' + urllib.parse.quote(pid, safe=''))
    z = requests.get(url, headers=h, timeout=90)
    if z.status_code != 200:
        return None
    zf = zipfile.ZipFile(_io.BytesIO(z.content))
    ncname = next((n for n in zf.namelist() if n.endswith('.nc')), None)
    if not ncname:
        return None
    with tempfile.NamedTemporaryFile(suffix='.nc', delete=False) as tf:
        tf.write(zf.read(ncname))
        path = tf.name
    try:
        import netCDF4
        import numpy as _np
        from pyproj import Proj
        ds = netCDF4.Dataset(path)
        x = ds.variables['x'][:]
        y = ds.variables['y'][:]
        fr = ds.variables['fire_result'][:]
        ds.close()
        H = 35786400.0
        P = Proj(proj='geos', h=H, a=6378137.0, b=6356752.0, lon_0=0, sweep='y')
        rows, cols = _np.where((fr >= 1) & (fr <= 3))
        fires = []
        for r, c in zip(rows.tolist(), cols.tolist()):
            lo, la = P(float(x[c]) * H, float(y[r]) * H, inverse=True)
            if -5.2 <= lo <= 9.8 and 41.2 <= la <= 51.3:
                fires.append({'lat': round(float(la), 4), 'lon': round(float(lo), 4),
                              'level': int(fr[r, c])})
        end = tstr.split('/')[-1] if tstr else None
        return {'fires': fires, 'time': end, 'n_world': int(len(rows))}
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


@app.route('/api/meteosat')
def api_meteosat():
    """Near-real-time (~10 min) geostationary fire pixels over the Gironde."""
    data = _cached('meteosat', 480, fetch_meteosat_fire)
    if data is None:
        return jsonify({'fires': [], 'time': None, 'available': False})
    z = _zone_from_req()
    if z is not None:
        b = z['sim_bbox']   # lon0, lat0, lon1, lat1
        fires = [f for f in data['fires']
                 if b[1] <= f['lat'] <= b[3] and b[0] <= f['lon'] <= b[2]]
    else:
        fires = [f for f in data['fires']
                 if 44.15 <= f['lat'] <= 45.4 and -1.7 <= f['lon'] <= -0.3]
    return jsonify({**data, 'fires': fires, 'available': True})


@app.route('/api/air-quality.png')
def api_air_quality_png():
    """Smooth Windy-style air-quality field (EAQI), cubic-interpolated raster.

    Source: Open-Meteo air quality = CAMS European model (~11 km), which
    already integrates wind-driven dispersion and thermal mixing — the plume
    shapes follow the wind. We sample a grid then upscale with cubic
    interpolation + gaussian smoothing for continuous contours.
    """
    def prod():
        n_lat, n_lon = 9, 10
        lat0, lat1, lon0, lon1 = 44.20, 45.55, -1.75, -0.25
        lats = np.linspace(lat0, lat1, n_lat)
        lons = np.linspace(lon0, lon1, n_lon)
        laq, loq = [], []
        for la in lats:
            for lo in lons:
                laq.append(round(float(la), 3))
                loq.append(round(float(lo), 3))
        r = requests.get('https://air-quality-api.open-meteo.com/v1/air-quality', params={
            'latitude': ','.join(map(str, laq)), 'longitude': ','.join(map(str, loq)),
            'current': 'european_aqi', 'timezone': 'UTC'}, timeout=25)
        if r.status_code != 200:
            return None
        res = r.json()
        if isinstance(res, dict):
            res = [res]
        vals = np.full(n_lat * n_lon, np.nan)
        for idx, x in enumerate(res[:n_lat * n_lon]):
            v = x.get('current', {}).get('european_aqi')
            if v is not None:
                vals[idx] = float(v)
        grid = vals.reshape(n_lat, n_lon)
        if np.isnan(grid).all():
            return None
        # fill occasional missing points with the mean
        grid = np.where(np.isnan(grid), np.nanmean(grid), grid)

        from scipy.interpolate import RegularGridInterpolator
        from scipy.ndimage import gaussian_filter
        f = RegularGridInterpolator((lats, lons), grid, method='cubic')
        fy = np.linspace(lat0, lat1, 320)
        fx = np.linspace(lon0, lon1, 400)
        YY, XX = np.meshgrid(fy, fx, indexing='ij')
        fine = f(np.column_stack([YY.ravel(), XX.ravel()])).reshape(320, 400)
        fine = gaussian_filter(fine, sigma=2.5)

        import matplotlib.colors as mcolors
        cmap = mcolors.LinearSegmentedColormap.from_list('aqi', [
            (0.00, '#2e7d32'), (0.18, '#8bc34a'), (0.34, '#cddc39'),
            (0.50, '#ffb300'), (0.66, '#fb8c00'), (0.82, '#e53935'),
            (1.00, '#8e24aa')])
        norm = np.clip(fine / 120.0, 0, 1)
        rgba = cmap(norm)
        rgba[..., 3] = np.clip(0.20 + norm * 0.65, 0.2, 0.8)
        img = np.flipud(rgba)  # row 0 = north
        import matplotlib.image as mpimg
        buf = io.BytesIO()
        mpimg.imsave(buf, img, format='png')
        buf.seek(0)
        return buf.read()

    png = _cached('airpng', 1800, prod)
    if png is None:
        return jsonify({'error': 'No air data'}), 503
    return Response(png, mimetype='image/png',
                    headers={'Cache-Control': 'public, max-age=900'})


def fetch_air_field():
    """Champ air Gironde : simple vue du fetch bbox commun (polluants inclus)."""
    return fetch_air_field_bbox((-1.75, 44.20, -0.25, 45.55),
                                n_lat=9, n_lon=10)


@app.route('/api/air-field')
def api_air_field():
    """Champ air horaire pour le client (index observé + PM2.5 pour le futur)."""
    z = _zone_from_req()
    if z is not None:
        if not z.get('ready'):
            return jsonify({'error': 'Zone loading'}), 503
        data = z.get('air_field')
    else:
        data = _cached('airfield', 1800, fetch_air_field)
    if data is None:
        return jsonify({'error': 'No air field'}), 503
    out = dict(data)
    if z is not None:
        sp = z.get('smoke_past')
        if sp:
            out['smoke_past'] = sp
    else:
        sp = _cached('smoke_past_gironde', 1800, _smoke_past_gironde)
        if sp:
            out['smoke_past'] = sp
    return jsonify(out)


def _sparse_smoke(pred, n_past, times_af, bbox, shape):
    """Encode la reanalyse {h: grille} en frames eparses pour le client."""
    frames = {}
    for h in range(n_past):
        g = pred.get(h)
        if g is None:
            continue
        arr = np.asarray(g)
        rr, cc = np.where(arr >= 3)
        if len(rr):
            frames[str(h)] = [[int(r), int(c), int(arr[r, c])]
                              for r, c in zip(rr, cc)]
    return {'t0': times_af[0], 'shape': list(shape),
            'bbox': list(bbox), 'frames': frames}


def _smoke_past_gironde():
    """Reanalyse du panache REEL (detections FIRMS + vents observes,
    emissions calibrees) pour l'affichage des heures passees."""
    try:
        from src.fire_front import hindcast_smoke, _SMK_BBOX, _SMK_NR, _SMK_NC
        af = _cached('airfield', 1800, fetch_air_field)
        if not af or not latest_data or _wind_field is None:
            return None
        times_af = af['times']
        now_key = datetime.utcnow().strftime('%Y-%m-%dT%H:00')
        n_past = sum(1 for t in times_af if t < now_key)
        if n_past < 10:
            return None
        t0 = datetime.strptime(times_af[0], '%Y-%m-%dT%H:%M')
        la0, lo0, la1, lo1 = _SMK_BBOX
        offs = []
        for hsp in latest_data['firms'].get('hotspots', []):
            dt = _parse_ts(hsp.get('timestamp'))
            if not dt:
                continue
            h0 = int((dt - t0).total_seconds() // 3600)
            if not (0 <= h0 < n_past):
                continue
            rr = round((la1 - hsp['lat']) / (la1 - la0) * (_SMK_NR - 1))
            cc = round((hsp['lon'] - lo0) / (lo1 - lo0) * (_SMK_NC - 1))
            if 0 <= rr < _SMK_NR and 0 <= cc < _SMK_NC:
                offs.append((rr, cc, h0))
        if len(offs) < 20:
            return None
        k = calibrate_smoke_k()
        pred = hindcast_smoke(offs, _wind_field, n_past,
                              set(range(n_past)), k_cal=k)
        return _sparse_smoke(pred, n_past, times_af,
                             [la0, lo0, la1, lo1], (_SMK_NR, _SMK_NC))
    except Exception as e:
        print(f"smoke past gironde err: {e}")
        return None


def _smoke_past_zone(z):
    """Meme reanalyse pour une zone (France incluse) sur sa grille fumee."""
    try:
        from src.fire_front import hindcast_smoke, _SMK_NR, _SMK_NC
        af = z.get('air_field')
        wf = z.get('wind_field')
        hs = ((z.get('latest') or {}).get('firms') or {}).get('hotspots') or []
        if not af or not wf or len(hs) < 20:
            return None
        times_af = af['times']
        now_key = datetime.utcnow().strftime('%Y-%m-%dT%H:00')
        n_past = sum(1 for t in times_af if t < now_key)
        if n_past < 10:
            return None
        t0 = datetime.strptime(times_af[0], '%Y-%m-%dT%H:%M')
        lon0, lat0, lon1, lat1 = z['sim_bbox']
        la0, lo0, la1, lo1 = lat0, lon0, lat1, lon1
        offs = []
        for hsp in hs:
            dt = _parse_ts(hsp.get('timestamp'))
            if not dt:
                continue
            h0 = int((dt - t0).total_seconds() // 3600)
            if not (0 <= h0 < n_past):
                continue
            rr = round((la1 - hsp['lat']) / (la1 - la0) * (_SMK_NR - 1))
            cc = round((hsp['lon'] - lo0) / (lo1 - lo0) * (_SMK_NC - 1))
            if 0 <= rr < _SMK_NR and 0 <= cc < _SMK_NC:
                offs.append((rr, cc, h0))
        if len(offs) < 20:
            return None
        k = calibrate_smoke_k()
        pred = hindcast_smoke(offs, wf, n_past, set(range(n_past)),
                              k_cal=k, smk_bbox=(la0, lo0, la1, lo1))
        return _sparse_smoke(pred, n_past, times_af,
                             [la0, lo0, la1, lo1], (_SMK_NR, _SMK_NC))
    except Exception as e:
        print(f"smoke past zone err: {e}")
        return None


def calibrate_smoke_k():
    """Auto-calibration of smoke emissions against REALITY.

    Replays the real fire's smoke (FIRMS detections + past winds) over the
    last days, compares the predicted PM2.5 with the CAMS-observed PM2.5
    anomaly at Bordeaux, and returns a global emission scaling factor.
    """
    try:
        from src.fire_front import hindcast_smoke, _SMK_BBOX, _SMK_NR, _SMK_NC
        af = _cached('airfield', 1800, fetch_air_field)
        if not af or not latest_data or _wind_field is None:
            saved_k = _warm_load('smoke_k.json')
            return float(saved_k['k']) if saved_k else 1.0
        times_af = af['times']
        now_key = datetime.utcnow().strftime('%Y-%m-%dT%H:00')
        n_past = sum(1 for t in times_af if t < now_key)
        if n_past < 30:
            return 1.0
        t0 = datetime.strptime(times_af[0], '%Y-%m-%dT%H:%M')
        la0, lo0, la1, lo1 = _SMK_BBOX
        offs = []
        for hsp in latest_data['firms'].get('hotspots', []):
            dt = _parse_ts(hsp.get('timestamp'))
            if not dt:
                continue
            h0 = int((dt - t0).total_seconds() // 3600)
            if not (0 <= h0 < n_past):
                continue
            rr = round((la1 - hsp['lat']) / (la1 - la0) * (_SMK_NR - 1))
            cc = round((hsp['lon'] - lo0) / (lo1 - lo0) * (_SMK_NC - 1))
            if 0 <= rr < _SMK_NR and 0 <= cc < _SMK_NC:
                offs.append((rr, cc, h0))
        if len(offs) < 50:
            return 1.0
        pred = hindcast_smoke(offs, _wind_field, n_past, set(range(n_past)))
        # Bordeaux dans les deux grilles
        r_b = round((la1 - 44.84) / (la1 - la0) * (_SMK_NR - 1))
        c_b = round((-0.58 - lo0) / (lo1 - lo0) * (_SMK_NC - 1))
        pm = np.asarray(af['pm25'])
        n_lat, n_lon = pm.shape[1], pm.shape[2]
        bi = round((44.84 - la0) / (la1 - la0) * (n_lat - 1))
        bj = round((-0.58 - lo0) / (lo1 - lo0) * (n_lon - 1))
        ratios = []
        for h in range(n_past):
            p = pred.get(h)
            if p is None:
                continue
            p_val = p[r_b][c_b]
            if p_val < 8:
                continue   # heures sans panache prédit : non informatives
            bg = float(np.percentile(pm[h], 10))     # fond régional
            anom = max(float(pm[h, bi, bj]) - bg, 0.0)
            ratios.append(anom / p_val)
        if len(ratios) < 6:
            return 1.0
        k = float(np.clip(np.median(ratios), 0.25, 4.0))
        _warm_save('smoke_k.json', {'k': float(k)})
        print(f"✓ Calibration fumée vs CAMS : k={k:.2f} ({len(ratios)} heures)")
        return k
    except Exception as e:
        print(f"calibration fumée erreur: {e}")
        return 1.0


@app.route('/api/air-grid')
def api_air_grid():
    """Spatial air-quality grid (Open-Meteo) for a coloured-zone overlay."""
    def prod():
        n = 7
        lat0, lat1, lon0, lon1 = 44.55, 45.10, -1.32, -0.46
        lats = np.linspace(lat0, lat1, n)
        lons = np.linspace(lon0, lon1, n)
        laq, loq = [], []
        for la in lats:
            for lo in lons:
                laq.append(round(float(la), 3))
                loq.append(round(float(lo), 3))
        r = requests.get('https://air-quality-api.open-meteo.com/v1/air-quality', params={
            'latitude': ','.join(map(str, laq)), 'longitude': ','.join(map(str, loq)),
            'current': 'pm2_5,european_aqi', 'timezone': 'UTC'}, timeout=20)
        if r.status_code != 200:
            return None
        res = r.json()
        if isinstance(res, dict):
            res = [res]
        pts = []
        for idx, x in enumerate(res):
            c = x.get('current', {})
            pts.append({'lat': laq[idx], 'lon': loq[idx],
                        'aqi': c.get('european_aqi'), 'pm25': c.get('pm2_5')})
        return {'points': pts,
                'dlat': (lat1 - lat0) / (n - 1), 'dlon': (lon1 - lon0) / (n - 1)}
    return jsonify(_cached('airgrid', 1800, prod) or {'points': []})


@app.route('/robots.txt')
def robots():
    host = (request.host or 'incendiebordeaux.fr').split(':')[0]
    return Response(f"User-agent: *\nAllow: /\nSitemap: https://{host}/sitemap.xml\n",
                    mimetype='text/plain')


@app.route('/sitemap.xml')
def sitemap():
    host = (request.host or 'incendiebordeaux.fr').split(':')[0]
    xml = ('<?xml version="1.0" encoding="UTF-8"?>'
           '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
           f'<url><loc>https://{host}/</loc><changefreq>hourly</changefreq></url>'
           f'<url><loc>https://{host}/methodologie</loc><changefreq>monthly</changefreq></url>'
           '</urlset>')
    return Response(xml, mimetype='application/xml')


@app.route('/methodologie')
def methodologie():
    return render_template('methodologie.html')


@app.route('/api/communes')
def api_communes():
    """Simplified commune boundaries (geo.api.gouv.fr) within the view."""
    z = _zone_from_req()
    if z is not None and z.get('id') == 'france':
        def prod_fr():
            # geo.api.gouv.fr ne fournit pas les contours des departements en
            # geojson -> source france-geojson (contours simplifies, statiques)
            r = requests.get('https://raw.githubusercontent.com/gregoiredavid/'
                             'france-geojson/master/'
                             'departements-version-simplifiee.geojson',
                             timeout=60)
            if r.status_code != 200:
                return None
            feats = []
            for f in r.json().get('features', []):
                geom = f.get('geometry') or {}
                polys = geom.get('coordinates') or []
                if geom.get('type') == 'Polygon':
                    polys = [polys]
                outp = []
                for poly in polys:
                    rings = []
                    for ring in poly[:1]:
                        dec = [[round(x, 3), round(y, 3)]
                               for i, (x, y) in enumerate(ring) if i % 4 == 0]
                        if len(dec) >= 4:
                            if dec[0] != dec[-1]:
                                dec.append(dec[0])
                            rings.append(dec)
                    if rings:
                        outp.append(rings)
                if outp:
                    feats.append({'type': 'Feature',
                                  'properties': {'nom': (f.get('properties') or {}).get('nom', '')},
                                  'geometry': {'type': 'MultiPolygon', 'coordinates': outp}})
            return {'type': 'FeatureCollection', 'features': feats}
        data = _cached('departements', 30 * 86400, prod_fr)
        return jsonify(data or {'type': 'FeatureCollection', 'features': []})
    if z is not None:
        dep = z.get('dep')
        if dep is None:
            try:
                r0 = requests.get('https://geo.api.gouv.fr/communes',
                                  params={'lat': z['lat'], 'lon': z['lon'],
                                          'fields': 'codeDepartement'}, timeout=10)
                dep = (r0.json() or [{}])[0].get('codeDepartement', '33')
            except Exception:
                dep = '33'
            z['dep'] = dep
        dep_code, zb = dep, z['sim_bbox']
        bbox_filter = (zb[1] - 0.1, zb[0] - 0.1, zb[3] + 0.1, zb[2] + 0.1)
        cache_key = f'communes_{dep_code}'
    else:
        dep_code, bbox_filter, cache_key = '33', (44.1, -1.9, 45.7, -0.1), 'communes'

    def prod():
        r = requests.get('https://geo.api.gouv.fr/communes',
                         params={'codeDepartement': dep_code, 'format': 'geojson',
                                 'geometry': 'contour'}, timeout=60)
        if r.status_code != 200:
            return None
        feats = []
        for f in r.json().get('features', []):
            geom = f.get('geometry') or {}
            name = (f.get('properties') or {}).get('nom', '')
            polys = (geom.get('coordinates') or [])
            if geom.get('type') == 'Polygon':
                polys = [polys]
            out_polys = []
            keep = False
            for poly in polys:
                rings = []
                for ring in poly[:1]:            # outer ring only
                    dec = [[round(x, 4), round(y, 4)]
                           for i, (x, y) in enumerate(ring) if i % 3 == 0]
                    if len(dec) < 4:
                        continue
                    if dec[0] != dec[-1]:
                        dec.append(dec[0])
                    rings.append(dec)
                    la0f, lo0f, la1f, lo1f = bbox_filter
                    if any(la0f <= y <= la1f and lo0f <= x <= lo1f for x, y in dec[::10]):
                        keep = True
                if rings:
                    out_polys.append(rings)
            if keep and out_polys:
                feats.append({'type': 'Feature',
                              'properties': {'nom': name},
                              'geometry': {'type': 'MultiPolygon',
                                           'coordinates': out_polys}})
        return {'type': 'FeatureCollection', 'features': feats}
    data = _cached(cache_key, 7 * 86400, prod)
    if data is None:
        return jsonify({'type': 'FeatureCollection', 'features': []})
    return jsonify(data)


@app.route('/api/aircraft-history')
def api_aircraft_history():
    """Recorded water-bomber positions (built by the host logger) for replay."""
    path = os.getenv('AIRCRAFT_LOG', '/data/aircraft.jsonl')
    pos = []
    try:
        with open(path) as f:
            for ln in f:
                try:
                    pos.append(json.loads(ln))
                except ValueError:
                    pass
    except FileNotFoundError:
        pass
    return jsonify({'positions': pos})


# Boot from the warm disk cache: serve slightly-stale data immediately after a
# restart instead of showing a loading screen for the recompute duration.
_w = _warm_load('latest.json')
if _w and _w.get('latest'):
    latest_data = _w['latest']
    try:
        last_update = datetime.fromisoformat(_w['last_update'])
    except (KeyError, ValueError):
        last_update = datetime.utcnow()
    print("✓ Warm cache: données servies depuis le disque")
for _l in _LUTTE:
    _w = _warm_load(f'views_{_l}.json')
    if _w and _w.get('views'):
        _ens_store[_l]['views'] = _w['views']
        _ens_store[_l]['ver'] = _w.get('ver')
        print(f"✓ Warm cache: ensemble lutte={_l} servi depuis le disque")
del _w

def _france_boot():
    try:
        for _lv in _LUTTE:
            _w2 = _warm_load(f'france_views_{_lv}.json')
            if _w2 and _w2.get('views'):
                ZONES.setdefault('france', {'id': 'france', 'ready': False, 'ens': {}})
                ZONES['france']['ens'][_lv] = {'ver': _w2.get('ver'),
                                               'hr_ver': _w2.get('hr_ver'),
                                               'views': _w2['views'],
                                               'views_hr': _w2.get('views_hr')}
                print(f"✓ Warm: ensemble national lutte={_lv}")
        saved = _warm_load('geo_names.json')
        if saved:
            _geo_names.update({tuple(map(float, k.split('|'))): v
                               for k, v in saved.items()})
        fr0 = _warm_load('france_fires.json')
        if fr0 and fr0.get('hotspots'):
            # /api/fires servi immediatement aussi (age force a ~55 min :
            # le fetch frais qui suit le remplacera)
            _layer_cache['france_fires'] = (time.time() - 3300, fr0)
            _refresh_france_zone(fr0)      # zone prete en ~5 s (etat precedent)
            print("✓ Warm: zone France servie depuis le disque")
        fr = fetch_france_fires()
        _warm_save('geo_names.json',
                   {f'{k[0]}|{k[1]}': v for k, v in _geo_names.items()})
        if fr.get('hotspots'):
            _warm_save('france_fires.json', fr)
        _layer_cache['france_fires'] = (time.time(), fr)
        _refresh_france_zone(fr)
    except Exception as e:
        print(f"france boot error: {e}")


def _merge_hires_views(nat_views, hires):
    """Remplace, dans chaque frame nationale, les points grossiers situés dans
    l'emprise d'un feu re-simulé à 500 m par les points fins de ce feu.
    Stats de surface et fumée restent celles de l'ensemble NATIONAL."""
    out = {}
    boxes = [bb for bb, _ in hires]

    def _inside(p):
        return any(bb[1] <= p[0] <= bb[3] and bb[0] <= p[1] <= bb[2]
                   for bb in boxes)
    for vk, nat in nat_views.items():
        if not nat or not nat.get('frames'):
            out[vk] = nat
            continue
        zvs = [(bb, zv.get(vk)) for bb, zv in hires if zv and zv.get(vk)]
        if not zvs:
            out[vk] = nat
            continue
        d = dict(nat)
        frames = []
        # alignement par TIMESTAMP : les ensembles zone/gironde ne partent pas
        # exactement a la meme heure que le national
        z_by_ts = [({(zf.get('timestamp') or '')[:13]: zf
                     for zf in (zv.get('frames') or [])}, bb, zv)
                   for bb, zv in zvs]
        for i, f in enumerate(nat['frames']):
            f2 = dict(f)
            pts = [p for p in (f.get('new_points') or []) if not _inside(p)]
            key_ts = (f.get('timestamp') or '')[:13]
            for zmap, bb, zv in z_by_ts:
                zf = zmap.get(key_ts)
                if zf is None and i < len(zv.get('frames') or []):
                    zf = zv['frames'][i]          # secours : index
                if zf is not None:
                    pts.extend(zf.get('new_points') or [])
            f2['new_points'] = pts
            frames.append(f2)
        d['frames'] = frames
        d['hires_n'] = len(zvs)
        d['hires_boxes'] = [list(bb) for bb, _ in zvs]
        out[vk] = d
    return out


def _score_zone(z):
    """Score de fiabilite du modele pour CE feu : backtest 24 h — on rejoue
    hier avec la vraie meteo et les params calibres, on compare au reel."""
    try:
        from src.backtest import _window_loss
        wf = z.get('wind_field')
        hs = ((z.get('latest') or {}).get('firms') or {}).get('hotspots') or []
        if not wf or len(hs) < 15 or z.get('veg') is None:
            return
        now_key = datetime.utcnow().strftime('%Y-%m-%dT%H:00')
        now_idx = next((i for i, t in enumerate(wf['times'])
                        if t >= now_key), len(wf['times']) - 1)
        params = {**_sim_params()}
        r = _window_loss(hs, wf, z['veg'], z['sim_bbox'],
                         max(now_idx - 24, 1), 24, params, n_runs=1)
        if r:
            # score lisible : 100 si surface juste ET bien placee ;
            # ~50 si surface x2 ; ~25 si surface x5
            ratio_pen = np.exp(-0.7 * abs(np.log(max(r['a_sim'], 1)
                                                 / max(r['a_obs'], 1))))
            score = int(np.clip(100 * ratio_pen * (0.45 + 0.55 * r['recall']),
                                3, 97))
            z['sim_score'] = {'score': score, 'a_sim': r['a_sim'],
                              'a_obs': r['a_obs'], 'recall': r['recall'],
                              'date': now_key}
            print(f"✓ Score simulation {z['id']} : {score}/100")
    except Exception as e:
        print(f"score zone {z.get('id')} err: {e}")


def _france_ens_loop():
    """Recalcule les ensembles nationaux en continu (ver change chaque heure) ;
    sert toujours la derniere version terminee pendant le calcul."""
    time.sleep(45)
    while True:
        try:
            # BACKTEST quotidien : calibre ros_cal/supp_base/cont_base sur les
            # 96 dernieres heures REELLES (fenetre croissance + fenetre declin)
            saved_bt = _warm_load('sim_params.json') or {}
            today = datetime.utcnow().strftime('%Y-%m-%d')
            if (saved_bt.get('date') != today
                    and (not saved_bt or datetime.utcnow().hour >= 3)
                    and latest_data and _wind_field is not None):
                try:
                    from src.backtest import backtest_calibrate
                    fuel_bt = _veg_cache.get('fuel')
                    hs_bt = (latest_data.get('firms') or {}).get('hotspots') or []
                    if fuel_bt is not None and len(hs_bt) > 50:
                        print("⚙ Backtest de calibration…")
                        best = backtest_calibrate(hs_bt, _wind_field,
                                                  fuel_bt, SIM_BBOX)
                        if best:
                            _warm_save('sim_params.json',
                                       {'params': best['params'],
                                        'loss': best['loss'],
                                        'windows': best['windows'],
                                        'date': today})
                            print(f"✓ Params calibrés : {best['params']} "
                                  f"(loss {best['loss']:.3f})")
                except Exception as ebt:
                    print(f"backtest err: {ebt}")
            # score de fiabilite de la fiche Gironde (backtest 24 h quotidien)
            try:
                if latest_data and _wind_field is not None \
                        and _veg_cache.get('fuel') is not None:
                    zg = {'id': 'gironde', 'wind_field': _wind_field,
                          'latest': latest_data,
                          'veg': _veg_cache['fuel'], 'sim_bbox': SIM_BBOX}
                    _score_zone(zg)
                    if zg.get('sim_score'):
                        _warm_save('sim_score_gironde.json', zg['sim_score'])
            except Exception as esg:
                print(f"score gironde err: {esg}")
            if ZONES.get('france', {}).get('ready'):
                tops = [c for c in ((_cached('france_fires', 3600,
                                              fetch_france_fires)
                                     or {}).get('clusters') or [])
                        if c.get('active') and c.get('n_total', 0) >= 5
                        ][:int(os.getenv('FR_HIRES', '30'))]
                for c in tops:
                    _get_zone(c['zone'])       # lance les refresh de zone
                for lv in ('med', 'low', 'high'):
                    compute_zone_ensemble('france', lv, wait=True)
                    # re-simulation 500 m des principaux feux + fusion
                    hires = []
                    for c in tops:
                        # le feu girondin utilise l'ensemble OFFICIEL de la
                        # fiche incendiebordeaux.fr (16 tirages, fumee
                        # calibree) — pas une re-simulation de zone
                        if 44.0 <= c['lat'] <= 45.4 and -1.6 <= c['lon'] <= -0.2:
                            gv = _ens_store.get(lv, {}).get('views') or {}
                            if gv:
                                hires.append((list(SIM_BBOX), gv))
                            continue
                        zz = ZONES.get(c['zone'])
                        for _ in range(45):
                            if zz and zz.get('ready'):
                                break
                            time.sleep(2)
                            zz = ZONES.get(c['zone'])
                        if not (zz and zz.get('ready')):
                            continue
                        try:
                            zv = compute_zone_ensemble(c['zone'], lv, wait=True)
                            if zv:
                                hires.append((zz['sim_bbox'], zv))
                            if lv == 'med':
                                _score_zone(zz)
                        except Exception as e3:
                            print(f"hires {c['zone']}/{lv} err: {e3}")
                    ent_fr = ZONES['france']['ens'].get(lv)
                    if ent_fr and ent_fr.get('views'):
                        if hires:
                            ent_fr['views_hr'] = _merge_hires_views(
                                ent_fr['views'], hires)
                            ent_fr['hr_ver'] = ent_fr['ver']
                            print(f"✓ Fusion hi-res {lv} : {len(hires)} feux à 500 m")
                        _warm_save(f'france_views_{lv}.json',
                                   {'ver': ent_fr['ver'],
                                    'hr_ver': ent_fr.get('hr_ver'),
                                    'views': ent_fr['views'],
                                    'views_hr': ent_fr.get('views_hr')})
                # zones ouvertes par les utilisateurs (fiches) : leur ensemble
                # med est calcule ici, en fond — jamais dans une requete
                top_ids = {c['zone'] for c in tops}
                extra = [zid for zid, zz in list(ZONES.items())
                         if zid not in ('france', 'gironde')
                         and zid not in top_ids and zz.get('ready')
                         and not (zz.get('ens') or {}).get('med')][:6]
                for zid in extra:
                    try:
                        compute_zone_ensemble(zid, 'med', wait=True)
                    except Exception as e4:
                        print(f"zone visitee {zid} err: {e4}")
        except Exception as e:
            print(f"france ens loop err: {e}")
        time.sleep(300)


Thread(target=_france_boot, daemon=True).start()
Thread(target=_france_ens_loop, daemon=True).start()

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
