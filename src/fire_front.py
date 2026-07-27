"""
Spatial multi-source wildfire propagation for the live map.

The fire is seeded from every real FIRMS hotspot and grown hour by hour on a
raster grid. Unlike a single-wind model, each cell uses its *local* conditions:

  * wind      — interpolated from a grid of Open-Meteo forecasts (spatial field)
  * humidity  — same spatial field (damp air slows the fire)
  * vegetation— NDVI-derived fuel map (forest burns fast, water/urban block it)

Each hour, burning cells push the fire into their downwind neighbours at the
local head-fire rate of spread. Fronts from neighbouring hotspots merge; the
burn cannot cross water (fuel = 0). For each hour we emit the burned-area
isochrone as lat/lon polygons.

Rate of spread: an empirical power law in wind speed (Cruz & Alexander 2010
family) calibrated to observed Landes head-fire rates (0.3-1.5 km/h), modulated
by humidity and local fuel. A deliberate, documented calibration — not the raw
(far too weak) Rothermel surface model bundled in the repo.
"""

import numpy as np

try:
    from scipy.ndimage import binary_dilation  # noqa: F401 (kept for parity)
    from scipy.interpolate import RegularGridInterpolator
except Exception:  # pragma: no cover
    RegularGridInterpolator = None

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_CAL = 0.25       # m/min per (m/s)^EXP
_EXP = 1.3
_ROS_CAP = 25.0   # m/min (~1.5 km/h)
_M_PER_DEG_LAT = 111_320.0

# 8-neighbour offsets: (di=row=north, dj=col=east)
_NEIGHBOURS = [(-1, 0), (1, 0), (0, -1), (0, 1),
               (-1, -1), (-1, 1), (1, -1), (1, 1)]


def _ros_grid(speed, rh, fuel, temp=None):
    """Per-cell head-fire ROS [m/min]: wind law * dryness(humidity,temp) * fuel.

    Hotter air pre-heats and dries the fuel, so the fire spreads faster: a
    simple linear temperature factor around 25 °C (≈ +4 %/°C above, capped
    0.6–1.6) modulates the rate, on top of the humidity dryness term.
    """
    base = _CAL * np.power(np.clip(speed, 0, None), _EXP)
    dryness = np.clip(1.3 - rh / 55.0, 0.08, 1.3)
    if temp is not None:
        temp_factor = np.clip(1.0 + 0.04 * (temp - 25.0), 0.6, 1.6)
        dryness = dryness * temp_factor
    return np.clip(base * dryness * fuel, 0.0, _ROS_CAP)


def _shift(mask, di, dj):
    """Shift a boolean array by (di,dj), filling exposed edges with False."""
    out = np.zeros_like(mask)
    r_src = slice(max(0, -di), mask.shape[0] - max(0, di))
    r_dst = slice(max(0, di), mask.shape[0] - max(0, -di))
    c_src = slice(max(0, -dj), mask.shape[1] - max(0, dj))
    c_dst = slice(max(0, dj), mask.shape[1] - max(0, -dj))
    out[r_dst, c_dst] = mask[r_src, c_src]
    return out


def simulate_fire_front(hotspots, wind_records, wind_field=None,
                        veg_fuel=None, veg_bbox=None, max_hours=48, emit_every=1):
    seeds = [(h['lat'], h['lon']) for h in hotspots if 'lat' in h and 'lon' in h]
    if not seeds or RegularGridInterpolator is None:
        return {'n_frames': 0, 'frames': [], 'n_seeds': len(seeds)}

    lats = np.array([s[0] for s in seeds])
    lons = np.array([s[1] for s in seeds])

    # Bigger margin for longer runs so the front doesn't hit the grid edge.
    margin = min(0.18 + max_hours * 0.0026, 0.65)
    lat_min, lat_max = lats.min() - margin, lats.max() + margin
    lon_min, lon_max = lons.min() - margin, lons.max() + margin

    cell_deg = 0.0025
    lat_axis = np.arange(lat_min, lat_max, cell_deg)
    lon_axis = np.arange(lon_min, lon_max, cell_deg)
    if len(lat_axis) * len(lon_axis) > 450_000:
        cell_deg = 0.004
        lat_axis = np.arange(lat_min, lat_max, cell_deg)
        lon_axis = np.arange(lon_min, lon_max, cell_deg)
    nrows, ncols = len(lat_axis), len(lon_axis)
    if nrows < 3 or ncols < 3:
        return {'n_frames': 0, 'frames': [], 'n_seeds': len(seeds)}

    mid_lat = (lat_min + lat_max) / 2.0
    cell_lat_m = cell_deg * _M_PER_DEG_LAT
    cell_lon_m = cell_deg * _M_PER_DEG_LAT * np.cos(np.radians(mid_lat))
    cell_area_ha = (cell_lat_m * cell_lon_m) / 10_000.0

    LON, LAT = np.meshgrid(lon_axis, lat_axis)
    pts = np.column_stack([LAT.ravel(), LON.ravel()])

    # unit neighbour vectors in (east, north)
    nbr = []
    for di, dj in _NEIGHBOURS:
        e, n = dj * cell_lon_m, di * cell_lat_m
        d = np.hypot(e, n)
        nbr.append((di, dj, e / d, n / d))

    # ---- fuel grid (vegetation) --------------------------------------------
    if veg_fuel is not None and veg_bbox is not None:
        vlon0, vlat0, vlon1, vlat1 = veg_bbox
        vh, vw = veg_fuel.shape
        # image row 0 = north (lat max)
        col = np.clip(((LON - vlon0) / (vlon1 - vlon0) * vw).astype(int), 0, vw - 1)
        row = np.clip(((vlat1 - LAT) / (vlat1 - vlat0) * vh).astype(int), 0, vh - 1)
        fuel_grid = veg_fuel[row, col]
    else:
        fuel_grid = np.full((nrows, ncols), 0.7)

    # ---- wind field interpolators ------------------------------------------
    def _interp_hour(field2d):
        f = RegularGridInterpolator(
            (np.array(wind_field['grid_lats']), np.array(wind_field['grid_lons'])),
            field2d, bounds_error=False, fill_value=None)
        return f(pts).reshape(nrows, ncols)

    have_field = (wind_field is not None
                  and RegularGridInterpolator is not None
                  and len(wind_field.get('times', [])) > 0)

    if have_field:
        times = wind_field['times'][:max_hours]
        speed_hrs, dir_hrs, rh_hrs = (np.asarray(wind_field['speed']),
                                      np.asarray(wind_field['dir']),
                                      np.asarray(wind_field['rh']))
        temp_hrs = np.asarray(wind_field['temp']) if wind_field.get('temp') is not None else None
        n_hours = min(max_hours, len(times))
    else:
        # fall back to a single-point series broadcast over the domain
        recs = (wind_records or [])[:max_hours]
        times = [r.get('timestamp') for r in recs]
        n_hours = len(recs)

    burned = np.zeros((nrows, ncols), dtype=bool)
    ri = np.clip(((lats - lat_min) / cell_deg).astype(int), 0, nrows - 1)
    ci = np.clip(((lons - lon_min) / cell_deg).astype(int), 0, ncols - 1)
    burned[ri, ci] = True

    min_poly_ha = cell_area_ha * 6

    def _seg_area_ha(seg):
        lo, la = seg[:, 0], seg[:, 1]
        x = (lo - lon_min) * cell_lon_m / cell_deg
        y = (la - lat_min) * cell_lat_m / cell_deg
        return abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))) / 2 / 10_000.0

    def _isochrone(mask):
        if not mask.any():
            return []
        try:
            cs = plt.contour(LON, LAT, mask.astype(float), levels=[0.5])
            polys = []
            for seg in cs.allsegs[0]:
                if len(seg) < 4 or _seg_area_ha(seg) < min_poly_ha:
                    continue
                step = max(1, len(seg) // 60)
                polys.append([[round(float(la), 5), round(float(lo), 5)]
                              for lo, la in seg[::step]])
            plt.close('all')
            return polys
        except Exception:
            plt.close('all')
            return []

    residual = np.zeros((nrows, ncols))
    frames = []
    for h in range(n_hours):
        if have_field:
            speed = _interp_hour(speed_hrs[h])
            wdir = None
            pe = _interp_hour(-np.sin(np.radians(dir_hrs[h])))  # propagation east
            pn = _interp_hour(-np.cos(np.radians(dir_hrs[h])))  # propagation north
            rh = _interp_hour(rh_hrs[h])
            temp = _interp_hour(temp_hrs[h]) if temp_hrs is not None else None
            ts = times[h]
        else:
            rec = wind_records[h]
            spd = float(rec.get('wind_speed_10m_ms') or 0.0)
            wf = float(rec.get('wind_direction_10m_deg') or 270.0)
            speed = np.full((nrows, ncols), spd)
            pe = np.full((nrows, ncols), -np.sin(np.radians(wf)))
            pn = np.full((nrows, ncols), -np.cos(np.radians(wf)))
            rh = np.full((nrows, ncols), float(rec.get('relative_humidity_pct') or 50.0))
            t = rec.get('temperature_c')
            temp = np.full((nrows, ncols), float(t)) if t is not None else None
            ts = rec.get('timestamp')

        # normalise propagation direction
        pmag = np.hypot(pe, pn) + 1e-9
        pe_u, pn_u = pe / pmag, pn / pmag

        ros = _ros_grid(speed, rh, fuel_grid, temp)
        residual += ros * 60.0

        # directed sub-cell propagation, respecting local wind + fuel
        for _ in range(8):
            ready = residual >= cell_lat_m
            if not ready.any():
                break
            grew = np.zeros_like(burned)
            for di, dj, ex, ny in nbr:
                push = (pe_u * ex + pn_u * ny) >= 0.25    # wind pushes toward nbr
                src = burned & ready & push
                if not src.any():
                    continue
                tgt = _shift(src, di, dj) & ~burned & (fuel_grid > 0)
                grew |= tgt
            burned |= grew
            residual = np.where(ready, residual - cell_lat_m, residual)
            if not grew.any():
                break

        # Emit a frame every `emit_every` hours (plus the very last one) to
        # keep the payload small on long (7-day) runs.
        if (h % emit_every != 0) and (h != n_hours - 1):
            continue

        burning_mask = burned
        mean_speed = float(speed[burning_mask].mean()) if burning_mask.any() else 0.0
        mean_rh = float(rh[burning_mask].mean()) if burning_mask.any() else 0.0
        mean_temp = (float(temp[burning_mask].mean())
                     if (temp is not None and burning_mask.any()) else None)
        frames.append({
            'hour': h,
            'timestamp': ts,
            'wind_speed_ms': round(mean_speed, 1),
            'humidity_pct': round(mean_rh),
            'temp_c': round(mean_temp) if mean_temp is not None else None,
            'area_ha': round(float(burned.sum() * cell_area_ha)),
            'polygons': _isochrone(burned),
        })

    return {'n_frames': len(frames), 'n_seeds': len(seeds), 'frames': frames}
