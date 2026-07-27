"""
Monte-Carlo wildfire propagation for the live map.

The fire is seeded from the *currently active* FIRMS hotspots and grown hour by
hour on a raster grid, driven by local wind / humidity / temperature
(interpolated forecast fields) and NDVI vegetation fuel (water blocks fire).

Because a single deterministic run overstates confidence, we run an ENSEMBLE:
`n_runs` simulations with perturbed inputs (wind direction & speed, humidity,
temperature, spread-rate calibration), then report, for each future timestamp,
the cells burned in >= 50 % of runs (the ensemble "mean" footprint) plus the
p10–p90 range of burned area.

Rate of spread: empirical power law in wind speed (Cruz & Alexander 2010
family) calibrated on observed Landes head-fire rates (0.3–1.5 km/h), modulated
by humidity dryness, temperature and local fuel. A deliberate, documented
calibration — not the (far too weak) Rothermel toy model bundled in the repo.
"""

import numpy as np

try:
    from scipy.ndimage import binary_dilation
    from scipy.interpolate import RegularGridInterpolator
except Exception:  # pragma: no cover
    binary_dilation = None
    RegularGridInterpolator = None

_CAL = 0.25       # m/min per (m/s)^EXP
_EXP = 1.3
_ROS_CAP = 25.0   # m/min (~1.5 km/h)
_M_PER_DEG_LAT = 111_320.0

_NEIGHBOURS = [(-1, 0), (1, 0), (0, -1), (0, 1),
               (-1, -1), (-1, 1), (1, -1), (1, 1)]


def _ros_grid(speed, rh, fuel, temp=None, cal_mult=1.0, soil=None):
    """Per-cell head-fire ROS [m/min]: wind law * dryness(RH,T,soil) * fuel."""
    base = _CAL * cal_mult * np.power(np.clip(speed, 0, None), _EXP)
    dryness = np.clip(1.3 - rh / 55.0, 0.08, 1.3)
    if temp is not None:
        dryness = dryness * np.clip(1.0 + 0.04 * (temp - 25.0), 0.6, 1.6)
    if soil is not None:
        # soil moisture 0-7 cm [m³/m³] as drought proxy: parched soil (<0.10)
        # accelerates spread, wet soil (>0.30) strongly damps it.
        dryness = dryness * np.clip(1.45 - 3.5 * soil, 0.45, 1.35)
    return np.clip(base * dryness * fuel, 0.0, _ROS_CAP)


def _shift(mask, di, dj):
    out = np.zeros_like(mask)
    r_src = slice(max(0, -di), mask.shape[0] - max(0, di))
    r_dst = slice(max(0, di), mask.shape[0] - max(0, -di))
    c_src = slice(max(0, -dj), mask.shape[1] - max(0, dj))
    c_dst = slice(max(0, dj), mask.shape[1] - max(0, -dj))
    out[r_dst, c_dst] = mask[r_src, c_src]
    return out


def _prepare_domain(hotspots, veg_fuel, veg_bbox, cell_deg=0.005):
    seeds = [(h['lat'], h['lon']) for h in hotspots if 'lat' in h and 'lon' in h]
    if not seeds:
        return None
    lats = np.array([s[0] for s in seeds])
    lons = np.array([s[1] for s in seeds])
    margin = 0.60
    lat_min, lat_max = lats.min() - margin, lats.max() + margin
    lon_min, lon_max = lons.min() - margin, lons.max() + margin
    lat_axis = np.arange(lat_min, lat_max, cell_deg)
    lon_axis = np.arange(lon_min, lon_max, cell_deg)
    nrows, ncols = len(lat_axis), len(lon_axis)
    if nrows < 3 or ncols < 3:
        return None

    mid_lat = (lat_min + lat_max) / 2.0
    cell_lat_m = cell_deg * _M_PER_DEG_LAT
    cell_lon_m = cell_deg * _M_PER_DEG_LAT * np.cos(np.radians(mid_lat))

    LON, LAT = np.meshgrid(lon_axis, lat_axis)
    if veg_fuel is not None and veg_bbox is not None:
        vlon0, vlat0, vlon1, vlat1 = veg_bbox
        vh, vw = veg_fuel.shape
        col = np.clip(((LON - vlon0) / (vlon1 - vlon0) * vw).astype(int), 0, vw - 1)
        row = np.clip(((vlat1 - LAT) / (vlat1 - vlat0) * vh).astype(int), 0, vh - 1)
        fuel_grid = veg_fuel[row, col]
    else:
        fuel_grid = np.full((nrows, ncols), 0.7)

    seed_mask = np.zeros((nrows, ncols), dtype=bool)
    ri = np.clip(((lats - lat_min) / cell_deg).astype(int), 0, nrows - 1)
    ci = np.clip(((lons - lon_min) / cell_deg).astype(int), 0, ncols - 1)
    seed_mask[ri, ci] = True

    nbr = []
    for di, dj in _NEIGHBOURS:
        e, n = dj * cell_lon_m, di * cell_lat_m
        d = np.hypot(e, n)
        nbr.append((di, dj, e / d, n / d))

    pts = np.column_stack([LAT.ravel(), LON.ravel()])
    return dict(lat_axis=lat_axis, lon_axis=lon_axis, nrows=nrows, ncols=ncols,
                cell_lat_m=cell_lat_m,
                cell_area_ha=(cell_lat_m * cell_lon_m) / 10_000.0,
                fuel_grid=fuel_grid, seed_mask=seed_mask, nbr=nbr, pts=pts,
                n_seeds=len(seeds))


def _run_once(dom, wind_field, wind_records, max_hours, pert=None,
              want_meta=False, suppression=False, scenario=None):
    """One propagation run; returns per-cell ignition hour (-1 = never)."""
    p = pert or {}
    sc = scenario or {}
    dir_off = p.get('dir_off', 0.0)
    speed_mult = p.get('speed_mult', 1.0) * sc.get('speed_mult_g', 1.0)
    rh_off = p.get('rh_off', 0.0)
    temp_off = p.get('temp_off', 0.0) + sc.get('temp_off_g', 0.0)
    cal_mult = p.get('cal_mult', 1.0) * sc.get('pyro_cal', 1.0)
    supp_mult = p.get('supp_mult', 1.0)
    supp_base = sc.get('supp_base', 0.75)
    soil_off = sc.get('soil_off_g', 0.0)
    dir_noise = p.get('dir_noise')          # per-hour array (wind uncertainty)
    speed_mult_h = p.get('speed_mult_h')    # per-hour array
    cal_h = p.get('cal_h')                  # per-hour array (pyroCb boost)
    spot_h = p.get('spot_h')                # per-hour spotting rate array
    spot_rate = sc.get('pyro_spot', 0.0)
    rng = p.get('rng')

    nrows, ncols = dom['nrows'], dom['ncols']
    cell_lat_m = dom['cell_lat_m']
    fuel_grid = dom['fuel_grid']

    have_field = (wind_field is not None and RegularGridInterpolator is not None
                  and len(wind_field.get('times', [])) > 0)
    if have_field:
        times = wind_field['times'][:max_hours]
        sp_h = np.asarray(wind_field['speed'])
        di_h = np.asarray(wind_field['dir'])
        rh_h = np.asarray(wind_field['rh'])
        tp_h = (np.asarray(wind_field['temp'])
                if wind_field.get('temp') is not None else None)
        so_h = (np.asarray(wind_field['soil'])
                if wind_field.get('soil') is not None else None)
        glats = np.array(wind_field['grid_lats'])
        glons = np.array(wind_field['grid_lons'])

        def interp(f2d):
            f = RegularGridInterpolator((glats, glons), f2d,
                                        bounds_error=False, fill_value=None)
            return f(dom['pts']).reshape(nrows, ncols)
    else:
        recs = (wind_records or [])[:max_hours]
        times = [r.get('timestamp') for r in recs]
    n_hours = min(max_hours, len(times))

    burned = dom['seed_mask'].copy()
    ign = np.full((nrows, ncols), -1, dtype=np.int16)
    residual = np.zeros((nrows, ncols))
    meta = [] if want_meta else None

    for h in range(n_hours):
        d_extra = dir_off + (float(dir_noise[h]) if dir_noise is not None else 0.0)
        if have_field:
            speed = interp(sp_h[h]) * speed_mult
            wdir = di_h[h] + d_extra
            pe = interp(-np.sin(np.radians(wdir)))
            pn = interp(-np.cos(np.radians(wdir)))
            rh = np.clip(interp(rh_h[h]) + rh_off, 3, 100)
            temp = interp(tp_h[h]) + temp_off if tp_h is not None else None
            soil = (np.clip(interp(so_h[h]) + soil_off, 0.02, 0.45)
                    if so_h is not None else None)
        else:
            rec = wind_records[h]
            spd = float(rec.get('wind_speed_10m_ms') or 0.0) * speed_mult
            wf = float(rec.get('wind_direction_10m_deg') or 270.0) + d_extra
            speed = np.full((nrows, ncols), spd)
            pe = np.full((nrows, ncols), -np.sin(np.radians(wf)))
            pn = np.full((nrows, ncols), -np.cos(np.radians(wf)))
            rh = np.full((nrows, ncols),
                         np.clip(float(rec.get('relative_humidity_pct') or 50.0) + rh_off, 3, 100))
            t = rec.get('temperature_c')
            temp = (np.full((nrows, ncols), float(t) + temp_off)
                    if t is not None else None)
            soil = None

        if speed_mult_h is not None:
            speed = speed * float(speed_mult_h[h])
        pmag = np.hypot(pe, pn) + 1e-9
        pe_u, pn_u = pe / pmag, pn / pmag

        cal_eff = cal_mult * (float(cal_h[h]) if cal_h is not None else 1.0)
        ros = _ros_grid(speed, rh, fuel_grid, temp, cal_eff, soil)
        if suppression:
            # Firefighting model: ground crews + air support (Canadairs, Dash,
            # helicopters). Effectiveness ramps up over ~18 h as resources
            # deploy, drops when wind is strong (uncontrollable head fire),
            # and varies between ensemble runs. Nominal peak: ~75 % ROS cut.
            ramp = min(1.0, h / 18.0)
            wind_pen = np.clip(1.15 - speed / 14.0, 0.25, 1.0)
            eff = np.clip(supp_base * supp_mult * ramp * wind_pen, 0.0, 0.95)
            ros = ros * (1.0 - eff)
        residual += ros * 60.0

        b0 = burned.copy()
        for _ in range(8):
            ready = residual >= cell_lat_m
            if not ready.any():
                break
            grew = np.zeros_like(burned)
            for di, dj, ex, ny in dom['nbr']:
                push = (pe_u * ex + pn_u * ny) >= 0.25
                src = burned & ready & push
                if not src.any():
                    continue
                grew |= _shift(src, di, dj) & ~burned & (fuel_grid > 0)
            burned |= grew
            residual = np.where(ready, residual - cell_lat_m, residual)
            if not grew.any():
                break
        ign[burned & ~b0] = h

        # PyroCumulonimbus spotting: the convective column lofts embers that
        # ignite NEW fires kilometres AHEAD of the front, downwind.
        rate_h = float(spot_h[h]) if spot_h is not None else spot_rate
        if rate_h > 0 and rng is not None:
            n_spots = rng.poisson(rate_h)
            if n_spots:
                recent = burned & (ign >= h - 3)
                if h < 3:
                    recent = recent | dom['seed_mask']
                fr_r, fr_c = np.where(recent)
                if len(fr_r):
                    for _ in range(int(n_spots)):
                        i = int(rng.integers(len(fr_r)))
                        # 2–8 km downwind, ±20° scatter
                        dist_m = float(rng.uniform(2000, 8000))
                        ang = np.radians(float(rng.normal(0, 20)))
                        ex, ny = pe_u[fr_r[i], fr_c[i]], pn_u[fr_r[i], fr_c[i]]
                        ca, sa = np.cos(ang), np.sin(ang)
                        ex2, ny2 = ex * ca - ny * sa, ex * sa + ny * ca
                        rr = fr_r[i] + int(round(ny2 * dist_m / cell_lat_m))
                        cc = fr_c[i] + int(round(ex2 * dist_m / cell_lat_m))
                        if (0 <= rr < nrows and 0 <= cc < ncols
                                and not burned[rr, cc] and fuel_grid[rr, cc] > 0.2):
                            burned[rr, cc] = True
                            ign[rr, cc] = h

        if want_meta:
            m = burned
            meta.append({
                'timestamp': times[h],
                'wind': round(float(speed[m].mean()), 1) if m.any() else 0,
                'rh': round(float(rh[m].mean())) if m.any() else 0,
                'temp': (round(float(temp[m].mean()))
                         if (temp is not None and m.any()) else None),
            })

    return ign, meta, n_hours


def simulate_monte_carlo(hotspots, wind_records, wind_field=None, veg_fuel=None,
                         veg_bbox=None, max_hours=168, emit_every=6, n_runs=16,
                         rng_seed=0, suppression=False, scenario=None):
    """Ensemble forecast: mean burned footprint per future timestamp."""
    if binary_dilation is None:
        return {'n_frames': 0, 'frames': [], 'n_seeds': 0, 'n_runs': 0}
    dom = _prepare_domain(hotspots, veg_fuel, veg_bbox)
    if dom is None:
        return {'n_frames': 0, 'frames': [], 'n_seeds': 0, 'n_runs': 0}

    pyro_std = (scenario or {}).get('pyro_dir_std', 0.0)
    rng = np.random.default_rng(rng_seed)
    igns, meta = [], None
    for k in range(n_runs):
        pert = {} if k == 0 else {
            'dir_off': float(rng.normal(0, 15)),
            'speed_mult': float(np.exp(rng.normal(0, 0.15))),
            'rh_off': float(rng.normal(0, 8)),
            'temp_off': float(rng.normal(0, 2)),
            'cal_mult': float(np.exp(rng.normal(0, 0.25))),
            'supp_mult': float(np.exp(rng.normal(0, 0.2))),
        }
        if pyro_std > 0:   # pyroCb: erratic hourly wind swings, every run
            pert['dir_noise'] = rng.normal(0, pyro_std, max_hours)
        pert['rng'] = rng   # for ember-spotting draws
        ign, m, n_hours = _run_once(dom, wind_field, wind_records, max_hours,
                                    pert, want_meta=(k == 0),
                                    suppression=suppression, scenario=scenario)
        igns.append(ign)
        if k == 0:
            meta = m
    igns = np.stack(igns)                       # (runs, rows, cols)
    seed = dom['seed_mask']

    emit_hours = sorted(set(list(range(0, n_hours, emit_every)) + [n_hours - 1]))
    frames = []
    prev_p50 = np.zeros_like(seed)
    for H in emit_hours:
        burned_runs = seed[None, :, :] | ((igns >= 0) & (igns <= H))
        prob = burned_runs.mean(axis=0)
        p50 = prob >= 0.5

        # per-run areas -> median + p10/p90 spread
        areas = burned_runs.sum(axis=(1, 2)) * dom['cell_area_ha']
        # newly "mean-burned" cells since previous emitted frame
        newly = p50 & ~prev_p50 & ~seed
        nr, nc = np.where(newly)
        new_pts = [[round(float(dom['lat_axis'][r]), 4),
                    round(float(dom['lon_axis'][c]), 4), int(H)]
                   for r, c in zip(nr.tolist(), nc.tolist())
                   if (r + c) % 2 == 0]          # 1-in-2 subsample
        prev_p50 = p50

        mm = meta[H] if meta and H < len(meta) else {}
        frames.append({
            'hour': int(H),
            'timestamp': mm.get('timestamp'),
            'wind_speed_ms': mm.get('wind', 0),
            'humidity_pct': mm.get('rh', 0),
            'temp_c': mm.get('temp'),
            'area_ha': int(np.median(areas)),
            'area_p10': int(np.percentile(areas, 10)),
            'area_p90': int(np.percentile(areas, 90)),
            'new_points': new_pts,
        })

    return {'n_frames': len(frames), 'n_seeds': dom['n_seeds'],
            'n_runs': n_runs, 'suppression': bool(suppression), 'frames': frames}


# ---------------------------------------------------------------------------
# Single "honest" ensemble: uncertainty comes ONLY from what is genuinely
# unpredictable — wind beyond ~day 2 (perturbation grows with lead time),
# pyroCb occurrence (stochastic per run), suppression effectiveness.
# Temperature / humidity / soil dryness are used as observed/forecast (they
# are predictable at these lead times). Views are read off the ensemble:
# ref = median footprint, opt = p10-ish, pess = p90-ish, pyro = pyroCb runs.
# ---------------------------------------------------------------------------

def simulate_ensemble(hotspots, wind_records, wind_field=None, veg_fuel=None,
                      veg_bbox=None, max_hours=168, n_runs=12, rng_seed=0,
                      pyro_daily_p=0.07):   # P(≥1 pyroCb sur 7 j) ≈ 40 %
    if binary_dilation is None:
        return None
    dom = _prepare_domain(hotspots, veg_fuel, veg_bbox)
    if dom is None:
        return None
    rng = np.random.default_rng(rng_seed)
    igns, pyro_flags, meta = [], [], None

    for k in range(n_runs):
        pert = {'rng': rng,
                'supp_mult': float(np.exp(rng.normal(0, 0.15)))}
        # wind uncertainty grows with lead time (random walk, ° and ×)
        dn = np.zeros(max_hours)
        sm = np.zeros(max_hours)
        for h in range(1, max_hours):
            day = h / 24.0
            dn[h] = dn[h - 1] * 0.92 + rng.normal(0, 1.8 + 1.1 * day)
            sm[h] = sm[h - 1] * 0.92 + rng.normal(0, 0.015 + 0.008 * day)
        if k > 0:  # run 0 = trajectoire de référence non perturbée
            pert['dir_noise'] = dn
            pert['speed_mult_h'] = np.exp(sm)
        # pyroCb: each afternoon may develop one (stochastic), boosting spread
        cal_arr = np.ones(max_hours)
        spot_arr = np.zeros(max_hours)
        pyro_any = False
        for d in range(max_hours // 24 + 1):
            if rng.random() < pyro_daily_p:
                pyro_any = True
                h0, h1 = d * 24 + 12, min(d * 24 + 22, max_hours)
                if h0 < max_hours:
                    cal_arr[h0:h1] = 1.35
                    spot_arr[h0:h1] = 1.2
                    if 'dir_noise' in pert:
                        pert['dir_noise'][h0:h1] += rng.normal(0, 35, max(h1 - h0, 0))
        pert['cal_h'] = cal_arr
        pert['spot_h'] = spot_arr
        ign, m, n_hours = _run_once(dom, wind_field, wind_records, max_hours,
                                    pert, want_meta=(k == 0), suppression=True)
        igns.append(ign)
        pyro_flags.append(pyro_any)
        if k == 0:
            meta = m

    return {'igns': np.stack(igns), 'pyro': np.array(pyro_flags),
            'dom': dom, 'meta': meta, 'n_hours': n_hours, 'n_runs': n_runs,
            'wind_field': wind_field}


# Smoke transport grid (matches the air-quality overlay bbox)
_SMK_BBOX = (44.45, -1.45, 45.20, -0.35)   # lat0, lon0, lat1, lon1
_SMK_NR, _SMK_NC = 32, 40


def _smoke_series(store, ig_sel, emit_hours):
    """Hourly advection-diffusion-decay smoke model over the burn ensemble.

    S(t+1) = advect(S, wind) * decay + emissions(burning cells)
    Emissions come from cells burning (ignited < 12 h ago) in the selected
    runs; the plume accumulates and drifts like real smoke, unlike an
    instantaneous puff. Returned as AQI-equivalent grids per emitted hour.
    """
    from scipy.ndimage import map_coordinates, gaussian_filter
    la0, lo0, la1, lo1 = _SMK_BBOX
    dom = store['dom']
    n_hours = store['n_hours']
    wf = store.get('wind_field')

    # map burn-domain cells -> smoke-grid indices (precomputed)
    glat, glon = dom['lat_axis'], dom['lon_axis']
    ri = np.clip(((la1 - glat) / (la1 - la0) * (_SMK_NR - 1)).astype(int), 0, _SMK_NR - 1)
    ci = np.clip(((glon - lo0) / (lo1 - lo0) * (_SMK_NC - 1)).astype(int), 0, _SMK_NC - 1)
    RI = np.repeat(ri[:, None], len(glon), 1)
    CI = np.repeat(ci[None, :], len(glat), 0)

    # hourly mean wind (east/north m/s) from the forecast grid
    if wf is not None and wf.get('times'):
        sp = np.asarray(wf['speed']).mean(axis=(1, 2))
        dr_all = np.asarray(wf['dir'])
        ss = np.sin(np.radians(dr_all)).mean(axis=(1, 2))
        cc = np.cos(np.radians(dr_all)).mean(axis=(1, 2))
        dr = (np.degrees(np.arctan2(ss, cc))) % 360
    else:
        sp = np.full(n_hours, 8.0)
        dr = np.full(n_hours, 270.0)

    cell_lat_deg = (la1 - la0) / (_SMK_NR - 1)
    cell_lon_deg = (lo1 - lo0) / (_SMK_NC - 1)
    mid_cos = np.cos(np.radians((la0 + la1) / 2))

    S = np.zeros((_SMK_NR, _SMK_NC))
    yy, xx = np.mgrid[0:_SMK_NR, 0:_SMK_NC].astype(float)
    out = {}
    burn_win = 12   # a cell emits smoke for ~12 h after ignition
    Q = 3.2         # emission strength per burning cell (AQI-equivalent)
    for h in range(n_hours):
        k = min(h, len(sp) - 1)
        u_e = sp[k] * (-np.sin(np.radians(dr[k])))   # toward-east component
        u_n = sp[k] * (-np.cos(np.radians(dr[k])))
        # displacement in grid cells over 1 h (row 0 = north)
        dx = u_e * 3600 / (111_320 * mid_cos) / cell_lon_deg
        dy = -u_n * 3600 / 111_320 / cell_lat_deg
        S = map_coordinates(S, [yy - dy, xx - dx], order=1, mode='constant')
        S = gaussian_filter(S, sigma=0.7) * 0.93
        burning = ((ig_sel >= max(h - burn_win, 0)) & (ig_sel <= h)).mean(axis=0)
        if h < burn_win:
            burning = np.maximum(burning, dom['seed_mask'] * 1.0)
        emit = np.zeros((_SMK_NR, _SMK_NC))
        np.add.at(emit, (RI.ravel(), CI.ravel()), burning.ravel())
        S = S + emit * Q
        if h in emit_hours:
            out[h] = np.clip(S, 0, 300).round(0).astype(int).tolist()
    return out


_VIEW_THR = {'ref': 0.5, 'opt': 0.85, 'pess': 0.2, 'pyro': 0.5}


def derive_view(store, view='ref', emit_every=3):
    """Cheap projection of the stored ensemble into a map view."""
    if store is None:
        return {'n_frames': 0, 'frames': [], 'n_seeds': 0, 'n_runs': 0}
    igns, dom, meta = store['igns'], store['dom'], store['meta']
    n_hours = store['n_hours']
    thr = _VIEW_THR.get(view, 0.5)
    sel = np.ones(len(igns), dtype=bool)
    if view == 'pyro' and store['pyro'].any():
        sel = store['pyro']
    ig = igns[sel]
    seed = dom['seed_mask']

    emit_hours = sorted(set(list(range(0, n_hours, emit_every)) + [n_hours - 1]))
    smoke = _smoke_series(store, ig, set(emit_hours))
    frames = []
    prev = np.zeros_like(seed)
    for H in emit_hours:
        burned_runs = seed[None, :, :] | ((ig >= 0) & (ig <= H))
        prob = burned_runs.mean(axis=0)
        foot = prob >= thr
        areas = burned_runs.sum(axis=(1, 2)) * dom['cell_area_ha']
        newly = foot & ~prev & ~seed
        nr, nc = np.where(newly)
        new_pts = [[round(float(dom['lat_axis'][r]), 4),
                    round(float(dom['lon_axis'][c]), 4), int(H)]
                   for r, c in zip(nr.tolist(), nc.tolist()) if (r + c) % 2 == 0]
        prev = foot
        mm = meta[H] if meta and H < len(meta) else {}
        # headline area matches the view: opt = p10, pess = p90, sinon médiane
        headline = {'opt': np.percentile(areas, 10),
                    'pess': np.percentile(areas, 90)}.get(view, np.median(areas))
        frames.append({
            'hour': int(H), 'timestamp': mm.get('timestamp'),
            'wind_speed_ms': mm.get('wind', 0), 'humidity_pct': mm.get('rh', 0),
            'temp_c': mm.get('temp'),
            'area_ha': int(headline),
            'area_p10': int(np.percentile(areas, 10)),
            'area_p90': int(np.percentile(areas, 90)),
            'new_points': new_pts,
            'smoke': smoke.get(H),
        })
    return {'n_frames': len(frames), 'n_seeds': dom['n_seeds'],
            'n_runs': int(sel.sum()), 'view': view,
            'pyro_runs': int(store['pyro'].sum()),
            'smoke_bbox': list(_SMK_BBOX), 'smoke_shape': [_SMK_NR, _SMK_NC],
            'frames': frames}
