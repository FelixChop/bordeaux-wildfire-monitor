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
    if p.get('fuel_mult') is not None:
        # frozen heterogeneity: fuel corridors & jackpots -> fingering fronts
        fuel_grid = fuel_grid * p['fuel_mult']
    P_GROW = 0.7   # stochastic front advance (per-attempt success)

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
    # Firefighting state: containment (held control lines on the fire edge)
    # + preventive fuel breaks cut AHEAD of the front (dozers, farmers felling
    # trees, tactical burns). One user knob scales everything: supp_level.
    supp_level = sc.get('supp_level', 1.0)
    contained = np.zeros((nrows, ncols), dtype=bool)
    _ones33 = np.ones((3, 3), dtype=bool)

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
            eff = np.clip(supp_base * sc.get('supp_level', 1.0) * supp_mult * ramp * wind_pen, 0.0, 0.95)
            ros = ros * (1.0 - eff)
        residual += ros * 60.0

        if rng is not None:
            ros = ros / P_GROW   # compensate the stochastic advance in mean speed
        b0 = burned.copy()
        for _ in range(8):
            ready = residual >= cell_lat_m
            if not ready.any():
                break
            grew = np.zeros_like(burned)
            for di, dj, ex, ny in dom['nbr']:
                push = (pe_u * ex + pn_u * ny) >= 0.25
                src = burned & ready & push & ~contained
                if not src.any():
                    continue
                grew |= _shift(src, di, dj) & ~burned & (fuel_grid > 0)
            if rng is not None:
                # stochastic front: each advance succeeds with P_GROW ->
                # rough, fingering fire edges instead of smooth rings
                grew &= rng.random(grew.shape) < P_GROW
            burned |= grew
            residual = np.where(ready, residual - cell_lat_m, residual)
            if not grew.any():
                break
        ign[burned & ~b0] = h

        # Ember spotting: routine short throws (0.3-1.5 km, breakouts ahead of
        # the front) + pyroCb long throws (2-8 km) when a fire storm is active.
        pyro_rate = float(spot_h[h]) if spot_h is not None else spot_rate
        if rng is not None:
            recent = burned & (ign >= h - 3) & ~contained
            if h < 3:
                recent = recent | (dom['seed_mask'] & ~contained)
            fr_r, fr_c = np.where(recent)
            if len(fr_r):
                for rate, dmin, dmax in ((pyro_rate, 2000, 8000), (0.35, 300, 1500)):
                    if rate <= 0:
                        continue
                    for _ in range(int(rng.poisson(rate))):
                        i = int(rng.integers(len(fr_r)))
                        dist_m = float(rng.uniform(dmin, dmax))
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

        # --- Active firefighting (capacity-limited) -------------------------
        if suppression and rng is not None:
            ramp_c = min(1.0, h / 18.0)
            # (a) CONTAINMENT: crews hold sections of the fire edge. Capacity
            # ~4 cells/h (≈2 km of line) at normal strength, easier where the
            # local wind is weak (flanks/rear), and it HOLDS once set.
            cap = 4.0 * supp_level * supp_mult * ramp_c
            n_cont = int(cap) + (1 if rng.random() < (cap % 1.0) else 0)
            if n_cont > 0:
                edge = burned & ~contained & binary_dilation(~burned & (fuel_grid > 0), _ones33)
                er, ec = np.where(edge)
                if len(er):
                    w = np.clip(1.15 - speed[er, ec] / 14.0, 0.1, 1.0)
                    w = w / w.sum()
                    pick = rng.choice(len(er), size=min(n_cont, len(er)),
                                      replace=False, p=w)
                    contained[er[pick], ec[pick]] = True
            # (b) PREVENTIVE FUEL BREAKS ahead of the head: every 6 h, dozers
            # and farmers cut a line downwind of the most advanced burning
            # cell (élagage, tranchées, feux tactiques -> fuel quasi nul).
            if h >= 8 and h % 6 == 0:
                act_r, act_c = np.where(burned & ~contained)
                if len(act_r):
                    ue_m = float(pe_u[act_r, act_c].mean())
                    vn_m = float(pn_u[act_r, act_c].mean())
                    proj = act_r * (-vn_m) + act_c * ue_m   # row axis points south
                    i_head = int(np.argmax(proj))
                    hr, hc = int(act_r[i_head]), int(act_c[i_head])
                    cr = hr - int(round(vn_m * 4))          # ~2 km devant la tête
                    cc = hc + int(round(ue_m * 4))
                    half = max(2, int(round(4 * supp_level * ramp_c)))
                    # ligne perpendiculaire à la direction de propagation
                    px, py = -vn_m, -ue_m
                    for t in range(-half, half + 1):
                        rr2 = cr + int(round(py * t))
                        cc2 = cc + int(round(px * t))
                        if 0 <= rr2 < nrows and 0 <= cc2 < ncols:
                            fuel_grid[rr2, cc2] *= 0.08

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
                      pyro_daily_p=0.07, scenario=None):
    if binary_dilation is None:
        return None
    dom = _prepare_domain(hotspots, veg_fuel, veg_bbox)
    if dom is None:
        return None
    from scipy.ndimage import gaussian_filter as _gf
    rng = np.random.default_rng(rng_seed)
    igns, pyro_flags, meta = [], [], None

    for k in range(n_runs):
        pert = {'rng': rng,
                'supp_mult': float(np.exp(rng.normal(0, 0.15)))}
        # frozen fuel heterogeneity (per run): corridors, jackpots, firebreak-
        # like gaps -> fronts grow fingers instead of smooth rings
        noise = _gf(rng.normal(0, 1, (dom['nrows'], dom['ncols'])), sigma=2.5)
        noise = noise / max(noise.std(), 1e-9)
        pert['fuel_mult'] = np.clip(np.exp(0.45 * noise), 0.3, 2.2)
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
                                    pert, want_meta=(k == 0), suppression=True,
                                    scenario=scenario)
        igns.append(ign)
        pyro_flags.append(pyro_any)
        if k == 0:
            meta = m

    return {'igns': np.stack(igns), 'pyro': np.array(pyro_flags),
            'dom': dom, 'meta': meta, 'n_hours': n_hours, 'n_runs': n_runs,
            'wind_field': wind_field}


# Smoke transport grid (matches the air-quality overlay bbox)
_SMK_BBOX = (44.20, -1.75, 45.55, -0.25)   # lat0, lon0, lat1, lon1
_SMK_NR, _SMK_NC = 32, 40


def _hmix_profile(hour_utc):
    """Diurnal boundary-layer mixing height [m] (orders of magnitude: Stull 1988).

    The SAME smoke mass gives much higher ground concentrations at night,
    when the boundary layer collapses to a few hundred metres."""
    h = hour_utc % 24            # heure locale approx = UTC+2
    if 8 <= h <= 17:
        return 1200.0            # couche convective diurne
    if h in (6, 7):
        return 300.0 + (h - 5) * 300.0
    if h in (18, 19):
        return 1200.0 - (h - 17) * 450.0
    return 300.0                 # couche nocturne stable


def _wind_grids(wf, k, LAT, LON):
    """Per-cell east/north wind [m/s] on the smoke grid at hour index k."""
    if wf is None or not wf.get('times'):
        return np.full(LAT.shape, 5.7), np.full(LAT.shape, 0.0)
    gl = np.array(wf['grid_lats'])
    gn = np.array(wf['grid_lons'])
    kk = min(k, len(wf['times']) - 1)
    sp = np.asarray(wf['speed'][kk])
    dr = np.asarray(wf['dir'][kk])
    ue = sp * (-np.sin(np.radians(dr)))
    vn = sp * (-np.cos(np.radians(dr)))
    pts = np.column_stack([LAT.ravel(), LON.ravel()])
    fe = RegularGridInterpolator((gl, gn), ue, bounds_error=False, fill_value=None)
    fn = RegularGridInterpolator((gl, gn), vn, bounds_error=False, fill_value=None)
    return fe(pts).reshape(LAT.shape), fn(pts).reshape(LAT.shape)


# --- PM2.5 emission physics (literature-anchored) ---------------------------
# Fuel consumed in Landes pine stands ~2.5 kg/m2, PM2.5 emission factor
# ~15 g/kg (Urbanski 2013; Andreae 2019), released over ~12 h of flaming +
# smouldering  =>  flux ~3.1 g/m2/h of burning surface.
_EMIT_FLUX = 3.1e6      # ug PM2.5 / m2 (burning surface) / h
_BURN_WIN = 12          # hours a cell keeps emitting after ignition
_DEP = 0.985            # hourly dry-deposition loss on the column burden


def smoke_engine(emission_fn, wf, times, n_hours, emit_hours, k_cal=1.0):
    """Column-burden transport: B [ug/m2] advected by the LOCAL wind field,
    diffused, deposited; ground concentration C = B / H_mix(t) [ug/m3].

    Mass-conservative 2D advection of the burden makes the nocturnal
    concentration spike emerge naturally when H_mix collapses.
    emission_fn(h) -> burden addition grid [ug/m2] for hour h.
    """
    from scipy.ndimage import map_coordinates, gaussian_filter
    la0, lo0, la1, lo1 = _SMK_BBOX
    lat_ax = np.linspace(la1, la0, _SMK_NR)     # row 0 = north
    lon_ax = np.linspace(lo0, lo1, _SMK_NC)
    LON, LAT = np.meshgrid(lon_ax, lat_ax)
    cell_lat_deg = (la1 - la0) / (_SMK_NR - 1)
    cell_lon_deg = (lo1 - lo0) / (_SMK_NC - 1)
    mid_cos = np.cos(np.radians((la0 + la1) / 2))
    yy, xx = np.mgrid[0:_SMK_NR, 0:_SMK_NC].astype(float)
    B = np.zeros((_SMK_NR, _SMK_NC))
    out = {}
    nT = len(times) if times else 0
    for h in range(n_hours):
        k = min(h, nT - 1) if nT else 0
        ue, vn = _wind_grids(wf, k, LAT, LON)
        dxg = ue * 3600 / (111_320 * mid_cos) / cell_lon_deg
        dyg = -vn * 3600 / 111_320 / cell_lat_deg
        emit = emission_fn(h) * k_cal
        disp = max(float(np.abs(dxg).max()), float(np.abs(dyg).max()), 1e-6)
        nsub = max(1, int(np.ceil(disp / 1.2)))
        dep_sub = _DEP ** (1.0 / nsub)
        sig_sub = 0.7 / np.sqrt(nsub)
        for _ in range(nsub):
            B = map_coordinates(B, [yy - dyg / nsub, xx - dxg / nsub],
                                order=1, mode='constant')
            B = gaussian_filter(B, sigma=sig_sub) * dep_sub
            B = B + emit / nsub
        if h in emit_hours:
            hh = 12
            if times and k < nT:
                try:
                    hh = int(times[k][11:13])
                except (ValueError, TypeError):
                    pass
            C = B / _hmix_profile(hh)
            out[h] = np.clip(C, 0, 800).round(0).astype(int).tolist()
    return out


def _smoke_geometry(dom=None):
    la0, lo0, la1, lo1 = _SMK_BBOX
    la_span = (la1 - la0) / (_SMK_NR - 1) * 111_320
    lo_span = ((lo1 - lo0) / (_SMK_NC - 1) * 111_320
               * np.cos(np.radians((la0 + la1) / 2)))
    return la_span * lo_span     # m2 per smoke cell


def _smoke_series(store, ig_sel, emit_hours, k_cal=1.0):
    """PM2.5 [ug/m3] smoke grids of the SIMULATED fire (one member run)."""
    dom = store['dom']
    wf = store.get('wind_field')
    n_hours = store['n_hours']
    times = ((wf or {}).get('times')
             or [m.get('timestamp') for m in (store.get('meta') or [])])
    la0, lo0, la1, lo1 = _SMK_BBOX
    glat, glon = dom['lat_axis'], dom['lon_axis']
    ri = np.clip(((la1 - glat) / (la1 - la0) * (_SMK_NR - 1)).astype(int), 0, _SMK_NR - 1)
    ci = np.clip(((glon - lo0) / (lo1 - lo0) * (_SMK_NC - 1)).astype(int), 0, _SMK_NC - 1)
    RI = np.repeat(ri[:, None], len(glon), 1)
    CI = np.repeat(ci[None, :], len(glat), 0)
    flux_b = _EMIT_FLUX * (dom['cell_lat_m'] ** 2) / _smoke_geometry()
    seed = dom['seed_mask']

    def emission(h):
        burning = ((ig_sel >= max(h - _BURN_WIN, 0)) & (ig_sel <= h)).mean(axis=0)
        if h < _BURN_WIN:
            burning = np.maximum(burning, seed * 1.0)
        emit = np.zeros((_SMK_NR, _SMK_NC))
        np.add.at(emit, (RI.ravel(), CI.ravel()), burning.ravel())
        return emit * flux_b

    return smoke_engine(emission, wf, times, n_hours, emit_hours, k_cal)


def hindcast_smoke(hotspot_offsets, wind_field, n_hours, emit_hours, k_cal=1.0):
    """Replay the REAL fire's smoke over past hours (for calibration).

    hotspot_offsets: [(row, col, ignition_hour_offset)] on the smoke grid;
    each VIIRS detection ~375 m pixel => burning surface ~1.4e5 m2."""
    src = {}
    for r, c, h0 in hotspot_offsets:
        src.setdefault(int(h0), []).append((r, c))
    flux_b = _EMIT_FLUX * 1.4e5 / _smoke_geometry()

    def emission(h):
        emit = np.zeros((_SMK_NR, _SMK_NC))
        for h0, cells in src.items():
            if 0 <= h - h0 < _BURN_WIN:
                for r, c in cells:
                    emit[r, c] += flux_b
        return emit

    times = (wind_field or {}).get('times')
    return smoke_engine(emission, wind_field, times, n_hours, emit_hours, k_cal)


def derive_view(store, view='ref', emit_every=3):
    """Project the ensemble onto a REPRESENTATIVE MEMBER run.

    The >=50 % probability footprint averaged all the chaos away into smooth
    concentric blobs; real fires finger and break out. Each view now shows one
    ACTUAL ensemble member, chosen by final burned area: ref = median run,
    opt = p10 run, pess = p90 run, pyro = median pyroCb run, 'mK' = member K
    (tirage individuel). Area labels keep the full-ensemble p10-p90 spread.
    """
    if store is None:
        return {'n_frames': 0, 'frames': [], 'n_seeds': 0, 'n_runs': 0}
    igns, dom, meta = store['igns'], store['dom'], store['meta']
    n_hours = store['n_hours']
    seed = dom['seed_mask']
    n = len(igns)

    finals = (seed[None, :, :] | ((igns >= 0) & (igns < n_hours))).sum(axis=(1, 2))
    order = np.argsort(finals).tolist()
    pyro = store['pyro']
    nonpyro = [i for i in order if not pyro[i]] or order
    pyros = [i for i in order if pyro[i]]
    # Sémantique claire : Référence & Optimiste = scénarios SANS orage de feu ;
    # « Si pyroCb » = scénarios AVEC ; Pessimiste = queue haute tous tirages.
    if view.startswith('m') and view[1:].isdigit():
        member = min(int(view[1:]), n - 1)
    elif view == 'pyro' and pyros:
        member = pyros[len(pyros) // 2]
    elif view == 'opt':
        member = nonpyro[int(round(0.10 * (len(nonpyro) - 1)))]
    elif view == 'pess':
        member = order[int(round(0.90 * (n - 1)))]
    else:  # ref
        member = nonpyro[int(round(0.50 * (len(nonpyro) - 1)))]
    ig_m = igns[member]

    emit_hours = sorted(set(list(range(0, n_hours, emit_every)) + [n_hours - 1]))
    smoke = _smoke_series(store, igns[member:member + 1], set(emit_hours),
                          k_cal=store.get('smoke_k', 1.0))
    frames = []
    prev_h = -1
    for H in emit_hours:
        burned_runs = seed[None, :, :] | ((igns >= 0) & (igns <= H))
        areas = burned_runs.sum(axis=(1, 2)) * dom['cell_area_ha']
        area_m = float((seed | ((ig_m >= 0) & (ig_m <= H))).sum() * dom['cell_area_ha'])
        newly = (ig_m > prev_h) & (ig_m <= H) & ~seed
        nr, nc = np.where(newly)
        new_pts = [[round(float(dom['lat_axis'][r]), 4),
                    round(float(dom['lon_axis'][c]), 4), int(ig_m[r, c])]
                   for r, c in zip(nr.tolist(), nc.tolist()) if (r + c) % 2 == 0]
        prev_h = H
        mm = meta[H] if meta and H < len(meta) else {}
        frames.append({
            'hour': int(H), 'timestamp': mm.get('timestamp'),
            'wind_speed_ms': mm.get('wind', 0), 'humidity_pct': mm.get('rh', 0),
            'temp_c': mm.get('temp'),
            'area_ha': int(area_m),
            'area_p10': int(np.percentile(areas, 10)),
            'area_p90': int(np.percentile(areas, 90)),
            'new_points': new_pts,
            'smoke': smoke.get(H),
        })
    return {'n_frames': len(frames), 'n_seeds': dom['n_seeds'],
            'n_runs': n, 'view': view, 'member': int(member),
            'member_pyro': bool(store['pyro'][member]),
            'pyro_runs': int(store['pyro'].sum()),
            'smoke_bbox': list(_SMK_BBOX), 'smoke_shape': [_SMK_NR, _SMK_NC],
            'frames': frames}

