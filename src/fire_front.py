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


def _prepare_domain(hotspots, veg_fuel, veg_bbox, cell_deg=0.005,
                    max_cells=360_000, burned_pts=None):
    seeds = [(h['lat'], h['lon']) for h in hotspots if 'lat' in h and 'lon' in h]
    if not seeds:
        return None
    lats = np.array([s[0] for s in seeds])
    lons = np.array([s[1] for s in seeds])
    margin = 0.60
    lat_min, lat_max = lats.min() - margin, lats.max() + margin
    lon_min, lon_max = lons.min() - margin, lons.max() + margin
    # résolution ADAPTATIVE : un domaine national grossit ses cellules pour
    # rester calculable (France entière -> ~2.4 km/cellule, ~350 k cellules)
    n_est = ((lat_max - lat_min) / cell_deg) * ((lon_max - lon_min) / cell_deg)
    if n_est > max_cells:
        cell_deg = float(np.sqrt((lat_max - lat_min) * (lon_max - lon_min) / max_cells))
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

    # Cicatrices REELLES : le NDVI (composite 8 j) montre encore vert ce qui
    # a brule — on retire le combustible des cellules deja parcourues par le
    # feu (detections passees), le front ne peut plus re-bruler le brule.
    if burned_pts:
        from scipy.ndimage import binary_dilation as _bd
        scar = np.zeros((nrows, ncols), dtype=bool)
        for bla, blo in burned_pts:
            r = int((bla - lat_min) / cell_deg)
            c = int((blo - lon_min) / cell_deg)
            if 0 <= r < nrows and 0 <= c < ncols:
                scar[r, c] = True
        scar = _bd(scar, np.ones((3, 3), bool))
        fuel_grid = np.where(scar & ~seed_mask, fuel_grid * 0.06, fuel_grid)

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
    cal_mult = (p.get('cal_mult', 1.0) * sc.get('pyro_cal', 1.0)
                * sc.get('ros_cal', 1.0))   # calibration backtest
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
        # Bilinéaire vectorisée à poids précalculés : ~100x plus rapide que
        # scipy sur les gros domaines (France ~350 k cellules x 168 h x 5 champs)
        LATg = dom['pts'][:, 0].reshape(nrows, ncols)
        LONg = dom['pts'][:, 1].reshape(nrows, ncols)
        _iy = np.clip(np.searchsorted(glats, LATg) - 1, 0, len(glats) - 2)
        _ix = np.clip(np.searchsorted(glons, LONg) - 1, 0, len(glons) - 2)
        _wy = np.clip((LATg - glats[_iy]) / (glats[_iy + 1] - glats[_iy]), 0, 1)
        _wx = np.clip((LONg - glons[_ix]) / (glons[_ix + 1] - glons[_ix]), 0, 1)
        _w00 = (1 - _wy) * (1 - _wx)
        _w10 = _wy * (1 - _wx)
        _w01 = (1 - _wy) * _wx
        _w11 = _wy * _wx

        def interp(f2d):
            f = np.asarray(f2d)
            return (f[_iy, _ix] * _w00 + f[_iy + 1, _ix] * _w10
                    + f[_iy, _ix + 1] * _w01 + f[_iy + 1, _ix + 1] * _w11)
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

    pyro_until = -1
    pyro_any = False
    pyro_mask = np.zeros((nrows, ncols), dtype=bool)
    for h in range(n_hours):
        # PyroCb ÉMERGENT : chaque après-midi, la probabilité qu'un orage de
        # feu se déclenche dépend de l'INTENSITÉ du feu simulé (surface active)
        # et de la chaleur — un feu mourant n'en produit plus.
        if rng is not None and h % 24 == 12:
            act = burned & ~contained
            if act.any():
                from scipy.ndimage import label as _label
                lab, nlab = _label(act)
                sizes = np.bincount(lab.ravel())[1:]
                act_ha = float(sizes.max()) * dom['cell_area_ha']
            else:
                act_ha = 0.0
            p_day = min(0.35, 0.12 * (act_ha / 20000.0)
                        * float(sc.get('pyro_mult', 1.0)))
            if act_ha > 0 and rng.random() < p_day:
                pyro_until = h + 10
                pyro_any = True
                # l'orage de feu est LOCAL : il se forme au-dessus du plus
                # grand incendie actif, pas sur tout le domaine
                big = (lab == (int(np.argmax(sizes)) + 1))
                br, bc = np.where(big)
                _rg, _cg = np.meshgrid(np.arange(nrows), np.arange(ncols),
                                       indexing='ij')
                _rad = 30_000.0 / dom['cell_lat_m']        # ~30 km
                pyro_mask = ((_rg - float(br.mean())) ** 2
                             + (_cg - float(bc.mean())) ** 2) < _rad ** 2
        pyro_on = (h <= pyro_until)
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
        if pyro_on and rng is not None and pyro_mask.any():
            # vents chaotiques de l'orage de feu, localises sur le foyer
            _a = np.radians(float(rng.normal(0, 35)))
            _ca, _sa = np.cos(_a), np.sin(_a)
            pe, pn = (np.where(pyro_mask, pe * _ca - pn * _sa, pe),
                      np.where(pyro_mask, pe * _sa + pn * _ca, pn))
        pmag = np.hypot(pe, pn) + 1e-9
        pe_u, pn_u = pe / pmag, pn / pmag

        cal_eff = cal_mult * (float(cal_h[h]) if cal_h is not None else 1.0)
        cal_grid = np.where(pyro_mask, cal_eff * 1.35, cal_eff) if pyro_on             else cal_eff                        # intensification convective locale
        ros = _ros_grid(speed, rh, fuel_grid, temp, cal_grid, soil)
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
        pyro_rate = (float(spot_h[h]) if spot_h is not None else spot_rate) + (1.2 if pyro_on else 0.0)
        if rng is not None:
            recent = burned & (ign >= h - 3) & ~contained
            if h < 3:
                recent = recent | (dom['seed_mask'] & ~contained)
            recent_py = recent & pyro_mask if pyro_on else recent
            fr_r, fr_c = np.where(recent)
            fp_r, fp_c = np.where(recent_py)
            if len(fr_r):
                for rate, dmin, dmax in ((pyro_rate, 2000, 8000), (0.35, 300, 1500)):
                    if rate <= 0:
                        continue
                    sr, sc2 = (fp_r, fp_c) if dmin >= 2000 else (fr_r, fr_c)
                    if not len(sr):
                        continue
                    for _ in range(int(rng.poisson(rate))):
                        i = int(rng.integers(len(sr)))
                        dist_m = float(rng.uniform(dmin, dmax))
                        ang = np.radians(float(rng.normal(0, 20)))
                        ex, ny = pe_u[sr[i], sc2[i]], pn_u[sr[i], sc2[i]]
                        ca, sa = np.cos(ang), np.sin(ang)
                        ex2, ny2 = ex * ca - ny * sa, ex * sa + ny * ca
                        rr = sr[i] + int(round(ny2 * dist_m / cell_lat_m))
                        cc = sc2[i] + int(round(ex2 * dist_m / cell_lat_m))
                        if (0 <= rr < nrows and 0 <= cc < ncols
                                and not burned[rr, cc] and fuel_grid[rr, cc] > 0.2
                                and rng.random() < min(1.0, (555.0 / cell_lat_m) ** 2)):
                            burned[rr, cc] = True
                            ign[rr, cc] = h

        # --- Nouveaux departs (ignitions humaines/foudre) -------------------
        # La France reelle allume 10-30 feux/jour : sans eux, toute prevision
        # nationale s'eteint. Tirage quotidien Poisson(sc['ignition_rate']),
        # place la ou le risque est maximal (conditions du moment x fuel).
        if rng is not None and h % 24 == 14 and sc.get('ignition_rate', 0) > 0:
            n_new = int(rng.poisson(float(sc['ignition_rate'])))
            if n_new > 0:
                risk = _ros_grid(speed, rh, fuel_grid, temp, 1.0, soil) \
                    * (fuel_grid > 0.25) * (~burned)
                flat = risk.ravel()
                tot = flat.sum()
                if tot > 0:
                    picks = rng.choice(flat.size, size=n_new, replace=False,
                                       p=flat / tot)
                    for pk in picks.tolist():
                        rr3, cc3 = pk // ncols, pk % ncols
                        burned[rr3, cc3] = True
                        ign[rr3, cc3] = h

        # --- Reprises de braises --------------------------------------------
        # Un sol brule garde des braises ~3 jours : par vent >6 m/s, une
        # cellule refroidie au contact de combustible intact peut repartir
        # (P ~0.5 %/h/cellule, modulable scenario 'rekindle').
        if rng is not None and h > 6:
            rk = float(sc.get('rekindle', 0.005))
            if rk > 0:
                embers = (burned & (ign < h - 6) & (ign > h - 72) & ~contained
                          & binary_dilation(~burned & (fuel_grid > 0.2), _ones33)
                          & (speed > 6.0))
                er2, ec2 = np.where(embers)
                if len(er2):
                    rekin = rng.random(len(er2)) < rk * np.clip(
                        speed[er2, ec2] / 8.0, 0.5, 2.5)
                    if rekin.any():
                        ign[er2[rekin], ec2[rekin]] = h   # redevient front actif
        # --- Active firefighting (capacity-limited) -------------------------
        if suppression and rng is not None:
            ramp_c = min(1.0, h / 18.0)
            # (a) CONTAINMENT: crews hold sections of the fire edge. Capacity
            # ~4 cells/h (≈2 km of line) at normal strength, easier where the
            # local wind is weak (flanks/rear), and it HOLDS once set.
            # cap_mult ~ nb d'incendies se partageant les moyens : rendements
            # decroissants (sqrt) ; 555/cell_lat_m rend la capacite (km de
            # ligne/h) invariante a la resolution du raster (calibree a 500 m)
            _cm = np.sqrt(max(1.0, sc.get('cap_mult', 1.0)))
            cap = (sc.get('cont_base', 4.0) * supp_level * supp_mult * ramp_c
                   * _cm * (555.0 / dom['cell_lat_m']))
            # rupture de lignes : une cellule tenue, au contact du feu actif
            # et par vent > 9 m/s, cede avec P=4 %/h (episodes reels)
            if contained.any():
                at_risk = (contained
                           & binary_dilation(burned & ~contained, _ones33)
                           & (speed > 9.0))
                if at_risk.any():
                    rr_b, cc_b = np.where(at_risk)
                    brk = rng.random(len(rr_b)) < 0.04
                    contained[rr_b[brk], cc_b[brk]] = False
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
                    order_p = np.argsort(-proj)
                    heads, n_lines = [], max(1, int(round(np.sqrt(
                        max(1.0, sc.get('cap_mult', 1.0))))))
                    for i_h in order_p.tolist():
                        hr, hc = int(act_r[i_h]), int(act_c[i_h])
                        if all(abs(hr - a) + abs(hc - b) > 12 for a, b in heads):
                            heads.append((hr, hc))
                        if len(heads) >= n_lines:
                            break
                    half = max(2, int(round(4 * supp_level * ramp_c)))
                    px, py = -vn_m, -ue_m
                    for hr, hc in heads:
                        cr = hr - int(round(vn_m * 4))      # devant la tête
                        cc = hc + int(round(ue_m * 4))
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

    return ign, meta, n_hours, pyro_any


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
        ign, m, n_hours, _pa = _run_once(dom, wind_field, wind_records, max_hours,
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
        dense = dom['cell_lat_m'] < 1000
        new_pts = [[round(float(dom['lat_axis'][r]), 4),
                    round(float(dom['lon_axis'][c]), 4), int(H)]
                   for r, c in zip(nr.tolist(), nc.tolist())
                   if not dense or (r + c) % 2 == 0]   # 1-in-2 si grille fine
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
                      pyro_daily_p=0.07, scenario=None, smk_bbox=None,
                      burned_pts=None):
    if binary_dilation is None:
        return None
    dom = _prepare_domain(hotspots, veg_fuel, veg_bbox, burned_pts=burned_pts)
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
        ign, m, n_hours, pyro_any = _run_once(
            dom, wind_field, wind_records, max_hours,
            pert, want_meta=(k == 0), suppression=True, scenario=scenario)
        igns.append(ign)
        pyro_flags.append(pyro_any)
        if k == 0:
            meta = m

    return {'igns': np.stack(igns), 'pyro': np.array(pyro_flags),
            'dom': dom, 'meta': meta, 'n_hours': n_hours, 'n_runs': n_runs,
            'wind_field': wind_field, 'smk_bbox': smk_bbox}


# Default smoke transport grid (Gironde) — overridable per zone
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


def smoke_engine(emission_fn, wf, times, n_hours, emit_hours, k_cal=1.0, smk_bbox=None):
    """Column-burden transport: B [ug/m2] advected by the LOCAL wind field,
    diffused, deposited; ground concentration C = B / H_mix(t) [ug/m3].

    Mass-conservative 2D advection of the burden makes the nocturnal
    concentration spike emerge naturally when H_mix collapses.
    emission_fn(h) -> burden addition grid [ug/m2] for hour h.
    """
    from scipy.ndimage import map_coordinates, gaussian_filter
    la0, lo0, la1, lo1 = smk_bbox or _SMK_BBOX
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


def _smoke_geometry(smk_bbox=None):
    la0, lo0, la1, lo1 = smk_bbox or _SMK_BBOX
    la_span = (la1 - la0) / (_SMK_NR - 1) * 111_320
    lo_span = ((lo1 - lo0) / (_SMK_NC - 1) * 111_320
               * np.cos(np.radians((la0 + la1) / 2)))
    return la_span * lo_span     # m2 per smoke cell


def _smoke_series(store, ig_sel, emit_hours, k_cal=1.0):
    smk_bbox = store.get('smk_bbox') or _SMK_BBOX
    """PM2.5 [ug/m3] smoke grids of the SIMULATED fire (one member run)."""
    dom = store['dom']
    wf = store.get('wind_field')
    n_hours = store['n_hours']
    times = ((wf or {}).get('times')
             or [m.get('timestamp') for m in (store.get('meta') or [])])
    la0, lo0, la1, lo1 = smk_bbox
    glat, glon = dom['lat_axis'], dom['lon_axis']
    ri = np.clip(((la1 - glat) / (la1 - la0) * (_SMK_NR - 1)).astype(int), 0, _SMK_NR - 1)
    ci = np.clip(((glon - lo0) / (lo1 - lo0) * (_SMK_NC - 1)).astype(int), 0, _SMK_NC - 1)
    RI = np.repeat(ri[:, None], len(glon), 1)
    CI = np.repeat(ci[None, :], len(glat), 0)
    flux_b = _EMIT_FLUX * (dom['cell_lat_m'] ** 2) / _smoke_geometry(smk_bbox)
    seed = dom['seed_mask']

    def emission(h):
        burning = ((ig_sel >= max(h - _BURN_WIN, 0)) & (ig_sel <= h)).mean(axis=0)
        if h < _BURN_WIN:
            burning = np.maximum(burning, seed * 1.0)
        emit = np.zeros((_SMK_NR, _SMK_NC))
        np.add.at(emit, (RI.ravel(), CI.ravel()), burning.ravel())
        return emit * flux_b

    return smoke_engine(emission, wf, times, n_hours, emit_hours, k_cal, smk_bbox)


def hindcast_smoke(hotspot_offsets, wind_field, n_hours, emit_hours, k_cal=1.0, smk_bbox=None):
    """Replay the REAL fire's smoke over past hours (for calibration).

    hotspot_offsets: [(row, col, ignition_hour_offset)] on the smoke grid;
    each VIIRS detection ~375 m pixel => burning surface ~1.4e5 m2."""
    src = {}
    for r, c, h0 in hotspot_offsets:
        src.setdefault(int(h0), []).append((r, c))
    flux_b = _EMIT_FLUX * 1.4e5 / _smoke_geometry(smk_bbox)

    def emission(h):
        emit = np.zeros((_SMK_NR, _SMK_NC))
        for h0, cells in src.items():
            if 0 <= h - h0 < _BURN_WIN:
                for r, c in cells:
                    emit[r, c] += flux_b
        return emit

    times = (wind_field or {}).get('times')
    return smoke_engine(emission, wind_field, times, n_hours, emit_hours, k_cal, smk_bbox)


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
        dense = dom['cell_lat_m'] < 1000
        new_pts = [[round(float(dom['lat_axis'][r]), 4),
                    round(float(dom['lon_axis'][c]), 4), int(ig_m[r, c])]
                   for r, c in zip(nr.tolist(), nc.tolist())
                   if not dense or (r + c) % 2 == 0]
        # re-detections : cellules brulees il y a 6-60 h qui couvent encore —
        # echantillon stable (~1/4) pour mimer les re-passages satellites
        smold = (ig_m >= 0) & (ig_m > H - 120) & (ig_m <= H - 6) & ~seed
        sr_, sc_ = np.where(smold)
        for r, c in zip(sr_.tolist(), sc_.tolist()):
            if (r * 7 + c * 13 + H) % 4 == 0:
                new_pts.append([round(float(dom['lat_axis'][r]), 4),
                                round(float(dom['lon_axis'][c]), 4),
                                int(H - 8)])
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
            'cell_m': int(round(dom['cell_lat_m'])),
            'n_runs': n, 'view': view, 'member': int(member),
            'member_pyro': bool(store['pyro'][member]),
            'pyro_runs': int(store['pyro'].sum()),
            'smoke_bbox': list(store.get('smk_bbox') or _SMK_BBOX),
            'smoke_shape': [_SMK_NR, _SMK_NC],
            'frames': frames}

