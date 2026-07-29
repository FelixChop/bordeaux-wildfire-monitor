"""Backtest de la propagation simulée contre les détections satellites réelles.

Principe : on rejoue des fenêtres passées (ex. la flambée de mercredi, puis le
déclin) — graines = foyers détectés AVANT t0, météo = champs RÉELLEMENT
observés — et on compare l'empreinte simulée aux nouvelles détections FIRMS
réellement survenues dans la fenêtre. Une recherche sur grille choisit les
paramètres (ros_cal, supp_base, cont_base) qui minimisent l'écart
(sur-catastrophisme comme sous-estimation).

Les paramètres retenus sont persistés et appliqués à toutes les simulations.
"""
import numpy as np
from datetime import datetime, timedelta

from .fire_front import _prepare_domain, _run_once


def _parse(ts):
    try:
        return datetime.fromisoformat((ts or '').replace('Z', '').replace(' ', 'T'))
    except ValueError:
        return None


def _slice_field(wf, i0):
    """Sous-champ météo démarrant à l'index horaire i0."""
    out = {}
    for k in ('grid_lats', 'grid_lons'):
        out[k] = wf[k]
    out['times'] = wf['times'][i0:]
    for k in ('speed', 'dir', 'rh', 'temp', 'soil'):
        v = wf.get(k)
        out[k] = np.asarray(v)[i0:] if v is not None else None
    return out


def _raster_cells(dom, pts):
    """Masque des cellules du domaine touchées par une liste de points."""
    m = np.zeros((dom['nrows'], dom['ncols']), dtype=bool)
    la0, lo0 = dom['lat_axis'][0], dom['lon_axis'][0]
    dlat = dom['lat_axis'][1] - dom['lat_axis'][0] if len(dom['lat_axis']) > 1 else 0.005
    for la, lo in pts:
        r = int((la - la0) / dlat)
        c = int((lo - lo0) / dlat)
        if 0 <= r < dom['nrows'] and 0 <= c < dom['ncols']:
            m[r, c] = True
    return m


def _window_loss(hotspots, wf, veg, bbox, t0_idx, horizon_h, params, n_runs=2):
    """Perte d'une fenêtre : simule depuis t0 et compare au réel."""
    from scipy.ndimage import binary_dilation
    times = wf['times']
    t0 = _parse(times[t0_idx])
    seeds = [h for h in hotspots
             if (_parse(h.get('timestamp')) or datetime.min) <= t0
             and (t0 - (_parse(h.get('timestamp')) or datetime.min)) <= timedelta(hours=24)]
    obs = [(h['lat'], h['lon']) for h in hotspots
           if (lambda d: d and t0 < d <= t0 + timedelta(hours=horizon_h))
              (_parse(h.get('timestamp')))]
    if len(seeds) < 5 or len(obs) < 3:
        return None
    dom = _prepare_domain(seeds, veg, bbox)
    wfx = _slice_field(wf, t0_idx)
    sc = {'supp_level': 1.0, **params}
    sims = []
    for s in range(n_runs):
        pert = {'rng': np.random.default_rng(1000 + s)}
        ign = _run_once(dom, wfx, None, horizon_h, pert=pert,
                        suppression=True, scenario=sc)[0]
        sims.append((ign >= 0) & ~dom['seed_mask'])
    sim_new = sims[int(np.argsort([s.sum() for s in sims])[len(sims) // 2])]
    obs_m = _raster_cells(dom, obs) & ~binary_dilation(dom['seed_mask'],
                                                       np.ones((3, 3), bool))
    a_sim = max(sim_new.sum() * dom['cell_area_ha'], dom['cell_area_ha'])
    a_obs = max(obs_m.sum() * dom['cell_area_ha'], dom['cell_area_ha'])
    sim_dil = binary_dilation(sim_new | dom['seed_mask'], np.ones((5, 5), bool))
    recall = float((obs_m & sim_dil).sum() / max(obs_m.sum(), 1))
    loss = float(np.log(a_sim / a_obs) ** 2 + 0.7 * (1.0 - recall) ** 2)
    return {'loss': loss, 'a_sim': int(a_sim), 'a_obs': int(a_obs),
            'recall': round(recall, 2)}


def backtest_calibrate(hotspots, wind_field, veg, bbox, log=print):
    """Recherche des paramètres qui collent aux dernières 96 h observées."""
    now = datetime.utcnow()
    times = wind_field['times']
    now_idx = next((i for i, t in enumerate(times)
                    if (_parse(t) or datetime.min) >= now.replace(
                        minute=0, second=0, microsecond=0)), len(times) - 1)
    # fenêtre A : J-4 -> J-2 (croissance) ; fenêtre B : J-2 -> maintenant (déclin)
    windows = [(max(now_idx - 96, 1), 48), (max(now_idx - 48, 1), 48)]
    grid = [{'ros_cal': rc, 'supp_base': sb, 'cont_base': cb}
            for rc in (0.5, 0.7, 0.85, 1.0)
            for sb in (0.75, 0.85, 0.92)
            for cb in (4.0, 8.0, 14.0)]
    results = []
    for k, params in enumerate(grid):
        losses, det = [], []
        for t0_idx, hz in windows:
            r = _window_loss(hotspots, wind_field, veg, bbox, t0_idx, hz, params)
            if r:
                losses.append(r['loss'])
                det.append(r)
        if losses:
            results.append({'params': params,
                            'loss': float(np.mean(losses)), 'windows': det})
        if (k + 1) % 9 == 0:
            log(f"backtest {k + 1}/{len(grid)}…")
    if not results:
        return None
    results.sort(key=lambda r: r['loss'])
    best = results[0]
    log("backtest terminé — top 3 :")
    for r in results[:3]:
        w = ' | '.join(f"sim {d['a_sim']:,} ha vs réel {d['a_obs']:,} ha "
                       f"(recall {d['recall']})" for d in r['windows'])
        log(f"  loss={r['loss']:.3f} {r['params']} — {w}")
    return best
