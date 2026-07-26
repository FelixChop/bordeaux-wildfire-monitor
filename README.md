# 🔥 Bordeaux Wildfire Monitor

Carte web temps réel du risque de feu de forêt autour de Bordeaux / Gironde, avec
simulation de propagation heure par heure pilotée par le vent, l'humidité et la
végétation réels.

**Démo :** https://wildfire.152-228-129-170.sslip.io

## Fonctionnalités

- **Feu réel (5 derniers jours)** — foyers NASA FIRMS (VIIRS) affichés en carte de
  chaleur pondérée par la puissance radiative (FRP), rejouables heure par heure.
- **Simulation de propagation (48 h)** — le feu part de *tous* les foyers détectés et
  la zone brûlée grandit heure par heure. Chaque cellule (~280 m) utilise :
  - le **vent local** interpolé depuis une grille de prévisions Open-Meteo,
  - l'**humidité locale** (l'air humide ralentit le feu),
  - la **densité de végétation** (NDVI MODIS) : la forêt brûle vite, l'eau bloque le feu.
- **Champ de vent** affiché en quadrillage de flèches (sens + force), par heure.
- **Overlay végétation** activable (forêt / eau / urbain).
- Interface **responsive** (mobile + desktop), timeline avec lecture animée.

## Sources de données

| Donnée | Source |
|--------|--------|
| Foyers actifs | NASA FIRMS (VIIRS S-NPP NRT) |
| Vent, humidité, température | Open-Meteo (ARPEGE / GFS) |
| Densité de végétation | NASA GIBS — MODIS NDVI 8-day |

## Lancer en local

```bash
pip install -r requirements.txt
export NASA_FIRMS_MAP_KEY="votre_clé"   # gratuite : https://firms.modaps.eosdis.nasa.gov/api/map_key/
python app.py                            # http://localhost:5000
```

## Docker

```bash
docker build -t wildfire-monitor .
docker run -d --name wildfire -p 8090:5000 \
  -e NASA_FIRMS_MAP_KEY="votre_clé" wildfire-monitor
```

Voir [`DEPLOYMENT.md`](DEPLOYMENT.md) pour le déploiement derrière un reverse-proxy.

## Endpoints

| Route | Description |
|-------|-------------|
| `/` | Carte interactive |
| `/api/data` | État courant (foyers, vent, périmètre) |
| `/api/fire-history` | Foyers FIRMS par heure (cumul) |
| `/api/simulation` | Propagation simulée, isochrones par heure |
| `/api/windfield` | Champ de vent (grille) |
| `/api/vegetation.png` | Carte de combustible (overlay) |

## Modèle de propagation

La vitesse du front (*rate of spread*) suit une loi empirique en puissance de la
vitesse du vent (famille Cruz & Alexander 2010), **calibrée sur les vitesses
réelles de feux de tête du massif landais** (0,3–1,5 km/h), modulée par l'humidité
et la densité de végétation locales. La propagation est un modèle raster
multi-sources (dilatation dirigée par le vent, bloquée par l'eau).

> ⚠️ Outil pédagogique / d'illustration du risque — **pas** un outil opérationnel de
> prévision. Le dépôt contient aussi un modèle Monte-Carlo plus complet
> (Rothermel + embers + propagation urbaine) dans `src/` et `run_simulations.py`.

## Structure

```
app.py                 # serveur Flask + fetch temps réel + API
src/fire_front.py      # simulation de propagation spatiale
src/                   # modèle physique (Rothermel, embers, transport, …)
templates/index.html   # frontend (Leaflet, responsive)
Dockerfile
```
