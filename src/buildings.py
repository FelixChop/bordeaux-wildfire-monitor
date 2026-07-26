import numpy as np
from dataclasses import dataclass
import json
from typing import List, Dict

@dataclass
class Building:
    id: int
    x: float
    y: float
    height: float
    floor_area: float
    construction_year: int
    roof_type: str
    wall_material: str
    window_fraction: float
    distances_to_neighbors: Dict[int, float]
    exposed_perimeter: float
    Rey_index: float

class UrbanGraph:
    """
    Bordeaux metropolitan area as graph of buildings.

    Nodes = buildings. Edges = fire transmission paths.
    GIS data source: BD TOPO + Cadastre + OpenStreetMap.

    For this simulation: synthetic urban grid representing
    typical Bordeaux suburban density (20-40 buildings/hectare).
    """

    def __init__(self, n_buildings: int = 5000, extent_m: float = 25000):
        self.buildings: Dict[int, Building] = {}
        self.n_buildings = n_buildings
        self.extent = extent_m
        self._generate_synthetic_buildings()

    def _generate_synthetic_buildings(self):
        """
        Synthetic Bordeaux suburb grid.
        """

        for bid in range(self.n_buildings):
            x = np.random.uniform(-self.extent / 2, self.extent / 2)
            y = np.random.uniform(-self.extent / 2, self.extent / 2)

            height = np.random.choice([8, 10, 12, 15, 20], p=[0.4, 0.3, 0.15, 0.1, 0.05])
            floor_area = height * np.random.uniform(100, 400)

            year = np.random.choice(
                list(range(1950, 2025)),
                p=np.exp(-np.arange(1950, 2025, dtype=float) / 2000)[::-1] /
                  sum(np.exp(-np.arange(1950, 2025, dtype=float) / 2000))
            )

            roof_type = np.random.choice(
                ["tile", "metal", "concrete", "asphalt"],
                p=[0.5, 0.2, 0.2, 0.1]
            )

            wall_material = np.random.choice(
                ["brick", "stone", "concrete", "wood"],
                p=[0.4, 0.3, 0.25, 0.05]
            )

            window_frac = np.random.uniform(0.15, 0.35)
            exposed_perim = np.random.uniform(80, 200)

            rey = np.clip(
                np.random.normal(0.7, 0.15),
                0.3, 1.0
            )

            self.buildings[bid] = Building(
                id=bid,
                x=x, y=y,
                height=height,
                floor_area=floor_area,
                construction_year=year,
                roof_type=roof_type,
                wall_material=wall_material,
                window_fraction=window_frac,
                distances_to_neighbors={},
                exposed_perimeter=exposed_perim,
                Rey_index=rey
            )

        self._compute_neighbors()

    def _compute_neighbors(self, radius: float = 150.0):
        """
        For each building, find neighbors within radius and compute distance.
        """

        for bid in self.buildings:
            b = self.buildings[bid]
            for oid in self.buildings:
                if oid == bid:
                    continue

                o = self.buildings[oid]
                dist = np.sqrt((b.x - o.x)**2 + (b.y - o.y)**2)

                if dist <= radius:
                    b.distances_to_neighbors[oid] = dist

    def get_neighbors(self, building_id: int, radius: float = 150.0) -> List[int]:
        """Return IDs of neighbors within radius."""
        if building_id not in self.buildings:
            return []

        return [
            nid for nid, d in self.buildings[building_id].distances_to_neighbors.items()
            if d <= radius
        ]

    def export_geojson(self, filename: str):
        """Export building locations as GeoJSON."""
        features = []
        for bid, b in self.buildings.items():
            features.append({
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [b.x, b.y]},
                "properties": {
                    "id": bid,
                    "height": b.height,
                    "roof": b.roof_type,
                    "year": b.construction_year
                }
            })

        geojson = {"type": "FeatureCollection", "features": features}
        with open(filename, 'w') as f:
            json.dump(geojson, f)
