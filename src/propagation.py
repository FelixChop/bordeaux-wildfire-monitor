import numpy as np
from typing import Set, Dict
from .buildings import Building, UrbanGraph
from .meteorology import MeteorologicalState

class UrbanPropagationModel:
    """
    Fire spread building-to-building via:
    - Radiative heat transfer
    - Convection + hot gas
    - Flying embers from burning building

    Calibration: fire dynamics from Drysdale + SFPE + post-wildfire surveys.
    """

    def __init__(self):
        self.stefan_boltzmann = 5.67e-8
        self.emissivity_flame = 0.95
        self.flame_base_temperature = 1000
        self.view_factor_scale = 0.5
        self.secondary_ember_production = 0.3

    def radiative_flux_at_building(
        self,
        source_building: Building,
        target_building: Building,
        flame_length: float,
        fireline_intensity: float
    ) -> float:
        """
        Radiative heat flux [kW/m²] from burning building to target.

        I = ε σ T^4 · (A_flame / r²) · view_factor

        Typical: 12.5 kW/m² is critical for ignition.
        """

        if flame_length < 1.0:
            return 0.0

        r = np.sqrt(
            (source_building.x - target_building.x)**2 +
            (source_building.y - target_building.y)**2
        )

        if r > flame_length * 5:
            return 0.0

        t_flame = self.flame_base_temperature + 100 * fireline_intensity / 1000
        radiation_intensity = self.stefan_boltzmann * self.emissivity_flame * (t_flame ** 4)

        view_factor = self.view_factor_scale * (flame_length / max(r, 1.0)) ** 1.5

        flux = radiation_intensity * view_factor

        return np.clip(flux, 0.0, 50.0)

    def propagation_probability(
        self,
        source_building: Building,
        target_building: Building,
        distance: float,
        radiative_flux: float,
        wind_speed: float,
        wind_direction: float,
        flame_length: float
    ) -> float:
        """
        P(fire spreads from source → target).

        Depends on:
        - Distance
        - Radiative flux
        - Wind (convection direction)
        - Building separation
        """

        if distance > flame_length * 3:
            return 0.0

        if radiative_flux < 5:
            base_prob = 0.05 * (1.0 - distance / (flame_length * 3))
        else:
            base_prob = 0.15 + 0.4 * (radiative_flux / 20.0)

        dx = target_building.x - source_building.x
        dy = target_building.y - source_building.y

        wind_angle = np.arctan2(dy, dx) - np.radians(wind_direction)
        wind_alignment = np.cos(wind_angle)

        if wind_alignment > 0.5:
            convection_factor = 1.0 + wind_speed / 5.0
        else:
            convection_factor = 0.5

        prob = base_prob * convection_factor
        return np.clip(prob, 0.0, 1.0)

    def step_urban_fire(
        self,
        graph: UrbanGraph,
        burning_buildings: Set[int],
        new_ignitions: Set[int],
        meteo: MeteorologicalState,
        flame_length: float,
        fireline_intensity: float
    ) -> Dict[int, float]:
        """
        Compute ignition probabilities for all neighbors of burning buildings.

        Returns: dict mapping building_id → P(ignition in next step)
        """

        ignition_probs = {}

        for source_id in burning_buildings:
            if source_id not in graph.buildings:
                continue

            source = graph.buildings[source_id]
            neighbors = graph.get_neighbors(source_id, radius=flame_length * 5)

            for target_id in neighbors:
                if target_id in burning_buildings or target_id in new_ignitions:
                    continue

                if target_id not in graph.buildings:
                    continue

                target = graph.buildings[target_id]
                dist = source.distances_to_neighbors.get(target_id, 150.0)

                rad_flux = self.radiative_flux_at_building(
                    source, target, flame_length, fireline_intensity
                )

                prob = self.propagation_probability(
                    source, target, dist, rad_flux,
                    meteo.wind_speed, meteo.wind_direction, flame_length
                )

                if target_id in ignition_probs:
                    ignition_probs[target_id] = max(ignition_probs[target_id], prob)
                else:
                    ignition_probs[target_id] = prob

        return ignition_probs

    def burn_buildings(
        self,
        ignition_probs: Dict[int, float]
    ) -> Set[int]:
        """
        Bernoulli trials: which buildings actually ignite?
        """

        newly_burning = set()
        for bid, prob in ignition_probs.items():
            if np.random.random() < prob:
                newly_burning.add(bid)

        return newly_burning
