import numpy as np
import json
from typing import Set, List, Dict
from dataclasses import dataclass, asdict
from tqdm import tqdm

from .meteorology import MeteorologicalModel
from .rothermel import RothermelModel
from .embers import EmbersModel
from .transport import AtmosphericTransportModel
from .buildings import UrbanGraph
from .ignition import IgnitionModel
from .propagation import UrbanPropagationModel

@dataclass
class SimulationResult:
    simulation_id: int
    scenario: str
    n_buildings_burned: int
    burned_area_m2: float
    max_extent_m: float
    fire_duration_min: int
    conflagration_occurred: bool
    initial_ember_count: int
    buildings_ignited_by_embers: int

class MonteCarloEngine:
    """
    Full coupled model: weather → wildfire → embers → transport → urban ignition → propagation.
    Supports both synthetic (Weibull-sampled) and observed (real-time) conditions.
    """

    def __init__(self, n_buildings: int = 5000, extent_m: float = 25000):
        self.meteo_model = MeteorologicalModel()
        self.rothermel = RothermelModel()
        self.embers_model = EmbersModel()
        self.transport_model = AtmosphericTransportModel()
        self.ignition_model = IgnitionModel()
        self.propagation_model = UrbanPropagationModel()
        self.graph = UrbanGraph(n_buildings=n_buildings, extent_m=extent_m)

        self.results: List[SimulationResult] = []

        # Real-data override modes
        self.fire_perimeter_source = None  # If set: use as ignition centroid instead of random
        self.meteorology_mode = 'sampled'  # 'sampled' (Weibull) or 'observed' (real wind vectors)
        self.observed_wind_vector = None  # If meteorology_mode='observed', use this (u, v)
        self.suppression_factor = 0.0  # 0.0 = no suppression, 0.7 = aggressive (2500 firefighters)

    def run_single_simulation(
        self,
        scenario: str = "average_summer",
        max_urban_timesteps: int = 100,
        fire_intensity_base: float = 5000
    ) -> SimulationResult:
        """
        One complete scenario: weather + wildfire + ember transport + urban spread.
        Supports both synthetic and real-time meteorology modes.
        """

        # Handle meteorology: sampled vs. observed
        if self.meteorology_mode == 'observed' and self.observed_wind_vector is not None:
            # Create synthetic meteo object with observed wind
            meteo = type('Meteo', (), {})()
            meteo.wind_speed = np.linalg.norm(self.observed_wind_vector)
            meteo.wind_direction = np.degrees(np.arctan2(self.observed_wind_vector[1], self.observed_wind_vector[0]))
            meteo.humidity = 0.5  # Average humidity
            meteo.temperature = 30.0  # Average summer temp
            meteo.fuel_moisture_dead_1h = 0.05  # Dry conditions
            meteo.fuel_moisture_dead_10h = 0.08
            meteo.fuel_moisture_dead_100h = 0.12
            meteo.fuel_moisture_live = 0.10
        else:
            # Standard sampled meteorology
            self.meteo_model.scenario = scenario
            meteo = self.meteo_model.sample()

        fire_chars = self.rothermel.characteristics(
            meteo=meteo,
            fuel_load_dead_1h=2.0,
            fuel_load_dead_10h=1.5,
            fuel_load_dead_100h=1.0,
            fuel_load_live=3.0,
            slope=0.1
        )

        fire_chars.fireline_intensity = max(fire_chars.fireline_intensity, fire_intensity_base)

        embers_generated = self.embers_model.generate_embers(
            fire_chars, meteo.wind_speed, area_ablaze_m2=5000
        )

        embers_landed = self.transport_model.transport_embers(
            n_embers=embers_generated.count,
            mass_dist=embers_generated.mass_distribution,
            diameter_dist=embers_generated.diameter_distribution,
            meteo=meteo,
            ember_model=self.embers_model
        )

        ember_density_field = self.transport_model.ember_density_field(
            embers_landed.x_landing,
            embers_landed.y_landing,
            embers_landed.mass_landing,
            grid_resolution=100.0
        )

        initial_ignitions: Set[int] = set()
        buildings_ignited_embers = 0

        # Real-data mode: seed from observed fire perimeter
        if self.fire_perimeter_source is not None:
            fire_lat, fire_lon = self.fire_perimeter_source
            # Find buildings near fire perimeter (within 5 km)
            for bid, building in self.graph.buildings.items():
                # Convert lat/lon to approximate grid coordinates (simplified)
                dist_to_fire = np.sqrt((building.x - fire_lat*1000)**2 + (building.y - fire_lon*1000)**2)
                if dist_to_fire < 5000:  # 5 km radius
                    initial_ignitions.add(bid)
                    buildings_ignited_embers += 1
        else:
            # Standard ember-transport mode
            for bid, building in self.graph.buildings.items():
                grid_x = building.x
                grid_y = building.y

                if len(embers_landed.x_landing) > 0:
                    min_dist_idx = np.argmin(
                        (embers_landed.x_landing - grid_x)**2 +
                        (embers_landed.y_landing - grid_y)**2
                    )
                    min_dist = np.sqrt(
                        (embers_landed.x_landing[min_dist_idx] - grid_x)**2 +
                        (embers_landed.y_landing[min_dist_idx] - grid_y)**2
                    )

                    if min_dist < 50:
                        local_ember_density = np.sum(
                            embers_landed.mass_landing[
                                np.sqrt((embers_landed.x_landing - grid_x)**2 +
                                       (embers_landed.y_landing - grid_y)**2) < 100
                            ]
                        ) / (100**2)

                        if self.ignition_model.building_ignites(
                            building, local_ember_density, 15.0
                        ):
                            initial_ignitions.add(bid)
                            buildings_ignited_embers += 1

        if len(initial_ignitions) == 0:
            return SimulationResult(
                simulation_id=-1,
                scenario=scenario,
                n_buildings_burned=0,
                burned_area_m2=0,
                max_extent_m=0,
                fire_duration_min=0,
                conflagration_occurred=False,
                initial_ember_count=embers_generated.count,
                buildings_ignited_by_embers=0
            )

        burning = initial_ignitions.copy()
        all_burned = initial_ignitions.copy()

        for step in range(max_urban_timesteps):
            if len(burning) == 0:
                break

            if len(all_burned) > self.graph.n_buildings * 0.15:
                conflagration = True
                break

            ignition_probs = self.propagation_model.step_urban_fire(
                self.graph, burning, all_burned - burning,
                meteo, fire_chars.flame_length, fire_chars.fireline_intensity
            )

            # Apply suppression factor: reduce ignition probabilities by (1 - suppression_factor)
            if self.suppression_factor > 0:
                for bid in ignition_probs:
                    ignition_probs[bid] *= (1.0 - self.suppression_factor)

            new_ignitions = self.propagation_model.burn_buildings(ignition_probs)

            if len(new_ignitions) == 0:
                break

            burning = new_ignitions.copy()
            all_burned.update(new_ignitions)

        conflagration_occurred = len(all_burned) > self.graph.n_buildings * 0.10

        burned_area = len(all_burned) * 250.0

        if len(all_burned) > 0:
            positions = np.array([
                [self.graph.buildings[bid].x, self.graph.buildings[bid].y]
                for bid in all_burned
            ])
            center = positions.mean(axis=0)
            max_extent = np.max(np.linalg.norm(positions - center, axis=1))
        else:
            max_extent = 0

        return SimulationResult(
            simulation_id=len(self.results),
            scenario=scenario,
            n_buildings_burned=len(all_burned),
            burned_area_m2=burned_area,
            max_extent_m=max_extent,
            fire_duration_min=len(all_burned),
            conflagration_occurred=conflagration_occurred,
            initial_ember_count=embers_generated.count,
            buildings_ignited_by_embers=buildings_ignited_embers
        )

    def run_ensemble(
        self,
        scenario: str = "average_summer",
        n_simulations: int = 500,
        save_results: bool = True
    ):
        """
        Run N Monte Carlo simulations, collect statistics.
        """

        print(f"\n=== MONTE CARLO ENSEMBLE: {n_simulations} sims ({scenario}) ===\n")

        for i in tqdm(range(n_simulations), desc=f"Simulating {scenario}"):
            result = self.run_single_simulation(scenario=scenario)
            self.results.append(result)

        if save_results:
            self.save_results(f"results_{scenario}.json")

        return self.compute_statistics()

    def compute_statistics(self) -> Dict:
        """
        Aggregate results from all simulations.
        """

        if not self.results:
            return {}

        n_buildings_all = np.array([r.n_buildings_burned for r in self.results])
        conflagration_flags = np.array([r.conflagration_occurred for r in self.results])

        return {
            "n_simulations": len(self.results),
            "mean_buildings_burned": float(np.mean(n_buildings_all)),
            "median_buildings_burned": float(np.median(n_buildings_all)),
            "std_buildings_burned": float(np.std(n_buildings_all)),
            "p05_buildings_burned": float(np.percentile(n_buildings_all, 5)),
            "p25_buildings_burned": float(np.percentile(n_buildings_all, 25)),
            "p75_buildings_burned": float(np.percentile(n_buildings_all, 75)),
            "p95_buildings_burned": float(np.percentile(n_buildings_all, 95)),
            "probability_conflagration": float(np.mean(conflagration_flags)),
            "max_buildings_burned": int(np.max(n_buildings_all)),
            "min_buildings_burned": int(np.min(n_buildings_all)),
            "mean_burned_area_hectares": float(np.mean([r.burned_area_m2 for r in self.results]) / 10000),
        }

    def save_results(self, filename: str):
        """Save all results to JSON."""
        def convert_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [convert_types(item) for item in obj]
            return obj

        results_data = [asdict(r) for r in self.results]
        results_data = [convert_types(r) for r in results_data]
        stats_data = convert_types(self.compute_statistics())

        data = {
            "results": results_data,
            "statistics": stats_data
        }

        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"Saved {len(self.results)} results to {filename}")
