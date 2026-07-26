import numpy as np
from scipy.special import expit
from .buildings import Building

class IgnitionModel:
    """
    Probabilistic model for building ignition from embers + radiation.

    Calibration: NIST fire experiments + SFPE Handbook + post-wildfire surveys.
    Main mechanisms:
    - Ember accumulation on roof
    - Convective heating
    - Radiative heating
    - Ignition delay (time to ignite after exposure)
    """

    def __init__(self):
        self.ember_density_threshold = 0.5
        self.radiation_threshold = 12.5
        self.time_to_ignition_mean = 300.0
        self.time_to_ignition_std = 150.0

    def ignition_probability_embers(
        self,
        ember_density: float,
        roof_type: str,
        roof_age_years: float
    ) -> float:
        """
        P(ignition | ember flux).

        Roof type combustibility: tile < concrete < metal < asphalt.
        Older roofs more vulnerable (bitumen degradation).
        """

        roof_combustibility = {
            "tile": 0.1,
            "concrete": 0.3,
            "metal": 0.25,
            "asphalt": 0.7
        }

        base_combustibility = roof_combustibility.get(roof_type, 0.4)
        age_factor = 1.0 + 0.01 * max(roof_age_years - 20, 0)
        combustibility = base_combustibility * age_factor

        if ember_density < self.ember_density_threshold:
            return 0.0

        logit_p = -5 + 2.5 * np.log(1 + ember_density) + 3.0 * combustibility
        p = expit(logit_p)

        return np.clip(p, 0.0, 1.0)

    def ignition_probability_radiation(
        self,
        radiative_heat_flux: float,
        window_fraction: float,
        construction_year: int
    ) -> float:
        """
        P(ignition | radiative flux).

        Larger window fraction = more interior exposure.
        Newer buildings = better glazing, less ignition risk.
        """

        if radiative_heat_flux < self.radiation_threshold:
            return 0.0

        interior_accessibility = window_fraction * (1.0 + 0.001 * max(2000 - construction_year, 0))

        logit_p = -8 + 0.4 * radiative_heat_flux + 5.0 * interior_accessibility
        p = expit(logit_p)

        return np.clip(p, 0.0, 1.0)

    def ignition_probability_combined(
        self,
        building: Building,
        ember_density: float,
        radiative_heat_flux: float,
        current_year: int = 2026
    ) -> float:
        """
        Combined ignition probability. Use envelope (max) or product?
        Here: envelope (union of mechanisms).
        """

        roof_age = current_year - building.construction_year

        p_embers = self.ignition_probability_embers(
            ember_density, building.roof_type, roof_age
        )

        p_radiation = self.ignition_probability_radiation(
            radiative_heat_flux, building.window_fraction, building.construction_year
        )

        p_combined = p_embers + p_radiation - p_embers * p_radiation

        return np.clip(p_combined, 0.0, 1.0)

    def time_to_ignition(self) -> float:
        """
        Random ignition delay [seconds].

        Log-normal: fast ignitions (5-10 min) + tail up to 30 min.
        """

        t = np.random.normal(
            self.time_to_ignition_mean,
            self.time_to_ignition_std
        )
        return np.clip(t, 30, 1200)

    def building_ignites(
        self,
        building: Building,
        ember_density: float,
        radiative_flux: float,
        current_year: int = 2026
    ) -> bool:
        """
        Bernoulli trial: does building ignite?
        """

        p_ign = self.ignition_probability_combined(
            building, ember_density, radiative_flux, current_year
        )

        return np.random.random() < p_ign
