import numpy as np
from dataclasses import dataclass
from scipy.stats import lognorm, gamma, exponweib
from .rothermel import FirelineCharacteristics

@dataclass
class Ember:
    x: float
    y: float
    mass: float
    diameter: float
    temperature: float
    lifetime_remaining: float

@dataclass
class EmbersGenerated:
    count: int
    mass_distribution: np.ndarray
    diameter_distribution: np.ndarray

class EmbersModel:
    """
    Ember generation from fire intensity + wind.

    Calibration: CSIRO experiments + USDA Forest Service.
    Lognormal mass distribution, exponential lifetime.
    """

    def __init__(self):
        self.mass_scale_factor = 0.8
        self.diameter_mean_mm = 8.0
        self.diameter_std_mm = 3.5
        self.lifetime_base_seconds = 600
        self.temperature_emission = 850

    def number_of_embers(
        self,
        fireline_intensity: float,
        flame_length: float,
        wind_speed: float,
        area_ablaze_m2: float = 1000.0
    ) -> int:
        """
        Empirical: embers ∝ intensity^1.3 * wind^0.8

        From CSIRO data: ~100-500 embers/m² fireline/min.
        Typical wildfire front = 0.1-1 km × 50-100 m flame length.
        """

        if fireline_intensity < 100:
            return 0

        ember_flux = (
            self.mass_scale_factor *
            (fireline_intensity / 1000) ** 1.3 *
            (wind_speed + 1) ** 0.8
        )

        perimeter = max(flame_length * 0.5, 50)
        total_embers = int(ember_flux * perimeter * (wind_speed + 1) / 10)

        return np.clip(total_embers, 0, 100000)

    def generate_embers(
        self,
        fire_characteristics: FirelineCharacteristics,
        wind_speed: float,
        area_ablaze_m2: float = 1000.0
    ) -> EmbersGenerated:
        """
        Sample ember mass + diameter distributions.
        """

        n_embers = self.number_of_embers(
            fireline_intensity=fire_characteristics.fireline_intensity,
            flame_length=fire_characteristics.flame_length,
            wind_speed=wind_speed,
            area_ablaze_m2=area_ablaze_m2
        )

        if n_embers == 0:
            return EmbersGenerated(
                count=0,
                mass_distribution=np.array([]),
                diameter_distribution=np.array([])
            )

        mass_dist = lognorm.rvs(
            s=0.7,
            scale=0.5,
            size=n_embers
        )
        mass_dist = np.clip(mass_dist, 0.01, 5.0)

        diameter_dist = np.random.normal(
            loc=self.diameter_mean_mm,
            scale=self.diameter_std_mm,
            size=n_embers
        )
        diameter_dist = np.clip(diameter_dist, 1.0, 20.0)

        return EmbersGenerated(
            count=n_embers,
            mass_distribution=mass_dist,
            diameter_distribution=diameter_dist
        )

    def terminal_velocity(self, diameter_mm: float, mass_g: float) -> float:
        """
        Terminal fall velocity [m/s].

        Empirical correlation from NIST experiments.
        Roughly: v ~ sqrt(mass / drag_coeff).
        """

        if diameter_mm < 1:
            return 0.5
        if diameter_mm > 15:
            return 3.5

        v = 0.2 * np.sqrt(mass_g / (diameter_mm * 0.1))
        return np.clip(v, 0.3, 4.0)

    def combustion_lifetime(self, diameter_mm: float, mass_g: float) -> float:
        """
        Time [seconds] ember combusts before self-extinguishing.

        Exponential: τ ~ 10-15 min for 1-2 mm embers.
        Larger embers: longer life (quadratic with diameter).
        """

        base_lifetime = self.lifetime_base_seconds * (diameter_mm / 8.0) ** 1.5
        return np.clip(base_lifetime, 60, 1200)
