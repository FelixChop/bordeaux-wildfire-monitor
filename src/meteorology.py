import numpy as np
from dataclasses import dataclass
from scipy.stats import weibull_min, beta, norm
from typing import Tuple

@dataclass
class MeteorologicalState:
    wind_speed: float
    wind_direction: float
    air_temperature: float
    relative_humidity: float
    fuel_moisture_dead_1h: float
    fuel_moisture_dead_10h: float
    fuel_moisture_dead_100h: float
    fuel_moisture_live: float

class MeteorologicalModel:
    """
    Calibrated on Météo-France / ERA5 climate data.
    Extreme fire weather scenarios: summer 2022, heatwaves.
    """

    def __init__(self, scenario: str = "average_summer"):
        self.scenario = scenario
        self._init_parameters()

    def _init_parameters(self):
        """
        Calibration from Météo-France BDCLIM + ERA5 reanalysis.
        Estimates for Bordeaux region (44°N, 0.5°W).
        """
        if self.scenario == "average_summer":
            self.wind_speed_params = {"c": 1.8, "scale": 4.5}
            self.wind_dir_params = {"loc": 270, "scale": 60}
            self.temp_params = {"loc": 24, "scale": 3}
            self.humidity_params = {"a": 2.0, "b": 3.0, "loc": 20, "scale": 60}
            self.fuel_moisture_1h = {"loc": 0.08, "scale": 0.03}

        elif self.scenario == "heatwave":
            self.wind_speed_params = {"c": 2.1, "scale": 6.2}
            self.wind_dir_params = {"loc": 270, "scale": 45}
            self.temp_params = {"loc": 32, "scale": 2}
            self.humidity_params = {"a": 1.5, "b": 2.0, "loc": 15, "scale": 35}
            self.fuel_moisture_1h = {"loc": 0.04, "scale": 0.02}

        elif self.scenario == "extreme_drought":
            self.wind_speed_params = {"c": 2.3, "scale": 7.1}
            self.wind_dir_params = {"loc": 270, "scale": 40}
            self.temp_params = {"loc": 35, "scale": 3}
            self.humidity_params = {"a": 1.2, "b": 1.8, "loc": 10, "scale": 30}
            self.fuel_moisture_1h = {"loc": 0.02, "scale": 0.01}

        elif self.scenario == "landiras_2022":
            self.wind_speed_params = {"c": 2.5, "scale": 8.5}
            self.wind_dir_params = {"loc": 280, "scale": 35}
            self.temp_params = {"loc": 38, "scale": 2}
            self.humidity_params = {"a": 1.0, "b": 1.5, "loc": 8, "scale": 25}
            self.fuel_moisture_1h = {"loc": 0.015, "scale": 0.008}

    def sample(self) -> MeteorologicalState:
        """Sample random weather state."""
        ws = weibull_min.rvs(
            c=self.wind_speed_params["c"],
            scale=self.wind_speed_params["scale"]
        )
        ws = np.clip(ws, 0.5, 25)

        wd = norm.rvs(
            loc=self.wind_dir_params["loc"],
            scale=self.wind_dir_params["scale"]
        ) % 360

        temp = norm.rvs(
            loc=self.temp_params["loc"],
            scale=self.temp_params["scale"]
        )

        rh = beta.rvs(
            a=self.humidity_params["a"],
            b=self.humidity_params["b"],
            loc=self.humidity_params["loc"],
            scale=self.humidity_params["scale"]
        )
        rh = np.clip(rh, 5, 95)

        fm1h = np.clip(
            norm.rvs(
                loc=self.fuel_moisture_1h["loc"],
                scale=self.fuel_moisture_1h["scale"]
            ),
            0.01, 0.5
        )

        fm10h = fm1h * 1.3
        fm100h = fm1h * 1.6
        fml = np.clip(fm1h * 3, 0.5, 2.0)

        return MeteorologicalState(
            wind_speed=ws,
            wind_direction=wd,
            air_temperature=temp,
            relative_humidity=rh,
            fuel_moisture_dead_1h=fm1h,
            fuel_moisture_dead_10h=fm10h,
            fuel_moisture_dead_100h=fm100h,
            fuel_moisture_live=fml
        )

    def wind_vector(self, state: MeteorologicalState) -> Tuple[float, float]:
        """Convert wind speed + direction to (u, v) components."""
        rad = np.radians(state.wind_direction)
        u = state.wind_speed * np.cos(rad)
        v = state.wind_speed * np.sin(rad)
        return u, v
