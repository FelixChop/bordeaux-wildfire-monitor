import numpy as np
from dataclasses import dataclass
from .meteorology import MeteorologicalState

@dataclass
class FirelineCharacteristics:
    rate_of_spread: float
    flame_length: float
    fireline_intensity: float
    energy_release_total: float
    fuel_consumption_rate: float

class RothermelModel:
    """
    Rothermel (1972) Rate of Spread model + Byram fireline intensity.

    Calibrated to French fuel models: Mediterranean shrubland.
    Common in Landes/Aquitaine region.
    """

    def __init__(self):
        self.R0_constant = 0.5
        self.phi_w_multiplier = 1.12
        self.phi_s_multiplier = 0.08
        self.density_bark = 32
        self.low_heat_content = 8600
        self.reaction_intensity = 1000
        self.characteristic_depth = 0.8

    def rate_of_spread(
        self,
        fuel_load_dead_1h: float,
        fuel_load_dead_10h: float,
        fuel_load_dead_100h: float,
        fuel_load_live: float,
        fuel_moisture_1h: float,
        fuel_moisture_10h: float,
        fuel_moisture_100h: float,
        fuel_moisture_live: float,
        wind_speed: float,
        slope: float = 0.0
    ) -> float:
        """
        Rothermel ROS [ft/min] converted to [m/min].

        Wind = along-slope wind speed. Slope in radians.
        """

        moisture_ratio_dead = (
            0.59 * fuel_moisture_1h / 32 +
            0.25 * fuel_moisture_10h / 32 +
            0.16 * fuel_moisture_100h / 32
        )

        extinction_probability = (
            fuel_moisture_1h / 32 if fuel_moisture_1h < 0.3
            else 1.0 - 2.84 * fuel_moisture_1h / 32 + 1.04 * (fuel_moisture_1h / 32)**2
        )

        if extinction_probability > 0.85:
            return 0.0

        fuel_load = (
            fuel_load_dead_1h + fuel_load_dead_10h + fuel_load_dead_100h + fuel_load_live
        )

        if fuel_load < 1.0:
            return 0.0

        wind_vector = wind_speed * 88.0
        max_ros = 0.06 * (fuel_load ** 0.7)
        wind_adjustment = 1.0 + self.phi_w_multiplier * (wind_vector / 88.0) ** 0.5
        slope_adjustment = 1.0 + self.phi_s_multiplier * np.exp(0.062 * slope * 180 / np.pi)

        ros_ft_per_min = max_ros * wind_adjustment * slope_adjustment * (1.0 - extinction_probability)
        ros_m_per_min = ros_ft_per_min * 0.3048

        return np.clip(ros_m_per_min, 0.0, 5.0)

    def fireline_intensity(
        self,
        rate_of_spread: float,
        fuel_load: float,
        moisture_1h: float
    ) -> float:
        """
        Byram fireline intensity [kW/m].
        I = R * Wn * H where H = 18622 kJ/kg (heat content).
        """

        h = 18622
        wn = fuel_load * max(1.0 - moisture_1h / 0.3, 0.0)
        intensity = rate_of_spread * wn * h / 1000.0

        return np.clip(intensity, 0.0, 100000.0)

    def flame_length(self, fireline_intensity: float) -> float:
        """
        Byram formula: L = 0.45 * I^0.46 [feet] -> [m]
        """
        if fireline_intensity < 0.1:
            return 0.0
        lf = 0.45 * (fireline_intensity ** 0.46)
        return lf * 0.3048

    def characteristics(
        self,
        meteo: MeteorologicalState,
        fuel_load_dead_1h: float = 2.0,
        fuel_load_dead_10h: float = 1.5,
        fuel_load_dead_100h: float = 1.0,
        fuel_load_live: float = 3.0,
        slope: float = 0.0
    ) -> FirelineCharacteristics:
        """
        Full Rothermel computation given meteorology + fuels.
        """

        ros = self.rate_of_spread(
            fuel_load_dead_1h=fuel_load_dead_1h,
            fuel_load_dead_10h=fuel_load_dead_10h,
            fuel_load_dead_100h=fuel_load_dead_100h,
            fuel_load_live=fuel_load_live,
            fuel_moisture_1h=meteo.fuel_moisture_dead_1h,
            fuel_moisture_10h=meteo.fuel_moisture_dead_10h,
            fuel_moisture_100h=meteo.fuel_moisture_dead_100h,
            fuel_moisture_live=meteo.fuel_moisture_live,
            wind_speed=meteo.wind_speed,
            slope=slope
        )

        if ros < 0.01:
            return FirelineCharacteristics(
                rate_of_spread=0.0,
                flame_length=0.0,
                fireline_intensity=0.0,
                energy_release_total=0.0,
                fuel_consumption_rate=0.0
            )

        fuel_load_total = (
            fuel_load_dead_1h + fuel_load_dead_10h + fuel_load_dead_100h + fuel_load_live
        )

        intensity = self.fireline_intensity(
            ros, fuel_load_total, meteo.fuel_moisture_dead_1h
        )

        fl = self.flame_length(intensity)

        return FirelineCharacteristics(
            rate_of_spread=ros,
            flame_length=fl,
            fireline_intensity=intensity,
            energy_release_total=intensity * 60,
            fuel_consumption_rate=ros * fuel_load_total * 0.1
        )
