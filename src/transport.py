import numpy as np
from dataclasses import dataclass
from scipy.stats import norm
from .meteorology import MeteorologicalState
from .embers import Ember, EmbersModel

@dataclass
class EmbersLanding:
    x_landing: np.ndarray
    y_landing: np.ndarray
    mass_landing: np.ndarray
    n_landed: int

class AtmosphericTransportModel:
    """
    Trajectory equation for embers under wind + gravity + turbulence.

    Newtonian particle dynamics:
    m * dv/dt = F_drag + F_gravity + F_turbulence

    Simplified 2D: u-wind, v-wind, vertical drop.
    """

    def __init__(self):
        self.drag_coefficient = 0.47
        self.ambient_air_density = 1.225
        self.gravity = 9.81
        self.z_injection_height = 150.0
        self.turbulence_scale = 2.0

    def drag_force(
        self,
        diameter_m: float,
        velocity_rel_x: float,
        velocity_rel_y: float,
        velocity_rel_z: float
    ) -> tuple:
        """
        Drag = 0.5 * ρ * Cd * A * v²
        """

        area = np.pi * (diameter_m / 2) ** 2
        vel_mag = np.sqrt(velocity_rel_x**2 + velocity_rel_y**2 + velocity_rel_z**2)

        if vel_mag < 0.01:
            return 0.0, 0.0, 0.0

        f_magnitude = (
            0.5 * self.ambient_air_density * self.drag_coefficient * area * vel_mag ** 2
        )

        fx = -f_magnitude * (velocity_rel_x / vel_mag)
        fy = -f_magnitude * (velocity_rel_y / vel_mag)
        fz = -f_magnitude * (velocity_rel_z / vel_mag)

        return fx, fy, fz

    def propagate_ember(
        self,
        ember: Ember,
        meteo: MeteorologicalState,
        ember_model: EmbersModel,
        dt: float = 1.0,
        max_time: float = 1200.0
    ) -> tuple:
        """
        Integrate trajectory from injection to landing.

        Returns: (x_landing, y_landing, mass_landing, time_to_land)
        """

        u_wind, v_wind = meteo.wind_speed * np.cos(np.radians(meteo.wind_direction)), \
                         meteo.wind_speed * np.sin(np.radians(meteo.wind_direction))

        x, y, z = 0.0, 0.0, self.z_injection_height
        vx, vy, vz = 0.1, 0.1, -0.5

        diameter_m = ember.diameter / 1000.0
        mass_kg = ember.mass / 1000.0

        t = 0.0
        time_combust_remaining = ember_model.combustion_lifetime(ember.diameter, ember.mass)

        while z > 0 and t < max_time and time_combust_remaining > 0:
            u_wind_turb = u_wind + np.random.normal(0, self.turbulence_scale)
            v_wind_turb = v_wind + np.random.normal(0, self.turbulence_scale)

            vel_rel_x = vx - u_wind_turb
            vel_rel_y = vy - v_wind_turb
            vel_rel_z = vz

            fx, fy, fz = self.drag_force(
                diameter_m, vel_rel_x, vel_rel_y, vel_rel_z
            )

            ax = fx / mass_kg
            ay = fy / mass_kg
            az = fz / mass_kg - self.gravity

            vx += ax * dt
            vy += ay * dt
            vz += az * dt

            x += vx * dt
            y += vy * dt
            z += vz * dt

            if z < 0:
                z = 0.0
                break

            t += dt
            time_combust_remaining -= dt

        mass_landing = ember.mass if time_combust_remaining > 0 else 0

        return x, y, mass_landing, t

    def transport_embers(
        self,
        n_embers: int,
        mass_dist: np.ndarray,
        diameter_dist: np.ndarray,
        meteo: MeteorologicalState,
        ember_model: EmbersModel
    ) -> EmbersLanding:
        """
        Transport all embers and return landing distribution.
        """

        x_land_all = []
        y_land_all = []
        mass_land_all = []

        for i in range(min(n_embers, 5000)):
            ember = Ember(
                x=0.0, y=0.0,
                mass=mass_dist[i],
                diameter=diameter_dist[i],
                temperature=850,
                lifetime_remaining=600
            )

            x_land, y_land, mass_land, _ = self.propagate_ember(
                ember, meteo, ember_model
            )

            if mass_land > 0.01:
                x_land_all.append(x_land)
                y_land_all.append(y_land)
                mass_land_all.append(mass_land)

        if not x_land_all:
            return EmbersLanding(
                x_landing=np.array([]),
                y_landing=np.array([]),
                mass_landing=np.array([]),
                n_landed=0
            )

        return EmbersLanding(
            x_landing=np.array(x_land_all),
            y_landing=np.array(y_land_all),
            mass_landing=np.array(mass_land_all),
            n_landed=len(x_land_all)
        )

    def ember_density_field(
        self,
        x_landing: np.ndarray,
        y_landing: np.ndarray,
        mass_landing: np.ndarray,
        grid_resolution: float = 100.0,
        grid_extent: float = 50000.0
    ) -> dict:
        """
        Discretize ember landing on grid. Returns density [embers/m²].
        """

        if len(x_landing) == 0:
            return {"density_grid": np.array([]), "x_grid": np.array([]), "y_grid": np.array([])}

        x_edges = np.arange(-grid_extent, grid_extent, grid_resolution)
        y_edges = np.arange(-grid_extent, grid_extent, grid_resolution)

        h, xe, ye = np.histogram2d(
            x_landing, y_landing,
            bins=[x_edges, y_edges],
            weights=mass_landing
        )

        density = h / (grid_resolution ** 2)

        return {
            "density_grid": density,
            "x_grid": (x_edges[:-1] + x_edges[1:]) / 2,
            "y_grid": (y_edges[:-1] + y_edges[1:]) / 2,
            "x_edges": x_edges,
            "y_edges": y_edges
        }
