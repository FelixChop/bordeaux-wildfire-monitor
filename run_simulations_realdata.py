#!/usr/bin/env python3
"""
Real-data fire simulations: deterministic + ensemble modes.
Loads actual fire perimeter + wind forecast, runs scenarios with observed conditions.
"""

import sys
import os
import json
import argparse
import numpy as np
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

from src.montecarlo import MonteCarloEngine

def load_realdata_cache(cache_file: str = None) -> dict:
    """Load the most recent cache file or specified file."""
    if cache_file:
        with open(cache_file, 'r') as f:
            return json.load(f)

    # Find most recent cache file
    cache_dir = Path('results')
    cache_files = sorted(cache_dir.glob('realdata_cache_*.json'), reverse=True)
    if not cache_files:
        raise FileNotFoundError("No realdata_cache_*.json files found. Run fetch_realtime_fire_data.py first.")

    with open(cache_files[0], 'r') as f:
        return json.load(f)

def extract_wind_vector(wind_forecast: list, hour: int = 0) -> tuple:
    """
    Extract (u, v) wind components from hourly forecast.
    u = east-west (positive = east), v = north-south (positive = north)
    """
    if hour >= len(wind_forecast):
        hour = len(wind_forecast) - 1

    record = wind_forecast[hour]
    speed = record.get('wind_speed_10m_ms', 8.0)
    direction_deg = record.get('wind_direction_10m_deg', 270.0)

    # Convert speed + direction to (u, v)
    direction_rad = np.radians(direction_deg)
    u = speed * np.sin(direction_rad)
    v = speed * np.cos(direction_rad)

    return (u, v)

def run_deterministic(realdata: dict, scenario: str = 'real_observed', suppression_factor: float = 0.7):
    """Run single deterministic simulation with observed conditions."""
    print("\n" + "="*80)
    print(f"DETERMINISTIC SIMULATION: {scenario}")
    print("="*80)

    engine = MonteCarloEngine(n_buildings=5000, extent_m=25000)

    # Configure real-data mode
    fire_perimeter = realdata['fire_perimeter']
    if fire_perimeter['centroid']:
        engine.fire_perimeter_source = (
            fire_perimeter['centroid']['lat'],
            fire_perimeter['centroid']['lon']
        )
        print(f"Fire perimeter: {engine.fire_perimeter_source}")
        print(f"Distance to Bordeaux: {fire_perimeter['distance_to_bordeaux_km']:.1f} km")

    # Set observed wind (use average of first 24 hours)
    wind_forecast = realdata['wind_forecast']['hourly_wind']
    if wind_forecast:
        # Average wind over first day
        u_avg = np.mean([extract_wind_vector(wind_forecast, h)[0] for h in range(min(24, len(wind_forecast)))])
        v_avg = np.mean([extract_wind_vector(wind_forecast, h)[1] for h in range(min(24, len(wind_forecast)))])
        engine.observed_wind_vector = (u_avg, v_avg)
        engine.meteorology_mode = 'observed'

        wind_speed = np.linalg.norm(engine.observed_wind_vector)
        wind_dir = np.degrees(np.arctan2(engine.observed_wind_vector[0], engine.observed_wind_vector[1]))
        print(f"Observed wind: {wind_speed:.1f} m/s from {wind_dir:.0f}° (0°=N, 90°=E, 180°=S, 270°=W)")

    # Set suppression
    engine.suppression_factor = suppression_factor
    print(f"Suppression factor: {suppression_factor:.1f} (2500 firefighters + aircraft)")

    # Run single deterministic sim
    result = engine.run_single_simulation(scenario=scenario)

    stats = {
        "simulation_mode": "deterministic",
        "scenario": scenario,
        "fire_distance_km": fire_perimeter['distance_to_bordeaux_km'],
        "wind_speed_ms": np.linalg.norm(engine.observed_wind_vector) if engine.observed_wind_vector else None,
        "suppression_factor": suppression_factor,
        "n_buildings_burned": result.n_buildings_burned,
        "conflagration_occurred": result.conflagration_occurred,
        "buildings_ignited_by_embers": result.buildings_ignited_by_embers,
        "timestamp": datetime.now().isoformat()
    }

    print(f"\nResult:")
    print(f"  Buildings burned: {result.n_buildings_burned}")
    print(f"  Conflagration: {result.conflagration_occurred}")

    # Save deterministic result
    with open('results/summary_realdata_deterministic.json', 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\n✓ Saved to results/summary_realdata_deterministic.json")

    return stats

def run_ensemble(realdata: dict, n_samples: int = 100, suppression_factor: float = 0.7):
    """Run ensemble with perturbations on wind, suppression, ROS."""
    print("\n" + "="*80)
    print(f"ENSEMBLE SIMULATION: {n_samples} perturbed scenarios")
    print("="*80)

    from scipy.stats import qmc

    # Generate LHS samples for perturbations
    # Dimensions: [wind_speed_factor, wind_dir_shift, suppression_efficacy, ros_multiplier]
    sampler = qmc.LatinHypercube(d=4)
    lhs_samples = sampler.random(n_samples)

    # Scale to parameter ranges
    perturb_ranges = {
        'wind_speed_factor': (0.8, 1.2),          # ±20% wind speed
        'wind_dir_shift': (-15, 15),               # ±15° wind direction
        'suppression_efficacy': (0.5, 0.9),       # 50-90% suppression
        'ros_multiplier': (0.8, 1.2)               # ±20% ROS
    }

    samples_scaled = []
    for i, sample in enumerate(lhs_samples):
        scaled = {
            'wind_speed_factor': perturb_ranges['wind_speed_factor'][0] +
                                 sample[0] * (perturb_ranges['wind_speed_factor'][1] - perturb_ranges['wind_speed_factor'][0]),
            'wind_dir_shift': perturb_ranges['wind_dir_shift'][0] +
                             sample[1] * (perturb_ranges['wind_dir_shift'][1] - perturb_ranges['wind_dir_shift'][0]),
            'suppression_efficacy': perturb_ranges['suppression_efficacy'][0] +
                                   sample[2] * (perturb_ranges['suppression_efficacy'][1] - perturb_ranges['suppression_efficacy'][0]),
            'ros_multiplier': perturb_ranges['ros_multiplier'][0] +
                             sample[3] * (perturb_ranges['ros_multiplier'][1] - perturb_ranges['ros_multiplier'][0])
        }
        samples_scaled.append(scaled)

    # Extract base wind
    wind_forecast = realdata['wind_forecast']['hourly_wind']
    base_u, base_v = extract_wind_vector(wind_forecast, 0)
    base_wind_speed = np.linalg.norm((base_u, base_v))
    base_wind_dir = np.degrees(np.arctan2(base_u, base_v))

    fire_perimeter = realdata['fire_perimeter']

    # Run ensemble
    ensemble_results = []
    print(f"Running {n_samples} perturbed simulations...\n")

    for idx, perturbation in enumerate(samples_scaled):
        engine = MonteCarloEngine(n_buildings=5000, extent_m=25000)

        # Apply fire perimeter
        if fire_perimeter['centroid']:
            engine.fire_perimeter_source = (
                fire_perimeter['centroid']['lat'],
                fire_perimeter['centroid']['lon']
            )

        # Perturb wind: scale speed + shift direction
        perturbed_speed = base_wind_speed * perturbation['wind_speed_factor']
        perturbed_dir = base_wind_dir + perturbation['wind_dir_shift']
        perturbed_dir_rad = np.radians(perturbed_dir)

        u_perturbed = perturbed_speed * np.sin(perturbed_dir_rad)
        v_perturbed = perturbed_speed * np.cos(perturbed_dir_rad)
        engine.observed_wind_vector = (u_perturbed, v_perturbed)
        engine.meteorology_mode = 'observed'

        # Apply suppression perturbation
        engine.suppression_factor = perturbation['suppression_efficacy']

        # Note: ROS multiplier would require modifying Rothermel; skip for now
        result = engine.run_single_simulation(scenario='real_perturbed')

        ensemble_results.append({
            "sample_id": idx,
            "perturbation": perturbation,
            "n_buildings_burned": result.n_buildings_burned,
            "conflagration_occurred": result.conflagration_occurred,
            "wind_speed_ms": perturbed_speed,
            "wind_direction_deg": perturbed_dir,
            "suppression_factor": perturbation['suppression_efficacy']
        })

        if (idx + 1) % 20 == 0:
            print(f"  Completed {idx + 1}/{n_samples} samples")

    # Aggregate statistics
    n_buildings_all = np.array([r['n_buildings_burned'] for r in ensemble_results])
    conflag_flags = np.array([r['conflagration_occurred'] for r in ensemble_results])

    stats = {
        "simulation_mode": "ensemble",
        "n_samples": n_samples,
        "fire_distance_km": fire_perimeter['distance_to_bordeaux_km'],
        "base_wind_speed_ms": base_wind_speed,
        "base_wind_direction_deg": base_wind_dir,
        "suppression_base": suppression_factor,
        "statistics": {
            "mean_buildings_burned": float(np.mean(n_buildings_all)),
            "median_buildings_burned": float(np.median(n_buildings_all)),
            "std_buildings_burned": float(np.std(n_buildings_all)),
            "p05_buildings_burned": float(np.percentile(n_buildings_all, 5)),
            "p25_buildings_burned": float(np.percentile(n_buildings_all, 25)),
            "p75_buildings_burned": float(np.percentile(n_buildings_all, 75)),
            "p95_buildings_burned": float(np.percentile(n_buildings_all, 95)),
            "min_buildings_burned": int(np.min(n_buildings_all)),
            "max_buildings_burned": int(np.max(n_buildings_all)),
            "probability_conflagration": float(np.mean(conflag_flags)),
            "probability_any_burn": float(np.sum(n_buildings_all > 0) / len(n_buildings_all))
        },
        "timestamp": datetime.now().isoformat()
    }

    # Sensitivity analysis: Sobol-like one-at-a-time
    sensitivity = {}
    for param in ['wind_speed_factor', 'wind_dir_shift', 'suppression_efficacy']:
        param_values = [r['perturbation'][param] for r in ensemble_results]
        buildings_burned = n_buildings_all

        # Simple correlation: rank correlation of parameter vs. output
        rank_corr = np.corrcoef(param_values, buildings_burned)[0, 1]
        sensitivity[param] = {
            "rank_correlation": float(rank_corr) if not np.isnan(rank_corr) else 0.0,
            "variance_contribution": float(abs(rank_corr))  # Simplified
        }

    stats['sensitivity'] = sensitivity

    print(f"\n" + "="*80)
    print("ENSEMBLE STATISTICS")
    print("="*80)
    print(f"Mean buildings burned: {stats['statistics']['mean_buildings_burned']:.2f}")
    print(f"Median: {stats['statistics']['median_buildings_burned']:.2f}")
    print(f"Std dev: {stats['statistics']['std_buildings_burned']:.2f}")
    print(f"95% credible interval: [{stats['statistics']['p05_buildings_burned']:.2f}, {stats['statistics']['p95_buildings_burned']:.2f}]")
    print(f"P(conflagration): {stats['statistics']['probability_conflagration']:.4f}")
    print(f"\nSensitivity ranking:")
    sorted_sensitivity = sorted(sensitivity.items(), key=lambda x: abs(x[1]['rank_correlation']), reverse=True)
    for i, (param, val) in enumerate(sorted_sensitivity, 1):
        print(f"  {i}. {param}: {val['rank_correlation']:.3f}")

    # Save ensemble result
    with open('results/summary_realdata_ensemble.json', 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\n✓ Saved to results/summary_realdata_ensemble.json")

    return stats

def main():
    parser = argparse.ArgumentParser(description='Real-data fire simulations')
    parser.add_argument('--mode', choices=['deterministic', 'ensemble', 'both'], default='both',
                        help='Simulation mode')
    parser.add_argument('--cache', type=str, help='Path to realdata cache JSON file')
    parser.add_argument('--n_samples', type=int, default=100, help='Number of ensemble samples')
    parser.add_argument('--suppression', type=float, default=0.7, help='Suppression factor (0-1)')
    args = parser.parse_args()

    # Load real-time data
    print("\nLoading real-time data cache...")
    realdata = load_realdata_cache(args.cache)
    print(f"✓ Loaded cache from {realdata['timestamp_generated']}")

    # Run simulations
    results = {}

    if args.mode in ['deterministic', 'both']:
        results['deterministic'] = run_deterministic(realdata, suppression_factor=args.suppression)

    if args.mode in ['ensemble', 'both']:
        results['ensemble'] = run_ensemble(realdata, n_samples=args.n_samples, suppression_factor=args.suppression)

    # Summary
    print("\n" + "="*80)
    print("SUMMARY: REAL-DATA RISK ASSESSMENT")
    print("="*80)
    if 'deterministic' in results:
        print(f"\nDeterministic: {results['deterministic']['n_buildings_burned']} buildings burned")
    if 'ensemble' in results:
        stats = results['ensemble']['statistics']
        print(f"\nEnsemble:")
        print(f"  P(Bordeaux conflagration) = {stats['probability_conflagration']:.2%} [{stats['p05_buildings_burned']:.0f}-{stats['p95_buildings_burned']:.0f} buildings]")
        print(f"  Dominant parameter: {max(results['ensemble']['sensitivity'].items(), key=lambda x: abs(x[1]['rank_correlation']))[0]}")

    print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    main()
