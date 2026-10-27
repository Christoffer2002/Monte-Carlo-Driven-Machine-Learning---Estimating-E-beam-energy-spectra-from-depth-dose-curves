#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 09:18:21 2026

@author: Christoffer
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from pathlib import Path


# ============================================================
# Setup
# ============================================================

root = Path(__file__).resolve().parent

template_file = root / "Main_template.txt"
simulation_folder = root / "FoilSimulations"
data_folder = root / "data"

simulation_folder.mkdir(exist_ok=True)
data_folder.mkdir(exist_ok=True)

# Set seed (script)
rng = np.random.default_rng(42)

# ============================================================
# Simulation parameters
# ============================================================
first_seed = 1001 # MC first seed

N_simulations = 1000 # MC sims
N_ref = 1 # 1 reference simulation with nominal thickness and energy
samples = N_simulations - N_ref

# Nominal values
nominal_thickness = 0.007620  # cm
nominal_energy = 13.3

# Parameter ranges
thickness_min = 0.5 * nominal_thickness
thickness_max = 1.5 * nominal_thickness

energy_min = 12.0  # MeV
energy_max = 14.0  # MeV

# Uniformly sample foil thickness and beam energy
#thicknesses = rng.uniform(thickness_min, thickness_max, N_simulations)
#energies = rng.uniform(energy_min, energy_max, N_simulations)

######################
# Latin Hypercube sampling
######################

# Sampling boundaries
thickness_bins = np.linspace(thickness_min, thickness_max, samples + 1)
energy_bins = np.linspace(energy_min, energy_max, samples + 1)


thicknesses = []
energies = []


for i in range(samples):
    # Draw random (and bound) thickness sample
    thickness = rng.uniform(thickness_bins[i], thickness_bins[i + 1])
    thicknesses.append(thickness)
    
    
    # Draw random (and bound) energy sample
    energy = rng.uniform(energy_bins[i], energy_bins[i + 1])
    energies.append(energy)


thicknesses = np.array(thicknesses)
energies = np.array(energies)

# Shuffle - avoid correlation -> low energy with thin foils
rng.shuffle(thicknesses)
rng.shuffle(energies)

# Insert ref sim at 0
thicknesses = np.insert(thicknesses, 0, nominal_thickness)
energies = np.insert(energies, 0, nominal_energy)

# ============================================================
# Load TOPAS template
# ============================================================

template = template_file.read_text()

# ============================================================
# Generate simulations
# ============================================================
simulation_parameters = []

for simulation_id, (thickness, energy) in enumerate(zip(thicknesses, energies), start=1):

    seed = first_seed + simulation_id - 1
    half_length = round(thickness / 2, 9)
    energy = round(energy, 6)
    simulation_id_string = f"{simulation_id:04d}"

    content = template.replace("{PRIMARY_FOIL_HL}", f"{half_length:.9f}")

    content = content.replace("{BEAM_ENERGY}", f"{energy:.6f}")

    content = content.replace("{SIMULATION_ID}", simulation_id_string)

    content = content.replace("{SEED}", str(seed))

    output_file = (simulation_folder / f"Main_{simulation_id_string}.txt")

    output_file.write_text(content)

    simulation_parameters.append({
        "simulation_id": simulation_id,
        "seed": seed,
        "primary_thickness_cm": thickness,
        "beam_energy_MeV": energy
    })

    # print(
    #     f"Simulation {simulation_id_string}: "
    #     f"thickness = {thickness:.6f} cm, "
    #     f"energy = {energy:.3f} MeV, "
    #     f"seed = {seed}"
    # )


# ============================================================
# Save parameter table
# ============================================================

parameters_df = pd.DataFrame(simulation_parameters)

parameters_df.to_csv(data_folder / "simulation_parameters.csv", index=False)

print(f"Created {N_simulations} TOPAS simulations.")


import matplotlib.pyplot as plt
# Plot to test LHS
plt.figure(figsize=(8, 6))
plt.scatter(thicknesses[1:], energies[1:], s=2, alpha=0.5, label="LHS samples")
plt.scatter(thicknesses[0], energies[0], color="red", marker="x", s=100, label="Reference")
plt.xlabel("Primary foil thickness (cm)")
plt.ylabel("Beam energy (MeV)")
plt.title("Latin Hypercube Sampling")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()