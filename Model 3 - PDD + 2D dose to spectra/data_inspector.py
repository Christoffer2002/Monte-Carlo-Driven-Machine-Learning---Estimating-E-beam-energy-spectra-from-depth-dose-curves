# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 10:00:40 2026

@author: Christoffer R. Olesen
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# Setup
# ============================================================

root = Path(__file__).resolve().parent
data_dir = root / "data"

simulation_id = "0001"


# ============================================================
# Load TOPAS data
# ============================================================

def load_topas_csv(filename):

    df = pd.read_csv(filename, comment="#", header=None)
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(how="all")

    return df


PDD = load_topas_csv(data_dir / f"PDD_{simulation_id}.csv")
Dose2D = load_topas_csv(data_dir / f"Dose2D_{simulation_id}.csv")
Fluence = load_topas_csv(data_dir / f"Fluence_{simulation_id}.csv")


# ============================================================
# Inspect data
# ============================================================

for name, df in [
    ("PDD", PDD),
    ("Dose2D", Dose2D),
    ("Fluence", Fluence)
]:
    print(f"\n{name}")
    print(f"Raw shape: {df.shape}")
    print(df.head())


# ============================================================
# Extract PDD
# ============================================================

pdd = PDD.iloc[:, -1].to_numpy()

# 90 bins of 1 mm
depth = np.arange(len(pdd)) + 0.5

print(f"\nPDD shape: {pdd.shape}")

# Find maximum dose and its depth
max_index = np.argmax(pdd)
max_dose = pdd[max_index]
max_depth = depth[max_index]

print(f"Maximum dose: {max_dose:.6f} Gy")
print(f"Depth of maximum dose (dmax): {max_depth:.1f} mm")

# ============================================================
# Extract 2D dose map
# ============================================================

def make_2d_map(df):

    # TOPAS columns: X bin, Y bin, Z bin, value
    x_bin = df.iloc[:, 0].to_numpy(dtype=int)
    z_bin = df.iloc[:, 2].to_numpy(dtype=int)
    values = df.iloc[:, -1].to_numpy(dtype=float)

    # Reconstruct using bin indices
    grid = np.full((100, 100), np.nan)
    grid[x_bin, z_bin] = values

    return grid


dose_map = make_2d_map(Dose2D)
fluence_map = make_2d_map(Fluence)

print(f"Dose2D shape: {dose_map.shape}")
print(f"Fluence shape: {fluence_map.shape}")

print(f"\nMissing Dose2D bins: {np.isnan(dose_map).sum()}")
print(f"Missing Fluence bins: {np.isnan(fluence_map).sum()}")


# ============================================================
# Plot PDD
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(depth, pdd)

plt.xlabel("Depth in water [mm]")
plt.ylabel("Dose to medium [Gy]")
plt.title(f"PDD - Simulation {simulation_id}")

plt.grid(True)
plt.tight_layout()
plt.show()


# ============================================================
# Plot 2D dose map
# ============================================================

plt.figure(figsize=(7, 6))

plt.imshow(
    dose_map.T,
    origin="lower",
    extent=[-50, 50, -50, 50],
    aspect="equal"
)

plt.colorbar(label="Dose to medium [Gy]")
plt.xlabel("X [mm]")
plt.ylabel("Z [mm]")
plt.title(f"2D Dose Map - Simulation {simulation_id}")

plt.tight_layout()
plt.show()


# ============================================================
# Plot fluence
# ============================================================

plt.figure(figsize=(7, 6))

plt.imshow(
    fluence_map.T,
    origin="lower",
    extent=[-50, 50, -50, 50],
    aspect="equal"
)

plt.colorbar(label="Fluence [1/mm²]")
plt.xlabel("X [mm]")
plt.ylabel("Z [mm]")
plt.title(f"Fluence - Simulation {simulation_id}")

plt.tight_layout()
plt.show()


# ============================================================
# Plot all PDDs
# ============================================================

plt.figure(figsize=(10, 6))

pdd_files = sorted(data_dir.glob("PDD_*.csv"))

for file in pdd_files:

    # Load PDD
    df = load_topas_csv(file)
    pdd = df.iloc[:, -1].to_numpy(dtype=float)

    # Normalize to maximum (same as dataloader)
    pdd = pdd / pdd.max()

    # Depth axis (1 mm bins)
    depth = np.arange(len(pdd)) + 0.5

    plt.plot(depth, pdd, alpha=0.3, color="steelblue")

plt.xlabel("Depth in water [mm]")
plt.ylabel("Normalized dose")
plt.title(f"All PDDs ({len(pdd_files)} simulations)")
plt.grid(True)
plt.tight_layout()
plt.show()

# ============================================================
# Compare 2D dose maps and fluence maps
# ============================================================

n_samples = 8

rng = np.random.default_rng(42)

dose_files = sorted(data_dir.glob("Dose2D_*.csv"))
simulation_ids = [file.stem.split("_")[-1] for file in dose_files]

selected_ids = rng.choice(
    simulation_ids,
    size=min(n_samples, len(simulation_ids)),
    replace=False
)

# 2 rows, n_samples columns
fig, axes = plt.subplots(
    2, len(selected_ids),
    figsize=(3 * len(selected_ids), 6),
    squeeze=False
)

for i, simulation_id in enumerate(selected_ids):

    dose_df = load_topas_csv(
        data_dir / f"Dose2D_{simulation_id}.csv"
    )

    fluence_df = load_topas_csv(
        data_dir / f"Fluence_{simulation_id}.csv"
    )

    dose_map = make_2d_map(dose_df)
    fluence_map = make_2d_map(fluence_df)

    dose_map = dose_map / np.nanmax(dose_map)
    fluence_map = fluence_map / np.nanmax(fluence_map)

    # First row: Dose2D
    axes[0, i].imshow(
        dose_map.T,
        origin="lower",
        extent=[-50, 50, -50, 50],
        cmap="viridis",
        vmin=0,
        vmax=1
    )

    # Second row: Fluence
    axes[1, i].imshow(
        fluence_map.T,
        origin="lower",
        extent=[-50, 50, -50, 50],
        cmap="viridis",
        vmin=0,
        vmax=1
    )

    axes[0, i].set_title(f"Sim {simulation_id}")

    for ax in axes[:, i]:
        ax.set_xlabel("X [mm]")
        ax.set_aspect("equal")

axes[0, 0].set_ylabel("Dose2D\nZ [mm]")
axes[1, 0].set_ylabel("Fluence\nZ [mm]")

plt.tight_layout()
plt.show()

# ============================================================
# Test dataloader
# ============================================================
# from data_loader import load_data

# (PDD_train, PDD_val,
#  Dose2D_train, Dose2D_val,
#  F_train, F_val) = load_data()

# print("PDD train:    ", PDD_train.shape)
# print("PDD val:      ", PDD_val.shape)

# print("Dose2D train: ", Dose2D_train.shape)
# print("Dose2D val:   ", Dose2D_val.shape)

# print("Fluence train:", F_train.shape)
# print("Fluence val:  ", F_val.shape)


# ============================================================
# Central dose profiles - Reference simulation
# ============================================================

# Reference parameters
nominal_thickness = 0.007620  # cm
nominal_energy = 13.3          # MeV

# Load simulation parameters
parameters = pd.read_csv(
    data_dir / "simulation_parameters.csv"
)

# Find simulation closest to reference parameters
parameters["distance"] = np.sqrt(
    ((parameters["primary_thickness_cm"] - nominal_thickness)
     / nominal_thickness)**2
    +
    ((parameters["beam_energy_MeV"] - nominal_energy)
     / nominal_energy)**2
)

closest = parameters.loc[parameters["distance"].idxmin()]
reference_id = f"{int(closest['simulation_id']):04d}"

print(f"\nClosest simulation: {reference_id}")
print(f"Foil thickness: {closest['primary_thickness_cm']:.6f} cm")
print(f"Beam energy: {closest['beam_energy_MeV']:.4f} MeV")


# ============================================================
# Load reference dose map
# ============================================================

reference_df = load_topas_csv(
    data_dir / f"Dose2D_{reference_id}.csv"
)

reference_dose = make_2d_map(reference_df)

# Normalize to maximum
#reference_dose = reference_dose / np.nanmax(reference_dose)


# ============================================================
# Extract central X and Z profiles
# ============================================================

nx, nz = reference_dose.shape

# Coordinates of bin centers in mm
x_coordinates = np.linspace(-50, 50, nx, endpoint=False) + 50/nx
z_coordinates = np.linspace(-50, 50, nz, endpoint=False) + 50/nz

# Average the two central bins (even-sized grid)
cx = nx // 2
cz = nz // 2

# X profile: vary X, hold Z at center
profile_x = np.mean(
    reference_dose[:, cz-1:cz+1],
    axis=1
)

# Z profile: vary Z, hold X at center
profile_z = np.mean(
    reference_dose[cx-1:cx+1, :],
    axis=0
)


# ============================================================
# Plot reference dose map and central X/Z profiles
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(17, 5))

# 2D dose map
im = axes[0].imshow(
    reference_dose.T,
    origin="lower",
    extent=[-50, 50, -50, 50],
    cmap="viridis",
    vmin=0,
    vmax=1
)

axes[0].axhline(0, color="red", linestyle="--", label="X profile")
axes[0].axvline(0, color="white", linestyle="--", label="Z profile")

axes[0].set_xlabel("X [mm]")
axes[0].set_ylabel("Z [mm]")
axes[0].set_title(f"2D Dose Map - Simulation {reference_id}")
axes[0].legend()
axes[0].set_aspect("equal")

fig.colorbar(im, ax=axes[0], label="Normalized dose")

# X profile
axes[1].plot(x_coordinates, profile_x, color="red")

axes[1].set_xlabel("X [mm]")
axes[1].set_ylabel("Normalized dose")
axes[1].set_title("Central X Profile")
axes[1].grid(True)

# Z profile
axes[2].plot(z_coordinates, profile_z, color="blue")

axes[2].set_xlabel("Z [mm]")
axes[2].set_ylabel("Normalized dose")
axes[2].set_title("Central Z Profile")
axes[2].grid(True)

plt.tight_layout()
plt.show()


# ============================================================
# Calculate flatness within central ±20 mm
# ============================================================

def calculate_flatness(profile, coordinates):

    region = np.abs(coordinates) <= 20
    values = profile[region]

    d_max = np.max(values)
    d_min = np.min(values)

    return 100 * (d_max - d_min) / (d_max + d_min)


flatness_x = calculate_flatness(profile_x, x_coordinates)
flatness_z = calculate_flatness(profile_z, z_coordinates)

print(f"\nFlatness within ±20 mm:")
print(f"X direction: {flatness_x:.2f}%")
print(f"Z direction: {flatness_z:.2f}%")



