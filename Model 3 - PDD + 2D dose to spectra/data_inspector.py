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
