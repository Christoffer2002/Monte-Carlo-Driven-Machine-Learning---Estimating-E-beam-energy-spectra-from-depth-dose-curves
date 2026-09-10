#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 14:33:47 2026

@author: Christoffer R. Olesen
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Number of depth dose curves
N_PDD = 10000
max_monoenergic = 14
min_monoenergic = 8

# Directory setup
root_directory = Path(__file__).resolve().parent

data_dir = root_directory / "data" / "MC-simulated-PDDs"

csv_files = list(data_dir.glob("*.csv"))


# Sort by energy in filename
csv_files.sort(key=lambda file: int(file.stem.split("_")[1].replace("MeV", "")))

r = []

# Load all files
for file in csv_files:
    df = pd.read_csv(file, skiprows=7)
    
    depth_dose = df.iloc[:, -1].to_numpy()
    
    r.append(depth_dose)

# Combine into NumPy array - r matrix with shape(depth bins, monoenergic beams)
r = np.column_stack(r)

# Amount of monoenergic beams from MC simulations
N_energies = r.shape[1]




def generate_spectra(N_PDD, min_energy, max_energy):
    """
    generate_spectra generates energy spectra sampled from a gaussian
    
    """
    
    energies = np.arange(min_energy, max_energy + 1)

    X = []

    for _ in range(N_PDD):

        # Random centre of spectrum
        mu = np.random.uniform(min_energy, max_energy)

        # Random width
        sigma = np.random.uniform(0.7, 2.0)

        spectrum = np.exp(-(energies - mu)**2 / (2 * sigma**2)
        )

        # Normalize area/sum to 1
        spectrum = spectrum / np.sum(spectrum)

        X.append(spectrum)

    X = np.array(X)

    return np.array(X)

def generate_spectra2(N_PDD, min_energy, max_energy):

    energies = np.arange(min_energy, max_energy + 1)

    X = []

    for _ in range(N_PDD):

        # First Gaussian
        mu1 = np.random.uniform(min_energy, max_energy)
        sigma1 = np.random.uniform(0.5, 2.0)

        gaussian1 = np.exp(
            -(energies - mu1)**2 / (2 * sigma1**2)
        )

        # Second Gaussian
        mu2 = np.random.uniform(min_energy, max_energy)
        sigma2 = np.random.uniform(0.5, 2.0)

        gaussian2 = np.exp(
            -(energies - mu2)**2 / (2 * sigma2**2)
        )

        # Relative contribution of the two components
        weight = np.random.uniform(0.2, 0.8)

        spectrum = (
            weight * gaussian1
            + (1 - weight) * gaussian2
        )

        # Normalize
        spectrum = spectrum / np.sum(spectrum)

        X.append(spectrum)

    return np.array(X)

def generate_realistic_electron_spectrum():
    
    
    return

#Create PDDs
# X = np.random.rand(N_energies, N_PDD)
# X = X / np.sum(X, axis=0, keepdims=True)

#Generate energy spectra
X = generate_spectra2(N_PDD, min_monoenergic, max_monoenergic)

# Transpose for r @ X
X = X.T


#Generate PDD from discrete Fredholm equation
PDDs = r @ X

# Transpose so each row becomes a sample
PDDs = PDDs.T
X = X.T

# Save data
np.save(root_directory / "data/PDDs.npy", PDDs)
np.save(root_directory / "data/Spectra.npy", X)


# -------------------------------------------------------
# Plot monoenergic depth dose curves
# -------------------------------------------------------
depth = np.arange(r.shape[0])

plt.figure(figsize=(8, 5))

for i in range(r.shape[1]):
    plt.plot(depth, r[:, i], label=f"Energy bin {i}")

plt.xlabel("Depth bin")
plt.ylabel("Dose")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()