# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 08:29:49 2026

@author: 45401
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

###############################################################################
# Uncertainty - 5 mil vs 50 mil histories
###############################################################################

root_directory = Path(__file__).resolve().parent
data_dir = root_directory / "data"

energies = [8, 10, 11, 12, 13, 14]

results = []

for energy in energies:

    # Load data
    df_5 = pd.read_csv(
        data_dir / f"differentHistories/Dose_{energy}MeV5mil.csv",
        skiprows=7
    )

    df_50 = pd.read_csv(
        data_dir / f"differentHistories/Dose_{energy}MeV50mil.csv",
        skiprows=7
    )

    PDD_5 = df_5.iloc[:, -1].to_numpy()
    PDD_50 = df_50.iloc[:, -1].to_numpy()

    # Normalize
    PDD_5 = PDD_5 / PDD_5.max()
    PDD_50 = PDD_50 / PDD_50.max()

    # Difference
    difference = PDD_5 - PDD_50

    # Metrics
    mse = np.mean(difference**2)
    mae = np.mean(np.abs(difference))
    max_diff = np.max(np.abs(difference))

    results.append([energy, mse, mae, max_diff])


results = pd.DataFrame(
    results,
    columns=[
        "Energy [MeV]",
        "MSE",
        "Mean absolute difference",
        "Maximum absolute difference"
    ]
)

print(results)




###############################################################################
# Uncertainty - 5 mil histories repeated
###############################################################################

root_directory = Path(__file__).resolve().parent
data_dir = root_directory / "data"

PDDs = []

for run in range(1, 6):

    file = data_dir / f"Uncertainty/Dose_11MeV5milRun{run}.csv"

    df = pd.read_csv(file, skiprows=7)
    PDD = df.iloc[:, -1].to_numpy()

    # Normalize
    PDD = PDD / PDD.max()

    PDDs.append(PDD)

PDDs = np.array(PDDs)

print(PDDs.shape)


mean_PDD = np.mean(PDDs, axis=0)
std_PDD = np.std(PDDs, axis=0, ddof=1)


plt.figure(figsize=(8, 5))

for i, PDD in enumerate(PDDs):
    plt.plot(PDD, label=f"Run {i+1}")

plt.xlabel("Depth bin")
plt.ylabel("Normalized dose")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()



plt.figure(figsize=(8, 5))

plt.plot(mean_PDD, label="Mean")

plt.fill_between(
    np.arange(len(mean_PDD)),
    mean_PDD - std_PDD,
    mean_PDD + std_PDD,
    alpha=0.3,
    label="±1 standard deviation"
)

plt.xlabel("Depth bin")
plt.ylabel("Normalized dose")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()


print("Mean standard deviation:", np.mean(std_PDD))
print("Maximum standard deviation:", np.max(std_PDD))