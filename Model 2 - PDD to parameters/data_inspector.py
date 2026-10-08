# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 10:00:40 2026

@author: 45401
"""

import matplotlib.pyplot as plt
from data_loader import load_data


PDD_train, PDD_val, X_train, X_val = load_data()

# Plot training PDDs
for pdd, thickness in zip(PDD_train, X_train):
    plt.plot(pdd, label=f"{thickness:.2f} t0")

plt.xlabel("Depth bin")
plt.ylabel("Normalized dose")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()




#####
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


root_directory = Path(__file__).resolve().parent
data_dir = root_directory / "data"

parameters = pd.read_csv(data_dir / "simulation_parameters.csv")

# Find simulation closest to nominal thickness
nominal_thickness = 0.007620

idx = np.argmin(
    np.abs(parameters["primary_thickness_cm"] - nominal_thickness)
)

simulation_id = int(parameters.iloc[idx]["simulation_id"])
thickness = parameters.iloc[idx]["primary_thickness_cm"]

file = data_dir / f"Dose_{simulation_id:04d}.csv"

df = pd.read_csv(file, skiprows=7)
dose = df.iloc[:, -1].to_numpy()

print("Simulation:", simulation_id)
print("Thickness:", thickness, "cm")
print("Maximum dose bin:", np.argmax(dose))
print("Maximum dose:", dose.max())
print("First 10 dose values:")
print(dose[:10])

plt.plot(dose)
plt.xlabel("Depth bin")
plt.ylabel("Dose")
plt.grid(True)
plt.show()

max_doses = []
thicknesses = []

for _, row in parameters.iterrows():

    simulation_id = int(row["simulation_id"])
    thickness = row["primary_thickness_cm"]

    file = data_dir / f"Dose_{simulation_id:04d}.csv"

    df = pd.read_csv(file, skiprows=7)
    dose = df.iloc[:, -1].to_numpy()

    thicknesses.append(thickness)
    max_doses.append(dose.max())

plt.plot(thicknesses, max_doses, "o-")
plt.xlabel("Primary foil thickness [cm]")
plt.ylabel("Maximum dose")
plt.grid(True)
plt.show()