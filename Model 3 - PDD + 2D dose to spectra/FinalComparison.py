# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 13:40:18 2026

@author: 45401
"""

from data_loader import load_data
import numpy as np
from tensorflow import keras
from pathlib import Path

root_directory = Path(__file__).resolve().parent

# Load data & model
(PDD_train, PDD_val, PDD_test, Dose2D_train, Dose2D_val, Dose2D_test, param_train, param_val, param_test) = load_data()
model = keras.models.load_model(root_directory / "models" / "final_model.keras")

# Find value in parameter space closest to center
center = np.array([0.5, 0.5, 0.5])
dist_center = np.linalg.norm(param_test - center, axis=1)
idx_center = np.argmin(dist_center)

# Low end of parameter space
low = np.array([0.0, 0.0, 0.0])

dist_low = np.linalg.norm(param_test - low, axis=1)
idx_low = np.argmin(dist_low)

# High end of parameter space
high = np.array([1.0, 1.0, 1.0])

dist_high = np.linalg.norm(param_test - high, axis=1)
idx_high = np.argmin(dist_high)



##########
# Ground truth
##########
# Selected simulations
selected_indices = [idx_low, idx_center, idx_high]

# Get selected normalized parameters
selected_params = param_test[selected_indices]

# Convert to physical values
thickness = 0.003810 + selected_params[:, 0] * (0.011430 - 0.003810)
energy = 12 + selected_params[:, 1] * 2
disk_scaling = 0.75 + selected_params[:, 2] * 0.5

# Convert disk scaling to actual radii
disk1_radius = 0.40005 * disk_scaling
disk2_radius = 0.49911 * disk_scaling

for i, name in enumerate(["Low", "Center", "High"]):
    print(f"\n{name} ground truth setup:")
    print(f"  Foil thickness: {thickness[i]:.6f} cm")
    print(f"  Beam energy:    {energy[i]:.4f} MeV")
    print(f"  Disk 1 radius:  {disk1_radius[i]:.6f} cm")
    print(f"  Disk 2 radius:  {disk2_radius[i]:.6f} cm")

##########
# Neural network
##########
# Retrieve PDD and Dosemap
PDD_selected = PDD_test[selected_indices]
Dose2D_selected = Dose2D_test[selected_indices]

# Predict
predicted_params = model.predict([PDD_selected, Dose2D_selected])

# Convert to physical values
thickness_pred = 0.003810 + predicted_params[:, 0] * (0.011430 - 0.003810)
energy_pred = 12 + predicted_params[:, 1] * 2
disk_scaling_pred = 0.75 + predicted_params[:, 2] * 0.5

# Convert predicted disk scaling to radii
disk1_radius_pred = 0.40005 * disk_scaling_pred
disk2_radius_pred = 0.49911 * disk_scaling_pred

for i, name in enumerate(["Low", "Center", "High"]):
    print(f"\n{name} predicted setup:")
    print(f"  Foil thickness: {thickness_pred[i]:.6f} cm")
    print(f"  Beam energy:    {energy_pred[i]:.4f} MeV")
    print(f"  Disk 1 radius:  {disk1_radius_pred[i]:.6f} cm")
    print(f"  Disk 2 radius:  {disk2_radius_pred[i]:.6f} cm")