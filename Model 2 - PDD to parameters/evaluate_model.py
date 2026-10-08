# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 12:12:24 2026

@author: Christoffer R. Olesen
"""
import numpy as np
from tensorflow import keras
from pathlib import Path
from data_loader import load_data
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error


# ============================================================
# Setup
# ============================================================

root_directory = Path(__file__).resolve().parent

nominal_thickness = 0.007620  # cm


# ============================================================
# Load model and data
# ============================================================

# model = keras.models.load_model(
#     root_directory / "models" / "foil_thickness_model.keras")
model = keras.models.load_model(
    root_directory / "models" / "Small_model_no_norm_foil_thickness_model.keras")


PDD_train, PDD_val, X_train, X_val = load_data()


# ============================================================
# Average predictor
# ============================================================

# Average normalized thickness in training set
X_mean = np.mean(X_train)

# Predict same thickness for every validation sample
X_pred_mean = np.full_like(X_val, X_mean)

mse_mean = mean_squared_error(
    X_val,
    X_pred_mean
)


# ============================================================
# Nearest neighbor
# ============================================================

knn = KNeighborsRegressor(n_neighbors=1)

knn.fit(
    PDD_train,
    X_train
)

X_pred_nn = knn.predict(PDD_val)

mse_nn = mean_squared_error(
    X_val,
    X_pred_nn
)


# ============================================================
# Neural network
# ============================================================

X_pred_network = model.predict(PDD_val).flatten()

mse_network = mean_squared_error(
    X_val,
    X_pred_network
)


# ============================================================
# Overall evaluation
# ============================================================

print("MSE")
print("Average predictor: ", mse_mean)
print("Nearest neighbor:  ", mse_nn)
print("Neural network:    ", mse_network)


# ============================================================
# Convert thicknesses to micrometers
# ============================================================

# X contains t/t0
# t [cm] = X * t0
# 1 cm = 10,000 µm

X_val_um = X_val * nominal_thickness * 10000
X_pred_mean_um = X_pred_mean * nominal_thickness * 10000
X_pred_nn_um = X_pred_nn * nominal_thickness * 10000
X_pred_network_um = X_pred_network * nominal_thickness * 10000


# ============================================================
# Print individual predictions
# ============================================================

# print("\nPredictions:")

# for i in range(len(X_val)):

#     print(
#         f"Sample {i}: "
#         f"True = {X_val_um[i]:.2f} µm, "
#         f"NN = {X_pred_network_um[i]:.2f} µm, "
#         f"Nearest = {X_pred_nn_um[i]:.2f} µm, "
#         f"Average = {X_pred_mean_um[i]:.2f} µm"
#     )


# ============================================================
# True vs predicted thickness
# ============================================================

plt.figure(figsize=(7, 6))

plt.scatter(
    X_val_um,
    X_pred_network_um,
    label="Neural network"
)

plt.scatter(
    X_val_um,
    X_pred_nn_um,
    label="Nearest neighbor"
)

plt.scatter(
    X_val_um,
    X_pred_mean_um,
    label="Average predictor"
)

# Perfect prediction line
minimum = X_val_um.min()
maximum = X_val_um.max()

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    "--",
    label="Perfect prediction"
)

plt.xlabel("True foil thickness [µm]")
plt.ylabel("Predicted foil thickness [µm]")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# ============================================================
# Absolute error for each validation sample
# ============================================================

error_mean = np.abs(
    X_pred_mean_um - X_val_um
)

error_nn = np.abs(
    X_pred_nn_um - X_val_um
)

error_network = np.abs(
    X_pred_network_um - X_val_um
)


print("\nMean absolute error [µm]")
print("Average predictor: ", np.mean(error_mean))
print("Nearest neighbor:  ", np.mean(error_nn))
print("Neural network:    ", np.mean(error_network))


# ============================================================
# Plot error for each validation sample
# ============================================================

sample_indices = np.arange(len(X_val))

plt.figure(figsize=(8, 5))

plt.plot(
    sample_indices,
    error_mean,
    marker="o",
    label="Average predictor"
)

plt.plot(
    sample_indices,
    error_nn,
    marker="o",
    label="Nearest neighbor"
)

plt.plot(
    sample_indices,
    error_network,
    marker="o",
    label="Neural network"
)

plt.xlabel("Validation sample")
plt.ylabel("Absolute thickness error [µm]")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()