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

# Setup
root_directory = Path(__file__).resolve().parent

# Load trained model
model = keras.models.load_model(root_directory / "models" / "fluence_model.keras")

# Load data
(PDD_train, PDD_val, Dose2D_train, Dose2D_val, F_train, F_val) = load_data()


###############################################################################
# Average predictor
###############################################################################
# Average fluence map from training set
F_mean = np.mean(F_train, axis=0)

# Predict same map for every validation sample
F_pred_mean = np.repeat( F_mean[np.newaxis, :, :], F_val.shape[0], axis=0)

mse_mean = np.mean((F_pred_mean - F_val)**2)

###############################################################################
# Nearest neighbor predictor
###############################################################################

# Combine PDD and Dose2D into feature vectors
PDD_train_flat = PDD_train.reshape(len(PDD_train), -1)
PDD_val_flat = PDD_val.reshape(len(PDD_val), -1)

Dose_train_flat = Dose2D_train.reshape(len(Dose2D_train), -1)
Dose_val_flat = Dose2D_val.reshape(len(Dose2D_val), -1)

# Scale each modality so the 10000 dose pixels do not
# automatically dominate the 90 PDD values
PDD_train_flat = PDD_train_flat / np.sqrt(PDD_train_flat.shape[1])
PDD_val_flat = PDD_val_flat / np.sqrt(PDD_val_flat.shape[1])

Dose_train_flat = Dose_train_flat / np.sqrt(Dose_train_flat.shape[1])
Dose_val_flat = Dose_val_flat / np.sqrt(Dose_val_flat.shape[1])

X_train = np.concatenate(
    [PDD_train_flat, Dose_train_flat], axis=1
)

X_val = np.concatenate(
    [PDD_val_flat, Dose_val_flat], axis=1
)

# Flatten fluence targets for sklearn
F_train_flat = F_train.reshape(len(F_train), -1)

# 1-nearest-neighbor
knn = KNeighborsRegressor(n_neighbors=1)

knn.fit(X_train, F_train_flat)

F_pred_nn = knn.predict(X_val)

# Reshape predictions into 2D maps
F_pred_nn = F_pred_nn.reshape(F_val.shape)

mse_nn = np.mean((F_pred_nn - F_val)**2)


###############################################################################
# Neural network predictor
###############################################################################

F_pred_network = model.predict([PDD_val, Dose2D_val])

mse_network = np.mean((F_pred_network - F_val)**2)


###############################################################################
# Evaluation
###############################################################################

print("Average predictor:       ", mse_mean)
print("Nearest neighbor:        ", mse_nn)
print("Neural network:          ", mse_network)

###############################################################################
# Compare predictions for individual fluence maps
###############################################################################

sample_indices = [0, 1, 2, 4]

for sample_index in sample_indices:

    maps = [
        F_val[sample_index],
        F_pred_network[sample_index],
        F_pred_nn[sample_index],
        F_pred_mean[sample_index]
    ]

    titles = [
        "True fluence",
        "Neural network",
        "Nearest neighbor",
        "Average predictor"
    ]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    for ax, fluence_map, title in zip(axes, maps, titles):

        im = ax.imshow(
            fluence_map.T,
            origin="lower",
            extent=[-50, 50, -50, 50],
            cmap="viridis",
            vmin=0,
            vmax=1
        )

        ax.set_title(title)
        ax.set_xlabel("X [mm]")
        ax.set_ylabel("Z [mm]")

    fig.colorbar(
        im,
        ax=axes,
        label="Normalized fluence",
        shrink=0.8
    )

    fig.suptitle(f"Validation sample {sample_index}")

    plt.show()


###############################################################################
# MSE for each validation sample
###############################################################################

mse_mean_each = np.mean(
    (F_pred_mean - F_val)**2,
    axis=(1, 2)
)

mse_nn_each = np.mean(
    (F_pred_nn - F_val)**2,
    axis=(1, 2)
)

mse_network_each = np.mean(
    (F_pred_network - F_val)**2,
    axis=(1, 2)
)

nn_better_than_nearest = np.sum(
    mse_network_each < mse_nn_each
)

nearest_better_than_nn = np.sum(
    mse_nn_each < mse_network_each
)

print("\nPer-sample comparison:")
print("NN better than nearest:", nn_better_than_nearest)
print("Nearest better than NN:", nearest_better_than_nn)


###############################################################################
# Plot distribution of MSE
###############################################################################

plt.figure(figsize=(8, 5))

plt.hist(
    mse_nn_each,
    bins=10,
    alpha=0.6,
    label="Nearest neighbor"
)

plt.hist(
    mse_network_each,
    bins=10,
    alpha=0.6,
    label="Neural network"
)

plt.xlabel("MSE per validation fluence map")
plt.ylabel("Number of maps")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

