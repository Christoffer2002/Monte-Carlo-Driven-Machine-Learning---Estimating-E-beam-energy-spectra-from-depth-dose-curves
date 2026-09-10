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
model = keras.models.load_model(
    root_directory / "models" / "spectrum_model.keras"
)

# Load data
PDD_train, PDD_val, X_train, X_val = load_data()


#-------------------------------------------------------
# Compare true and predicted spectrum from NN
#-------------------------------------------------------
# Predict a spectrum
# sample_index = 0

# PDD_sample = PDD_val[sample_index]
# X_true = X_val[sample_index]

# X_pred = model.predict(PDD_sample.reshape(1, -1))[0]


# plt.figure(figsize=(8, 5))

# plt.plot(X_true, marker="o", label="True spectrum")
# plt.plot(X_pred, marker="o", label="Predicted spectrum")

# plt.xlabel("Energy bin")
# plt.ylabel("Weight")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

#-------------------------------------------------------
# Average predictor
#-------------------------------------------------------

# Average spectrum from training set
X_mean = np.mean(X_train, axis=0)

# Predict the same mean spectrum for every validation sample
X_pred_mean = np.tile(X_mean, (X_val.shape[0], 1))

# Mean squared error
mse_mean = np.mean((X_pred_mean - X_val)**2)

#-------------------------------------------------------
# Nearest neighbor test
#-------------------------------------------------------
# 1-nearest-neighbor
knn = KNeighborsRegressor(n_neighbors=1)

# Learn relationship between PDDs and spectra
knn.fit(PDD_train, X_train)

# Predict spectra for validation PDDs
X_pred_nn = knn.predict(PDD_val)

# MSE
mse_nn = mean_squared_error(X_val, X_pred_nn)



X_pred_network = model.predict(PDD_val)

mse_network = mean_squared_error(
    X_val,
    X_pred_network
)


#Evaluation
print("Average predictor:       ", mse_mean)
print("Nearest neighbor:        ", mse_nn)
print("Neural network:          ", mse_network)




# -------------------------------------------------------
# Compare predictions for individual spectra
# -------------------------------------------------------

energies = np.arange(8, 15)

sample_indices = [0, 1, 2, 4]

for sample_index in sample_indices:

    plt.figure(figsize=(8, 5))

    plt.plot(
        energies,
        X_val[sample_index],
        marker="o",
        label="True spectrum"
    )

    plt.plot(
        energies,
        X_pred_network[sample_index],
        marker="o",
        label="Neural network"
    )

    plt.plot(
        energies,
        X_pred_nn[sample_index],
        marker="o",
        label="Nearest neighbor"
    )

    plt.plot(
        energies,
        X_pred_mean[sample_index],
        marker="o",
        label="Average predictor"
    )

    plt.xlabel("Energy [MeV]")
    plt.ylabel("Weight")
    plt.title(f"Validation sample {sample_index}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    


# -------------------------------------------------------
# MSE for each validation sample
# -------------------------------------------------------

mse_mean_each = np.mean(
    (X_pred_mean - X_val)**2,
    axis=1
)

mse_nn_each = np.mean(
    (X_pred_nn - X_val)**2,
    axis=1
)

mse_network_each = np.mean(
    (X_pred_network - X_val)**2,
    axis=1
)

nn_better_than_nearest = np.sum(
    mse_network_each < mse_nn_each
)

nearest_better_than_nn = np.sum(
    mse_nn_each < mse_network_each
)

plt.figure(figsize=(8, 5))

plt.hist(
    mse_nn_each,
    bins=50,
    alpha=0.6,
    label="Nearest neighbor"
)

plt.hist(
    mse_network_each,
    bins=50,
    alpha=0.6,
    label="Neural network"
)

plt.xlabel("MSE per validation spectrum")
plt.ylabel("Number of spectra")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()