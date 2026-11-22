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

# Setup
root_directory = Path(__file__).resolve().parent

# Load trained model
model = keras.models.load_model(root_directory / "models" / "final_model.keras")

# Load data
(PDD_train, PDD_val, PDD_test, Dose2D_train, Dose2D_val, Dose2D_test, param_train, param_val, param_test) = load_data()

param_names = ["Thickness", "Energy", "Disk scaling"]

###############################################################################
# Average predictor
###############################################################################
param_mean = np.mean(param_train, axis=0)

param_pred_mean = np.tile(param_mean, (param_test.shape[0], 1))

mse_mean = np.mean((param_pred_mean - param_test)**2)

###############################################################################
# Nearest neighbor predictor
###############################################################################

# Combine PDD and Dose2D into feature vectors
PDD_train_flat = PDD_train.reshape(len(PDD_train), -1)
PDD_test_flat = PDD_test.reshape(len(PDD_test), -1)

Dose_train_flat = Dose2D_train.reshape(len(Dose2D_train), -1)
Dose_test_flat = Dose2D_test.reshape(len(Dose2D_test), -1)

# Scale each modality so the 10000 dose pixels do not
# automatically dominate the 90 PDD values
PDD_train_flat = PDD_train_flat / np.sqrt(PDD_train_flat.shape[1])
PDD_test_flat = PDD_test_flat / np.sqrt(PDD_test_flat.shape[1])

Dose_train_flat = Dose_train_flat / np.sqrt(Dose_train_flat.shape[1])
Dose_test_flat = Dose_test_flat / np.sqrt(Dose_test_flat.shape[1])

X_train = np.concatenate([PDD_train_flat, Dose_train_flat], axis=1)

X_test = np.concatenate([PDD_test_flat, Dose_test_flat], axis=1)


# 1-nearest-neighbor
knn = KNeighborsRegressor(n_neighbors=1)

knn.fit(X_train, param_train)

param_pred_nn = knn.predict(X_test)

mse_nn = np.mean((param_pred_nn - param_test)**2)


###############################################################################
# Neural network predictor
###############################################################################

param_pred_network = model.predict([PDD_test, Dose2D_test])

mse_network = np.mean((param_pred_network - param_test)**2)


###############################################################################
# Evaluation
###############################################################################

print("Average predictor: ", mse_mean)
print("Nearest neighbor:  ", mse_nn)
print("Neural network:    ", mse_network)

mse_mean_param = np.mean((param_pred_mean - param_test)**2, axis=0)

mse_nn_param = np.mean((param_pred_nn - param_test)**2, axis=0)

mse_network_param = np.mean((param_pred_network - param_test)**2, axis=0)

fig, axes = plt.subplots(3, 1, figsize=(6, 15))

for i, ax in enumerate(axes):

    ax.scatter(param_test[:, i], param_pred_network[:, i], alpha=0.6, label="Neural network")

    ax.scatter(param_test[:, i], param_pred_nn[:, i], alpha=0.4, label="Nearest neighbor")

    ax.axhline(param_mean[i], linestyle=":", color="gray", label="Average predictor")

    ax.plot([0, 1], [0, 1], "k--", label="Perfect prediction")

    ax.set_title(param_names[i])
    ax.set_xlabel("True parameter")
    ax.set_ylabel("Predicted parameter")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(True)
    ax.legend()

plt.tight_layout()
plt.show()