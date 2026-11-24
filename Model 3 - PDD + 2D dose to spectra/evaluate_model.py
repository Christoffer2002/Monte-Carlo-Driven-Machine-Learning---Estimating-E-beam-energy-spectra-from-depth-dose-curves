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
# Neural network predictor
###############################################################################

param_pred_network = model.predict([PDD_test, Dose2D_test])

mse_network = np.mean((param_pred_network - param_test)**2)


###############################################################################
# Evaluation
###############################################################################

print("Neural network:    ", mse_network)

mse_network_param = np.mean((param_pred_network - param_test)**2, axis=0)

fig, ax = plt.subplots(figsize=(5, 5))

ax.scatter(param_test[:, 0], param_pred_network[:, 0],
           alpha=0.5, s=20, marker="o", label="Thickness")

ax.scatter(param_test[:, 1], param_pred_network[:, 1],
           alpha=0.5, s=20, marker="^", label="Energy")

ax.scatter(param_test[:, 2], param_pred_network[:, 2],
           alpha=0.5, s=20, marker="s", label="Disk scaling")

ax.plot([0, 1], [0, 1], "k--", linewidth=1,
        label="Perfect prediction")

ax.set_xlabel("True parameter")
ax.set_ylabel("Predicted parameter")

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect("equal")

ax.grid(True, alpha=0.25)
ax.legend(loc="upper left")

plt.tight_layout()
fig.savefig("TestPredictions.pdf", bbox_inches="tight")
plt.show()