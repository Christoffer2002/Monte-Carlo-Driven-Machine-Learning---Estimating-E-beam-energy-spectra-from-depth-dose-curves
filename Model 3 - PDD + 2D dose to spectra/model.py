"""
Created on Thu Oct  1 10:00:40 2026

@author: Christoffer R. Olesen
"""

from tensorflow import keras
from tensorflow.keras import layers
from data_loader import load_data
import matplotlib.pyplot as plt
from pathlib import Path


# Setup
root_directory = Path(__file__).resolve().parent
model_folder = root_directory / "models"

# Load data
(PDD_train, PDD_val, Dose2D_train, Dose2D_val, F_train, F_val) = load_data()


# Build model - Two input model
# Branch 1 - PDD
pdd_input = keras.Input(shape=(90,), name="PDD")
pdd = layers.Dense(64, activation="relu")(pdd_input) # 64 features

# Branch 2 - Dosemap
dose_input = keras.Input(shape=(100, 100), name="Dose2D")

dose = layers.Flatten()(dose_input)
dose = layers.Dense(128, activation="relu")(dose) # 128 features

# Combining the two branches
combined = layers.Concatenate()([pdd, dose])


x = layers.Dense(256, activation="relu")(combined) # 192 -> 256 features

x = layers.Dense(100 * 100, activation="sigmoid")(x) # Sigmoid due to normed data

fluence_output = layers.Reshape((100, 100))(x) # Reshape to match F


model = keras.Model(inputs=[pdd_input, dose_input], outputs=fluence_output)

model.compile(optimizer="adam", loss="mse")

model.summary()






history = model.fit([PDD_train, Dose2D_train], F_train, validation_data=(
        [PDD_val, Dose2D_val], F_val), epochs=2500, batch_size=16
    #callbacks=[stop_training]
)

# Save model
model.save(model_folder / "fluence_model.keras")



###############################################################################
# Plot training history - epochs against MSE
###############################################################################
plt.figure(figsize=(8, 5))

plt.plot(history.history["loss"], label="Training loss")
plt.plot(history.history["val_loss"], label="Validation loss")

plt.xlabel("Epoch")
plt.ylabel("MSE loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
