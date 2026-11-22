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
keras.utils.set_random_seed(42)

# Load data
(PDD_train, PDD_val, PDD_test, Dose2D_train, Dose2D_val, Dose2D_test, param_train, param_val, param_test) = load_data()
#Model
# Branch 1 - PDD
pdd_input = keras.Input(shape=(45,), name="PDD")
pdd = layers.Dense(64, activation="relu")(pdd_input) # 64 features

# Branch 2 - Dosemap
dose_input = keras.Input(shape=(50, 50), name="Dose2D")

dose = layers.Flatten()(dose_input)
dose = layers.Dense(128, activation="relu")(dose) # 128 features

# Combine both branches
combined = layers.Concatenate()([pdd, dose])

x = layers.Dense(512, activation="relu")(combined) # 192 -> 512 features

param_output = layers.Dense(3, activation="sigmoid")(x) # 3 params

model = keras.Model(inputs=[pdd_input, dose_input], outputs=param_output)

#model.compile(optimizer="adam", loss="mse")
model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4),loss="mse")

model.summary()

early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=300, restore_best_weights=True)

# history = model.fit([PDD_train, Dose2D_train], param_train, validation_data=(
#         [PDD_val, Dose2D_val], param_val), epochs=5000, batch_size=16)
history = model.fit([PDD_train, Dose2D_train], param_train, validation_data=([PDD_val, Dose2D_val], param_val),
    epochs=5000, batch_size=16, callbacks=[early_stopping])
# Save model
model.save(model_folder / "final_model.keras")



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
