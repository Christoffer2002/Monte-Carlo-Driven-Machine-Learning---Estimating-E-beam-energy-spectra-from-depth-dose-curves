from tensorflow import keras
from tensorflow.keras import layers
from data_loader import load_data
import matplotlib.pyplot as plt
from pathlib import Path



# Setup
root_directory = Path(__file__).resolve().parent

# Load data
PDD_train, PDD_val, X_train, X_val = load_data()

# Build model - my model

# model = keras.Sequential([
#     layers.Input(shape=(90,)),
#     layers.Dense(128, activation="relu"),
#     layers.Dense(256, activation="relu"),
#     layers.Dense(1)
# ])
model = keras.Sequential([
    layers.Input(shape=(90,)),
    layers.Dense(8, activation="relu"),
    layers.Dense(4, activation="relu"),
    layers.Dense(1)
])

model.compile(
    optimizer="adam",
    loss="mse"
)

model.summary()



# Training stop when validation loss has not improved
# stop_training = keras.callbacks.EarlyStopping(
#     monitor="val_loss",
#     patience=100,
#     restore_best_weights=True
# )

# Train model
# history = model.fit(
#     PDD_train,
#     X_train,
#     validation_data=(PDD_val, X_val),
#     epochs=50000,
#     batch_size=64,
#     callbacks=[stop_training]
# )

# Train model
history = model.fit(
    PDD_train,
    X_train,
    validation_data=(PDD_val, X_val),
    epochs=3000,
    batch_size=64
)

model.save(root_directory / "models" / "Small_model_no_norm_foil_thickness_model.keras")


# -------------------------------------------------------
# Plot training history - epochs against MSE
# -------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(history.history["loss"], label="Training loss")
plt.plot(history.history["val_loss"], label="Validation loss")

plt.xlabel("Epoch")
plt.ylabel("MSE loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
