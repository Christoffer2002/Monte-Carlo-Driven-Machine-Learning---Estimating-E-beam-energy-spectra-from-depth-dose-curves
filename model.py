from tensorflow import keras
from tensorflow.keras import layers
from data_loader import load_data
import matplotlib.pyplot as plt
from pathlib import Path



# Setup
root_directory = Path(__file__).resolve().parent

# Build model - my model
# model = keras.Sequential([
#     layers.Input(shape=(90,)),
#     layers.Dense(128, activation="relu"),
#     layers.Dense(256, activation="relu"),
#     # layers.Dense(7, activation="relu")
#     layers.Dense(7, activation="softmax")
# ])

# Article model
model = keras.Sequential([
    layers.Input(shape=(90,)),
    layers.Dense(2000, activation="relu"),
    # layers.Dense(7, activation="relu")
    layers.Dense(7, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="mse"
)

model.summary()


# Load data - PDDs: (10000, 90)  X:    (10000, 7)
PDD_train, PDD_val, X_train, X_val = load_data()


# Train model
history = model.fit(
    PDD_train,
    X_train,
    validation_data=(PDD_val, X_val),
    epochs=1000,
    batch_size=64
)

model.save(root_directory / "models" / "spectrum_model.keras")


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
