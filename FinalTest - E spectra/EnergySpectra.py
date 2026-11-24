#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 09:18:21 2026

@author: Christoffer
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

# ============================================================
# Setup
# ============================================================

root = Path(__file__).resolve().parent
data_folder = root / "data"

E_min = 0
E_max = 15

# ============================================================
# Load spectrum
# ============================================================

def load_spectrum(filename, n_bins):

    data = pd.read_csv(
        data_folder / filename,
        comment="#",
        header=None
    )

    # Convert to NumPy array
    data = data.to_numpy().reshape(-1, 3)

    # Remove underflow and overflow bins
    data = data[1:-1, :]

    # Extract fluence (Sum)
    fluence = data[:, 0]

    # Energy bin edges
    energy_edges = np.linspace(E_min, E_max, n_bins + 1)

    return energy_edges, fluence


# ============================================================
# Plot spectrum comparison
# ============================================================
def plot_spectrum(ax, true_file, pred_file, n_bins, title):

    energy_edges, true_fluence = load_spectrum(true_file, n_bins)
    _, pred_fluence = load_spectrum(pred_file, n_bins)

    bin_width = energy_edges[1] - energy_edges[0]
    energy_centers = (energy_edges[:-1] + energy_edges[1:]) / 2

    # Predicted: grey histogram
    ax.bar(
        energy_centers,
        pred_fluence,
        width=bin_width * 0.92,
        color="#B0B0B0",
        edgecolor="#555555",
        linewidth=0.5,
        label="Predicted",
        zorder=2
    )

    # Ground truth: red outline
    ax.stairs(
        true_fluence,
        energy_edges,
        color="#FF3030",
        linewidth=1.3,
        label="Ground truth",
        zorder=3
    )

    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Energy [MeV]", fontsize=11)
    ax.set_ylabel("Fluence", fontsize=11)

    ax.set_xlim(0, 15)
    ax.set_ylim(bottom=0)

    ax.legend(
        loc="upper right",
        fontsize=9,
        frameon=True,
        facecolor="white",
        edgecolor="#CCCCCC"
    )

    ax.tick_params(direction="out", length=3, width=0.8)

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("#555555")



# ============================================================
# Low configuration
# ============================================================

fig, axes = plt.subplots(2, 1, figsize=(5, 5.5))

plot_spectrum(
    axes[0],
    "Low_True_ElectronSpectrum100.csv",
    "Low_Pred_ElectronSpectrum100.csv",
    100,
    "Electron spectrum"
)

plot_spectrum(
    axes[1],
    "Low_True_PhotonSpectrum100.csv",
    "Low_Pred_PhotonSpectrum100.csv",
    100,
    "Photon spectrum"
)

plt.tight_layout()
fig.savefig(root / "Low_spectra.pdf", bbox_inches="tight")
plt.show()


# ============================================================
# Center configuration
# ============================================================

fig, axes = plt.subplots(2, 1, figsize=(5, 5.5))

plot_spectrum(
    axes[0],
    "Center_True_ElectronSpectrum100.csv",
    "Center_Pred_ElectronSpectrum100.csv",
    100,
    "Electron spectrum"
)

plot_spectrum(
    axes[1],
    "Center_True_PhotonSpectrum100.csv",
    "Center_Pred_PhotonSpectrum100.csv",
    100,
    "Photon spectrum"
)

plt.tight_layout()
fig.savefig(root / "Center_spectra.pdf", bbox_inches="tight")
plt.show()


# ============================================================
# High configuration
# ============================================================

fig, axes = plt.subplots(2, 1, figsize=(5, 5.5))

plot_spectrum(
    axes[0],
    "High_True_ElectronSpectrum100.csv",
    "High_Pred_ElectronSpectrum100.csv",
    100,
    "Electron spectrum"
)

plot_spectrum(
    axes[1],
    "High_True_PhotonSpectrum100.csv",
    "High_Pred_PhotonSpectrum100.csv",
    100,
    "Photon spectrum"
)

plt.tight_layout()
fig.savefig(root / "High_spectra.pdf", bbox_inches="tight")
plt.show()