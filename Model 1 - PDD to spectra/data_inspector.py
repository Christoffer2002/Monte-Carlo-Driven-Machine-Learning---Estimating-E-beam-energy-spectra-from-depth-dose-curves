# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 10:00:40 2026

@author: 45401
"""

import matplotlib.pyplot as plt
from data_loader import load_data


PDD_train, PDD_val, X_train, X_val = load_data()

# Plot training PDDs
for pdd, thickness in zip(PDD_train, X_train):
    plt.plot(pdd, label=f"{thickness:.2f} t0")

plt.xlabel("Depth bin")
plt.ylabel("Normalized dose")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()