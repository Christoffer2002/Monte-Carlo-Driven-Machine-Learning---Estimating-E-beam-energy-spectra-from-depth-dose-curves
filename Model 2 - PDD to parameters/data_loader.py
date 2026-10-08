import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


def load_data():

    root_directory = Path(__file__).resolve().parent
    data_dir = root_directory / "data"

    # Load simulation parameters
    parameters = pd.read_csv(data_dir / "simulation_parameters.csv")

    PDDs = []
    X = []

    # Load all simulations
    for _, row in parameters.iterrows():

        simulation_id = int(row["simulation_id"])
        thickness = row["primary_thickness_cm"]

        # Load corresponding TOPAS output
        file = data_dir / f"Dose_{simulation_id:04d}.csv"

        df = pd.read_csv(file, skiprows=7)

        # Last column contains dose
        depth_dose = df.iloc[:, -1].to_numpy()

        PDDs.append(depth_dose)
        X.append(thickness)


    PDDs = np.array(PDDs)
    X = np.array(X)
    
    # Convert to nominal thicknes range is [0.5-1.5]
    nominal_thickness = 0.007620  # cm
    X = X / nominal_thickness

    # Normalize each PDD to its own maximum
    # PDDs = PDDs / PDDs.max(axis=1, keepdims=True)

    # Train / validation split
    PDD_train, PDD_val, X_train, X_val = train_test_split(
        PDDs,
        X,
        test_size=0.2,
        random_state=42
    )

    return PDD_train, PDD_val, X_train, X_val
