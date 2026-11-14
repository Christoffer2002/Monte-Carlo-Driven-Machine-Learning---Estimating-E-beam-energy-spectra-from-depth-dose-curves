import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


def load_data():

    root_directory = Path(__file__).resolve().parent
    data_dir = root_directory / "data"

    # Load simulation parameters
    parameters = pd.read_csv(data_dir / "simulation_parameters.csv")
    # Only load the first 500 simulations################################################################
    parameters = parameters.iloc[:500]



    PDDs = []
    Dose2Ds = []
    Fluences = []


    # Load TOPAS data
    for _, row in parameters.iterrows():

        simulation_id = int(row["simulation_id"])
        ID = f"{simulation_id:04d}"

        # Load PDD
        pdd_df = pd.read_csv(data_dir / f"PDD_{ID}.csv", comment="#", header=None)
        # Extract PDD
        pdd = pdd_df.iloc[:, -1].to_numpy(dtype=float)
        
        # Load dosemap
        dose_df = pd.read_csv(data_dir / f"Dose2D_{ID}.csv", comment="#", header=None)
        # Create dosemap
        dose_map = np.full((50, 50), np.nan)
        x = dose_df.iloc[:, 0].to_numpy(dtype=int)
        z = dose_df.iloc[:, 2].to_numpy(dtype=int)

        dose_map[x, z] = dose_df.iloc[:, -1].to_numpy()
        
        # Load fluence
        fluence_df = pd.read_csv(data_dir / f"Fluence_{ID}.csv", comment="#", header=None)
        # Create fluence mapping
        fluence_map = np.full((50, 50), np.nan)
        x = fluence_df.iloc[:, 0].to_numpy(dtype=int)
        z = fluence_df.iloc[:, 2].to_numpy(dtype=int)
        fluence_map[x, z] = fluence_df.iloc[:, -1].to_numpy()
        
        # Rebin fluence from 50x50 to 25x25#####################################################################
        #fluence_map = fluence_map.reshape(25, 2, 25, 2).mean(axis=(1, 3))
        fluence_map = fluence_map.reshape(10, 5, 10, 5).mean(axis=(1, 3))

        # Check dimensions and missing bins
        if pdd.shape != (45,):
            raise ValueError(f"Incorrect PDD shape: {ID}")

        if np.isnan(dose_map).any():
            raise ValueError(f"Missing Dose2D bins: {ID}")

        if np.isnan(fluence_map).any():
            raise ValueError(f"Missing Fluence bins: {ID}")

        PDDs.append(pdd)
        Dose2Ds.append(dose_map)
        Fluences.append(fluence_map)

    # Convert to NumPy arrays
    PDDs = np.array(PDDs)
    Dose2Ds = np.array(Dose2Ds)
    Fluences = np.array(Fluences)

    # Normalize data
    # Normalize each PDD to its own maximum
    PDDs = PDDs / PDDs.max(axis=1, keepdims=True)

    # Normalize each 2D dose map to its own maximum
    Dose2Ds = Dose2Ds / Dose2Ds.max(axis=(1, 2), keepdims=True)

    # Normalize each fluence map to its own maximum
    Fluences = Fluences / Fluences.max(axis=(1, 2), keepdims=True)

    # Train / validation split (20:80)

    (PDD_train, PDD_val, Dose2D_train, Dose2D_val, F_train, F_val) = train_test_split(
        PDDs,
        Dose2Ds,
        Fluences,
        test_size=0.2,
        random_state=42)

    return (PDD_train, PDD_val, Dose2D_train, Dose2D_val, F_train, F_val)
