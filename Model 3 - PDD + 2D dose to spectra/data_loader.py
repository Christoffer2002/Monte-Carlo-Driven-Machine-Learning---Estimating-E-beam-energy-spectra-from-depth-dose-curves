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
    Parameters = []


    # Load TOPAS data
    for _, row in parameters.iterrows():

        simulation_id = int(row["simulation_id"])
        ID = f"{simulation_id:04d}"
        
        thickness = row["primary_thickness_cm"]
        energy = row["beam_energy_MeV"]
        disk1_radius = row["disk1_radius_cm"]
        disk2_radius = row["disk2_radius_cm"]

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
        #dose_map = dose_map.reshape(25, 2, 25, 2).mean(axis=(1, 3))
        # dose_map = dose_map.reshape(10, 5, 10, 5).mean(axis=(1, 3))
        
        # Check dimensions and missing bins
        if pdd.shape != (45,):
            raise ValueError(f"Incorrect PDD shape: {ID}")

        if np.isnan(dose_map).any():
            raise ValueError(f"Missing Dose2D bins: {ID}")


        PDDs.append(pdd)
        Dose2Ds.append(dose_map)
        Parameters.append([thickness, energy, disk1_radius, disk2_radius])

    PDDs = np.array(PDDs)
    Dose2Ds = np.array(Dose2Ds)
    Parameters = np.array(Parameters)

    # Normalize data to maximum
    #PDDs = PDDs / PDDs.max(axis=1, keepdims=True)
    #Dose2Ds = Dose2Ds / Dose2Ds.max(axis=(1, 2), keepdims=True)
    param_min = np.array([0.003810, 12, 0.40005*0.75, 0.49911*0.75])
    param_max = np.array([0.011430, 14, 0.40005*1.25, 0.49911*1.25])

    Parameters = (Parameters - param_min) / (param_max - param_min)
    
    # Disk radii are varied using same scaling factor - after norm col 3&4 are equal:
    Parameters = Parameters[:, :3]

    # Train / validation split (20:80)
    (PDD_train, PDD_val, Dose2D_train, Dose2D_val, param_train, param_val) = train_test_split(
        PDDs, Dose2Ds, Parameters, test_size=0.2, random_state=42)
    
    
    #Normalize to max value
    pdd_max = PDD_train.max()
    dose_max = Dose2D_train.max()
    
    PDD_train = PDD_train / pdd_max
    PDD_val = PDD_val / pdd_max
    Dose2D_train = Dose2D_train / dose_max
    Dose2D_val = Dose2D_val / dose_max

    return (PDD_train, PDD_val, Dose2D_train, Dose2D_val, param_train, param_val)

#(PDD_train, PDD_val, Dose2D_train, Dose2D_val, param_train, param_val) = load_data()

#print(Dose2D_train.shape)



