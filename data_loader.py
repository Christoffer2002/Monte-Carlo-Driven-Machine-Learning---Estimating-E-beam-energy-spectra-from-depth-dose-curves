import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split


def load_data():

    root_directory = Path(__file__).resolve().parent
    data_dir = root_directory / "data"

    PDDs = np.load(data_dir / "PDDs.npy")
    # Normalize each PDD to its own maximum
    #PDDs = PDDs / PDDs.max(axis=1, keepdims=True)
    
    
    X = np.load(data_dir / "Spectra.npy")

    PDD_train, PDD_val, X_train, X_val = train_test_split(
        PDDs,
        X,
        test_size=0.2,
        random_state=42
    )

    return PDD_train, PDD_val, X_train, X_val