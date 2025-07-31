
import pandas as pd
import numpy as np
from sklearn.datasets import make_blobs
import pathlib

DATA_DIR = pathlib.Path("datasets") 

def generate_synthetic_data(n_samples, n_features, n_clusters, random_state):
    X, y_true = make_blobs(n_samples=n_samples, n_features=n_features, centers=n_clusters, random_state=random_state)
    
    return X.astype(np.float64), y_true
  
def load_3d_road(n_rows=1_000_000):
    path = DATA_DIR / "3D_spatial_network.csv"
    
    X = pd.read_csv(path, sep=r"\s+|,", engine="python",  
                    header=None, usecols=[1, 2, 3],
                    nrows=n_rows, dtype=np.float64).to_numpy()
    return X, None
    
def load_susy(n_rows=1_000_000):
    path = DATA_DIR / "SUSY.csv"
    df = pd.read_csv(path, header=None, nrows=n_rows,
                     dtype=np.float64, names=[f"c{i}" for i in range(9)])
    # start time 
    X = df.iloc[:, 1:].to_numpy()     
    return X, None

synth_specs = [
    # number of samples; number of features, number of clusters, random seeds
    ("SYNTH_C_5_F_80_n100k", 100_000, 5,  5, 0),
    # ("SYNTH_C_80_F_5_n100k", 1000_000, 5, 80, 1),
    # ("SYNTH_C_80_30_n100k", 1000_000, 30, 80, 1)
]


# Real-dataset
real_datasets = {
    "3D_ROAD": load_3d_road,
    "SUSY":    load_susy,
} 


columns_A = [
    'DatasetName', 'DatasetSize', 'NumClusters', 
    'Mode', 'Cap', 'tolerance_single', 'iter_single', 'iter_double', 'Suite',
    'Time', 'Memory_MB', 'ARI', 'DBI', 'Inertia'
]

columns_B = [
    'DatasetName', 'DatasetSize', 'NumClusters',
    'Mode', 'tolerance_single', 'iter_single', 'iter_double', 'Suite',
    'Time', 'Memory_MB', 'ARI', 'DBI', 'Inertia'
]

columns_C = columns_A  # since it's same structure as A

columns_D = [
    'DatasetName', 'DatasetSize', 'NumClusters',
    'Mode', 'tolerance_single', 'Cap', 'iter_single', 'iter_double', 'Suite',
    'Time', 'Memory_MB', 'ARI', 'DBI', 'Inertia'
]
