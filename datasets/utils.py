
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
    # ("SYNTH_C_5_F_5_n100k", 500_000, 5,  5, 0),
    # ("SYNTH_C_80_F_5_n100k", 10000_00, 5, 80, 1),
    # ("SYNTH_C_30_F_5_n100k", 10000_00, 5, 40, 1),
    ("SYNTH_C_5_F_50_n100k", 10000_00, 100, 5, 1),
]

# Real-dataset
real_datasets = {
    "3D_ROAD": load_3d_road,
    "SUSY":    load_susy,
} 

columns_A = [
    'DatasetName', 'DatasetSize', 'NumClusters', 
    'Mode', 'Cap', 'tolerance_single', 'iter_single', 'iter_double', 'Suite',
    'Time', 'Memory_MB', 'Inertia'
]

columns_B = [
    'DatasetName', 'DatasetSize', 'NumClusters',
    'Mode', 'tolerance_single', 'iter_single', 'iter_double', 'Suite',
    'Time', 'Memory_MB',  'Inertia'
]

columns_C = columns_A  # since it's same structure as A

columns_D = [
    "DatasetName", "DatasetSize", "NumClusters",
    "Mode",                 # "D"
    "chunk_single",         # int (the single-precision burst length used by the variant)
    "improve_threshold",    # float (relative inertia improvement threshold used by the variant)
    "iter_single", "iter_double",
    "Suite",                # "Double" or "Adaptive"
    "Time", "Memory_MB", "Inertia"
]



columns_E = [
    "DatasetName", "DatasetSize", "NumClusters",
    "Mode",                 # "E"
    "MB_Iter",              # int (minibatch iterations)
    "MB_Batch",             # int (batch size)
    "RefineIter",           # int (full KMeans refinement iterations)
    "iter_single", "iter_double",
    "Suite",                # "Double" or "MiniBatch+Full"
    "Time", "Memory_MB", "Inertia"
]


columns_F = [
    "DatasetName", "DatasetSize", "NumClusters",
    "Mode",                 # "F"
    "tol_single",           # float (per-cluster movement tolerance)
    "tol_double",           # float (KMeans refinement tolerance)
    "single_iter_cap",      # int (cap for phase-1 iterations)
    "freeze_stable",        # bool
    "freeze_patience",      # int
    "iter_single", "iter_double",
    "Suite",                # "Double" or "MixedPerCluster"
    "Time", "Memory_MB", "Inertia"
]


# SVM result schemas (11 columns each; no "Mode" column here)
svm_columns_A = [
    'DatasetName', 'DatasetSize', 'NumClasses',
    'Tolerance', 'Cap',
    'iter_single', 'iter_double', 'Suite',
    'Time', 'Memory_MB', 'Accuracy'
]

svm_columns_B = [
    'DatasetName', 'DatasetSize', 'NumClasses',
    'Tolerance', 'Cap',
    'iter_single', 'iter_double', 'Suite',
    'Time', 'Memory_MB', 'Accuracy'
]




