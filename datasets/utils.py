
import math
import pandas as pd
import numpy as np
from sklearn.datasets import make_blobs
import pathlib
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
import numpy as np

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
    # ("SYNTH_C_2_F_50_n1000k_logistic", 1000_000, 5,  2, 0),
    # number of features should be greater than the number of clusters
    ("SYNTH_C_30_F_50_n1000_000k_kmeans", 100_00, 50, 30, 1),
    ("SYNTH_C_5_F_50_n1000_000k_kmeans", 1000_0, 50, 5, 1),
    ("SYNTH_C_80_F_120_n1000_000k_kmeans", 1000_0, 120, 80, 1),
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

columns_D = ["DatasetName", "DatasetSize", "NumClusters", "Mode", "chunk_single", "improve_threshold", "iter_single", "iter_double", "Suite", "Time", "Memory_MB", "Inertia"]

columns_E = [
    "DatasetName", "DatasetSize", "NumClusters", "Mode", "MB_Iter", "MB_Batch", "RefineIter", "iter_single", "iter_double", "Suite", "Time", "Memory_MB", "Inertia"]

columns_F = [
    "DatasetName", "DatasetSize", "NumClusters", "Mode", "tol_single", "tol_double", "single_iter_cap", "freeze_stable", "freeze_patience", "iter_single", "iter_double", "Suite", "Time", "Memory_MB", "Inertia"]


# utils.py
svm_columns_A = [ "DatasetName","DatasetSize","NumClasses","Tolerance","Cap", "iter_single","iter_double","Suite","Time","Memory_MB","Accuracy"
]
svm_columns_B = svm_columns_A  # same schema

# Optional column headers you can reuse when building DataFrames
enet_columns_A = [
    "DatasetName", "DatasetSize", "NumFeatures",
    "Mode", "Cap", "tolerance_single",
    "iter_single", "iter_double", "Suite",
    "Time", "Memory_MB", "R2", "MSE",
]

enet_columns_B = [
    "DatasetName", "DatasetSize", "NumFeatures",
    "Mode", "tolerance_single",
    "iter_single", "iter_double", "Suite",
    "Time", "Memory_MB", "R2", "MSE",
]






