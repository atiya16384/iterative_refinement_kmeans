
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


def generate_synthetic_data_en(
    n_samples: int,
    n_features: int,
    seed: int = 0,
    sparsity: float = 0.1,
    noise: float = 1.0,
    rho: float = 0.5,
):
    """
    ElasticNet/Lasso-friendly synthetic regression:
      - Correlated features with Toeplitz covariance (rho^{|i-j|})
      - Sparse true beta (first floor(sparsity * n_features) non-zero)
      - y = X @ beta + ε
    """
    rng = np.random.default_rng(seed)

    # Toeplitz covariance
    idx = np.arange(n_features)
    cov = rho ** np.abs(idx[:, None] - idx[None, :])

    # sample X ~ N(0, cov)
    L = np.linalg.cholesky(cov + 1e-12 * np.eye(n_features))
    Z = rng.standard_normal((n_samples, n_features))
    X = Z @ L.T  # correlated

    # sparse beta
    k = max(1, int(sparsity * n_features))
    beta = np.zeros(n_features)
    nz = rng.choice(n_features, size=k, replace=False)
    beta[nz] = rng.normal(0, 1.0, size=k)

    y = X @ beta + noise * rng.standard_normal(n_samples)
    return X.astype(np.float64, copy=False), y.astype(np.float64, copy=False)

# A tiny spec list you can expand
enet_specs = [
    # (tag, n_samples, n_features, sparsity, noise, seed)
    ( "EN_SYNTH_n100k_d200_s10", 100_000, 200, 0.10, 1.0, 0 ),
    ( "EN_SYNTH_n200k_d400_s05", 200_000, 400, 0.05, 1.0, 1 ),
]

synth_specs = [
    # number of samples; number of features, number of clusters, random seeds
    # ("SYNTH_C_5_F_5_n1000_000k_logistic", 1000_000, 150,  5, 0),
    ("SYNTH_C_80_F_5_n1000_000k_kmeans", 10000_00, 5, 80, 1),
    ("SYNTH_C_40_F_5_n1000_000k_kmeans", 10000_00, 5, 40, 1),
    ("SYNTH_C_5_F_5_n1000_000k_kmeans", 10000_00, 5, 5, 1),
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

# Result schemas
lr_columns_A = [  'DatasetName', 'DatasetSize', 'NumClasses', 'Mode', 'Cap', 'tolerance_single', 'iter_single', 'iter_double', 'Suite', 'Time', 'Memory_MB', 'Accuracy']

lr_columns_B = [ 'DatasetName', 'DatasetSize', 'NumClasses','Mode', 'tolerance_single', 'iter_single', 'iter_double', 'Suite', 'Time', 'Memory_MB', 'Accuracy']

# Column schemas (used by main to build DataFrames)
en_columns_A = [
    "DatasetName","NumFeatures","Mode","Cap","tolerance_single",
    "iter_single","iter_double","Suite","Time","Memory_MB","R2","MSE"
]
en_columns_B = [
    "DatasetName","NumFeatures","Mode","tolerance_single",
    "iter_single","iter_double","Suite","Time","Memory_MB","R2","MSE"
]





