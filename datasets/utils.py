
import pandas as pd
import numpy as np
from sklearn.datasets import make_blobs
import pathlib
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
import numpy as np

DATA_DIR = pathlib.Path("datasets") 

def generate_synthetic_data(n_samples, n_features, n_clusters, random_state):
    X, y_true = make_blobs(n_samples=n_samples, n_features=n_features, centers=n_clusters, random_state=random_state, cluster_std = 3.0, center_box=(-2.0, 2.0))
    
    return X.astype(np.float64), y_true


def generate_synthetic_data_lr(n_samples, n_features, n_classes, seed,
                               class_sep=1.0, flip_y=0.05,
                               informative_ratio=0.6, redundant_ratio=0.2,
                               imbalance=0.0):
    """
    Logistic-regression–friendly synthetic data.
    Returns (X, y) with standardized features (float64).
    """
    n_inform = max(1, int(informative_ratio * n_features))
    n_redund = max(0, int(redundant_ratio * n_features))

    weights = None
    if imbalance > 0 and n_classes == 2:
        weights = [min(0.95, 0.5 + imbalance/2.)]

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_inform,
        n_redundant=n_redund,
        n_repeated=0,
        n_classes=n_classes,
        n_clusters_per_class=2,
        class_sep=class_sep,
        flip_y=flip_y,
        weights=weights,
        random_state=seed,
    )
    X = StandardScaler().fit_transform(X).astype(np.float64)
    return X, y
  
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
    ("SYNTH_C_5_F_5_n100k", 1000_000, 5,  5, 0),
    # ("SYNTH_C_80_F_5_n100k", 10000_00, 5, 80, 1),
    # ("SYNTH_C_30_F_5_n100k", 10000_00, 5, 40, 1),
    # ("SYNTH_C_5_F_50_n100k", 10000_00, 100, 5, 1),
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





