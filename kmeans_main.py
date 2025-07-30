import matplotlib.pyplot as plt
import pandas as pd
from aoclda.sklearn import skpatch
skpatch() # Apply AOCL patch before any KMeans usage
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore")
from scipy.spatial import Voronoi, voronoi_plot_2d
import sys
import os
import pathlib
from sklearn.decomposition import PCA
import time
import numpy as np
from visualisations.kmeans_visualisations import (
    plot_cap_vs_time,
    plot_hybrid_cap_vs_inertia,
    plot_tolerance_vs_time,
    plot_tolerance_vs_inertia,
    plot_with_ci,
    boxplot_comparison
)

from experiments.kmeans_experiments import (
    run_experiment_A, run_experiment_B,
    run_experiment_C, run_experiment_D
)
from datasets.kmeans_datasets import (
    generate_synthetic_data, real_datasets, synth_specs,
    columns_A, columns_B, columns_C, columns_D
)

DATA_DIR = pathlib.Path(".")         
RESULTS_DIR = pathlib.Path("Results")
RESULTS_DIR.mkdir(exist_ok=True)

PLOTS_DIR= pathlib.Path("ClusterPlots")
PLOTS_DIR.mkdir(exist_ok = True)

# CONFIGURATION PARAMETERS 
dataset_sizes = [100000]
# for the cluster size we are varying this for all datasets
n_clusters_list = [30]

max_iter = 300
# Understand what the experiment parameters mean

config = {
    "n_repeats": 1,
    "cap_grid": [0, 50, 100, 150, 200, 250, 300],
    "tol_fixed_A": 1e-16,
    "max_iter_A": 300,

    "max_iter_B": 1000,
    "tol_double_B": 1e-5,
    "tol_single_grid": [1e-1, 1e-2, 1e-3, 1e-4],

    "max_iter_C": 300,
    "tol_fixed_C": 1e-16,
    "cap_C": int(300 * 0.8),

    "max_iter_D": 1000,
    "tol_double_D": 1e-3,
    "tol_single_D": 0.8 * 1e-3
}

rng_global = np.random.default_rng(0)

# Define dictionary of precisions
precisions = {
    "Single Precision": np.float32,
    "Double Precision": np.float64
}

def run_one_dataset(ds_name: str, X_full: np.ndarray, y_full, rows_A, rows_B, rows_C, rows_D):
    if ds_name.startswith("SYNTH"):
        sample_sizes = dataset_sizes
    else:
        sample_sizes = [len(X_full)]
    
    print(f"\n=== Starting dataset: {ds_name}  |  total rows={len(X_full):,} ===",
          flush=True)

    for n_samples in sample_sizes:
        if n_samples < len(X_full):
            sel      = rng_global.choice(len(X_full), n_samples, replace=False)
            X_ns     = X_full[sel]
            y_ns     = None if y_full is None else y_full[sel]
        else:
            X_ns, y_ns = X_full, y_full

        n_features = X_ns.shape[1]
        for n_clusters in n_clusters_list:      

                X_cur = X_ns  
                y_true_cur = y_ns          

                print(f"→ n={n_samples:,} F={n_features} C={n_clusters}  "f"({ds_name})", flush=True)
                print(f"The total number of features is : F={n_features}")
                
                np.random.seed(0)
                # random_indices = np.random.choice(X_cur.shape[0], size=n_clusters, replace=False)
                # initial_centers = X_cur[random_indices].copy()

                init_kmeans = KMeans(n_clusters=n_clusters, init='random', n_init=1, random_state=0,  max_iter = 1)
                initial_fit = init_kmeans.fit(X_cur)
                initial_centers = init_kmeans.cluster_centers_

                rows_A += run_experiment_A(tag, X, y, n_clusters, initial_centers, config)
                rows_B += run_experiment_B(tag, X, y, n_clusters, initial_centers, config)
                rows_C += run_experiment_C(tag, X, y, n_clusters, initial_centers, config)
                rows_D += run_experiment_D(tag, X, y, n_clusters, initial_centers, config)

                    
    return rows_A, rows_B, rows_C, rows_D

all_rows = []


rows_A = []
rows_B = []
rows_C = []
rows_D = []

for tag, n, d, k, seed in synth_specs:
    X, y = generate_synthetic_data(n, d, k, seed)
    print(f"[SYNTH] {tag:14s}  shape={X.shape}  any_NaN={np.isnan(X).any()}",
          flush=True)
    # check if the mappings are correct to the run_one_dataset
    run_one_dataset(tag, X, y, rows_A, rows_B, rows_C, rows_D)

# real datasets
# for tag, loader in real_datasets.items():
#    print(f"loading {tag} …")
#    X_real, y_real = loader()
#    all_rows += run_one_dataset(tag, X_real, y_real, rows_A, rows_B)

df_A = pd.DataFrame(rows_A, columns=columns_A)
df_B = pd.DataFrame(rows_B, columns=columns_B)
df_C= pd.DataFrame(rows_C, columns=columns_C)
df_D = pd.DataFrame(rows_D, columns=columns_D)

df_A.to_csv(RESULTS_DIR / "hybrid_kmeans_results_expA.csv", index=False)
df_B.to_csv(RESULTS_DIR / "hybrid_kmeans_results_expB.csv", index=False)
df_C.to_csv(RESULTS_DIR / "hybrid_kmeans_results_expC.csv", index=False)
df_D.to_csv(RESULTS_DIR / "hybrid_kmeans_results_expD.csv", index=False)

print("Saved:")
print("- hybrid_kmeans_results_expA.csv")
print("- hybrid_kmeans_results_expB.csv")

# === SUMMARY: Experiment A ===
print("\n==== SUMMARY: EXPERIMENT A ====")
print(df_A.groupby([
    'DatasetSize', 'NumClusters', 'Mode', 'Cap',
    'tolerance_single', 'iter_single', 'iter_double', 'Suite'
])[['Time', 'Memory_MB', 'ARI', 'DBI', 'Inertia']].mean())

# === SUMMARY: Experiment B ===
print("\n==== SUMMARY: EXPERIMENT B ====")
print(df_B.groupby([
    'DatasetSize', 'NumClusters', 'Mode',
    'tolerance_single', 'iter_single', 'iter_double', 'Suite'
])[['Time', 'Memory_MB', 'ARI', 'DBI', 'Inertia']].mean())

# === SUMMARY: Experiment C ===
print("\n==== SUMMARY: EXPERIMENT C ====")
print(df_C.groupby([
    'DatasetSize', 'NumClusters', 'Mode', 'Cap',
    'tolerance_single', 'iter_single', 'iter_double', 'Suite'
])[['Time', 'Memory_MB', 'ARI', 'DBI', 'Inertia']].mean())

# === SUMMARY: Experiment D ===
print("\n==== SUMMARY: EXPERIMENT D ====")
print(df_D.groupby([
    'DatasetSize', 'NumClusters', 'Mode',
    'tolerance_single', 'Cap', 'iter_single', 'iter_double', 'Suite'
])[['Time', 'Memory_MB', 'ARI', 'DBI', 'Inertia']].mean())


# how to plot for the different types of graphs that we have
# Plots for Experiment A and C (Cap-based)
# Cap-based plots (Experiments A & C)
plot_cap_vs_time("Results/hybrid_kmeans_results_expA.csv")
plot_hybrid_cap_vs_inertia("Results/hybrid_kmeans_results_expA.csv")

plot_cap_vs_time("Results/hybrid_kmeans_results_expC.csv")
plot_hybrid_cap_vs_inertia("Results/hybrid_kmeans_results_expC.csv")

# Tolerance-based plots (Experiments B & D)
plot_tolerance_vs_time("Results/hybrid_kmeans_results_expB.csv")
plot_tolerance_vs_inertia("Results/hybrid_kmeans_results_expB.csv")

plot_tolerance_vs_time("Results/hybrid_kmeans_results_expD.csv")
plot_tolerance_vs_inertia("Results/hybrid_kmeans_results_expD.csv")


print("\nResults saved to 'hybrid_kmeans_results_expA.csv")
print("\nResults saved to 'hybrid_kmeans_results_expB.csv")
print(os.getcwd())

