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
from visualisation.kmeans_visualisation import (
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

def evaluate_metrics(X, labels, y_true, inertia):
    if y_true is None:
        ari = np.nan                
    else:
        ari = adjusted_rand_score(y_true, labels)

    db_index = davies_bouldin_score(X, labels)
    return ari, db_index, inertia

# FULL DOUBLE PRECISION RUN
def run_full_double(X, initial_centers, n_clusters, max_iter, tol, y_true):
    start_time = time.time()
    kmeans = KMeans(n_clusters=n_clusters, init=initial_centers, n_init=1, max_iter=max_iter,tol=tol, algorithm='lloyd', random_state=0)
    kmeans.fit(X)
    iters_double_tot = kmeans.n_iter_
    elapsed = time.time() - start_time
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    inertia = kmeans.inertia_
    
    print(f"This is total for double run: {iters_double_tot}")
    iters_single_tot = 0

    ari, dbi, inertia = evaluate_metrics(X, labels, y_true, inertia)
    mem_MB_double = X.astype(np.float64).nbytes / 1e6
    return centers, labels, iters_double_tot, iters_single_tot,  elapsed, mem_MB_double, ari, dbi, inertia, 

# Hybrid precison loop 
def run_hybrid(X, initial_centers, n_clusters, max_iter_total, tol_single, tol_double, single_iter_cap, y_true, seed=0):
    # single floating point type
    start_time_single = time.time()
    X_single = X.astype(np.float32)
    # Define the initial centers
    initial_centers_32 = initial_centers.astype(np.float32)
    # this
    max_iter_single=max(1,min(single_iter_cap, max_iter_total))
    # K=means algorithm for single precision
    kmeans_single = KMeans(n_clusters=n_clusters, init=initial_centers_32, n_init=1, max_iter=max_iter_single, tol=tol_single,random_state=0, algorithm = 'lloyd'
    )
    # start time 
    kmeans_single.fit(X_single)
    end_time_single = time.time() - start_time_single

    iters_single = kmeans_single.n_iter_
    print(f"This is total single iteration: {iters_single}")
    centers32 = kmeans_single.cluster_centers_

    remaining_iter = max_iter_total - iters_single
    remaining_iter = max(1, remaining_iter)
    start_time_double = time.time()
    # X_double = X.astype(np.float64)
    initial_centers_64 = centers32.astype(np.float64)

    kmeans_double = KMeans( n_clusters=n_clusters, init=initial_centers_64, n_init=1, max_iter=remaining_iter, tol=tol_double, random_state=seed, algorithm= 'lloyd')
    kmeans_double.fit(X)
    
    end_time_double = time.time() - start_time_double

    labels_final = kmeans_double.labels_
    centers_final = kmeans_double.cluster_centers_
    inertia = kmeans_double.inertia_

    ari, dbi, inertia = evaluate_metrics(X, labels_final, y_true, inertia)
    mem_MB_double = X.astype(np.float64).nbytes / 1e6
    mem_MB_total = mem_MB_double + X_single.nbytes / 1e6
    iters_double = kmeans_double.n_iter_
    print(f"This is iters_double: {iters_double}")
    total_time = end_time_single + end_time_double
    
    return ( labels_final, centers_final, iters_single, iters_double,total_time, mem_MB_total, ari, dbi, inertia)
    

        
#         Compute residual (center movement)
#         center_shift = norm(centers_single - prev_centers) / norm(prev_centers)
        
#         if center_shift <= residual_tol:
#             converged = True
#             break
    
#     elapsed_single = time.time() - start_total
    
#     # Only switch to double if not sufficiently converged
#     if not converged and (max_iter_total - total_single_iters) > 0:
#         remaining_iters = max_iter_total - total_single_iters
        


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
    run_one_dataset(tag, X, y, rows_A, rows_B)

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

