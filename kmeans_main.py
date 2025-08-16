import matplotlib.pyplot as plt
import pandas as pd
from aoclda.sklearn import skpatch
skpatch() # Apply AOCL patch before any KMeans usage
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore")
import os
import pathlib
import numpy as np
from visualisations.kmeans_visualisations import KMeansVisualizer

from experiments.kmeans_experiments import (
    run_experiment_A, run_experiment_B,
    run_experiment_C, run_experiment_D, run_experiment_E, run_experiment_F
)
from datasets.utils import (
    generate_synthetic_data, real_datasets, synth_specs, load_3d_road, load_susy,
    columns_A, columns_B, columns_C, columns_D, columns_E, columns_F
)

RESULTS_DIR = pathlib.Path("Results")
RESULTS_DIR.mkdir(exist_ok=True)

PLOTS_DIR= pathlib.Path("ClusterPlots")
PLOTS_DIR.mkdir(exist_ok = True)

# for the cluster size we are varying this for all datasets
n_clusters_list = [80]

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
    "cap_C_pct": int(300 * 0.8),

    # D (Adaptive Hybrid – global switch)
    "max_iter_D": 300,
    "tol_double_baseline_D": 1e-16,
    "chunk_single_D": 20,
    "improve_threshold_D": 1e-3,
    "shift_tol_D": 1e-3,
    "stability_threshold_D": 0.02,
    
  # --- E (Mini-batch Hybrid) : sweep these ---
    "E_mb_iter_grid":   [25, 50, 100, 150],   # was 100
    "E_batch_grid":     [512, 1024, 2048],    # optional sweep
    "E_refine_grid":    [50, 100, 150],       # was 100
    "tol_double_baseline_E": 0.0,             # keep; baseline stops by budget

    # keep single “default” values used if grids not provided
    "mb_iter_E": 100, "mb_batch_E": 2048, "max_refine_iter_E": 100,

    # --- F (Per‑cluster Mixed) : sweep these ---
    "max_iter_F": 300,
    "F_cap_grid":       [0, 25, 50, 75, 100, 150],  # Phase‑1 cap
    "F_tol_single_grid":[1e-2, 1e-3, 5e-4],         # stability tol (log-x)
    "tol_double_F":     1e-4,
    "freeze_stable_F":  True,
    "freeze_patience_F":1,

     # keep single defaults used if grids not provided
    "single_iter_cap_F": 100, "tol_single_F": 1e-3,
  
}

rng_global = np.random.default_rng(0)

# Define dictionary of precisions
precisions = {
    "Single Precision": np.float32,
    "Double Precision": np.float64
}

def run_one_dataset(ds_name: str, X_full: np.ndarray, y_full, rows_D, rows_E, rows_F): #rows_A, rows_B # 
    X_ns, y_ns = X_full, y_full

    n_features = X_ns.shape[1]
    for n_clusters in n_clusters_list:      
            X_cur = X_ns  
            y_true_cur = y_ns          

            print(f"→ n={len(X_ns)} F={n_features} C={n_clusters}  "f"({ds_name})", flush=True)
            print(f"The total number of features is : F={n_features}")
            
            np.random.seed(0)
            # random_indices = np.random.choice(X_cur.shape[0], size=n_clusters, replace=False)
            # initial_centers = X_cur[random_indices].copy()

            init_kmeans = KMeans(n_clusters=n_clusters, init='random', n_init=1, random_state=0,  max_iter = 1)
            initial_fit = init_kmeans.fit(X_cur)
            initial_centers = init_kmeans.cluster_centers_

            # print("Running A")
            # rows_A += run_experiment_A(ds_name, X_cur, y_true_cur, n_clusters, initial_centers, config)
            # print("Running B")
            # rows_B += run_experiment_B(ds_name, X_cur, y_true_cur, n_clusters, initial_centers, config)
            # print("Running C")
            # rows_C += run_experiment_C(ds_name, X_cur, y_true_cur, n_clusters, initial_centers, config)
            print("Running D")
            rows_D += run_experiment_D(ds_name, X_cur, y_true_cur, n_clusters, initial_centers, config)
            print("Running E")
            rows_E += run_experiment_E(ds_name, X_cur, y_true_cur, n_clusters, initial_centers, config)
            print("Running F")
            rows_F += run_experiment_F(ds_name, X_cur, y_true_cur, n_clusters, initial_centers, config)


    return  rows_D, rows_E, rows_F

all_rows = []

# rows_A = []
# rows_B = []
# rows_C = []
rows_D = []
rows_E, rows_F= [], []

# for tag, n, d, k, seed in synth_specs:
#     X, y = generate_synthetic_data(n, d, k, seed)
#     print(f"[SYNTH] {tag:14s}  shape={X.shape}  any_NaN={np.isnan(X).any()}",
#           flush=True)
#     # check if the mappings are correct to the run_one_dataset
#     run_one_dataset(tag, X, y,rows_D, rows_E, rows_F)

# real datasets
# for tag, loader in real_datasets.items():
#     X_real, y_real = loader()
#     run_one_dataset(tag, X_real, y_real, rows_A, rows_B, rows_C, rows_D)

# df_A = pd.DataFrame(rows_A, columns=columns_A)
# df_B = pd.DataFrame(rows_B, columns=columns_B)
# df_C= pd.DataFrame(rows_C, columns=columns_C)
df_D = pd.DataFrame(rows_D, columns=columns_D)
df_E = pd.DataFrame(rows_E, columns=columns_E)
df_F = pd.DataFrame(rows_F, columns=columns_F)



# df_A = pd.read_csv("Results/hybrid_kmeans_Results_expA.csv")
# df_B = pd.read_csv("Results/hybrid_kmeans_Results_expB.csv")
# df_C = pd.read_csv("Results/hybrid_kmeans_Results_expC.csv")
df_D = pd.read_csv("Results/hybrid_kmeans_Results_expD.csv")
df_E = pd.read_csv("Results/hybrid_kmeans_Results_expE.csv")
df_F = pd.read_csv("Results/hybrid_kmeans_Results_expF.csv")

print("Saved:")
print("- hybrid_kmeans_results_expA.csv")
print("- hybrid_kmeans_results_expB.csv")
print("- hybrid_kmeans_Results_expE.csv")
print("- hybrid_kmeans_Results_expF.csv")



# === SUMMARY: Experiment A ===
# print("\n==== SUMMARY: EXPERIMENT A ====")
# print(df_A.groupby([
#     'DatasetSize', 'NumClusters', 'Mode', 'Cap',
#     'tolerance_single', 'iter_single', 'iter_double', 'Suite'
# ])[['Time', 'Memory_MB', 'Inertia']].mean())

# # === SUMMARY: Experiment B ===
# print("\n==== SUMMARY: EXPERIMENT B ====")
# print(df_B.groupby([
#     'DatasetSize', 'NumClusters', 'Mode',
#     'tolerance_single', 'iter_single', 'iter_double', 'Suite'
# ])[['Time', 'Memory_MB', 'Inertia']].mean())s

# === SUMMARY: Experiment C ===
# print("\n==== SUMMARY: EXPERIMENT C ====")
# print(df_C.groupby([
#     'DatasetSize', 'NumClusters', 'Mode', 'Cap',
#     'tolerance_single', 'iter_single', 'iter_double', 'Suite'
# ])[['Time', 'Memory_MB','Inertia']].mean())

# === SUMMARY: Experiment D ===
print("\n==== SUMMARY: EXPERIMENT D ====")
print(df_D.groupby([
    'DatasetSize', 'NumClusters', 'Mode',
    'tolerance_single','iter_single', 'iter_double', 'Suite'
])[['Time', 'Memory_MB', 'Inertia']].mean())

print("\n==== SUMMARY: EXPERIMENT E ====")
print(df_E.groupby(['DatasetSize','NumClusters','Mode','MB_Iter','MB_Batch','RefineIter','Suite'])
        [['Time','Memory_MB','Inertia']].mean())

print("\n==== SUMMARY: EXPERIMENT F ====")
print(df_F.groupby(['DatasetSize','NumClusters','Mode','tol_single','tol_double','single_iter_cap',
                    'freeze_stable','freeze_patience','Suite'])
        [['Time','Memory_MB','Inertia']].mean())



# Plots for Experiment A and C (Cap-based)
# Cap-based plots (Experiments A & C)

kmeans_vis = KMeansVisualizer()
# kmeans_vis.plot_cap_vs_time(df_A)
# kmeans_vis.plot_hybrid_cap_vs_inertia(df_A)
# kmeans_vis.plot_tolerance_vs_inertia(df_B)
# kmeans_vis.plot_tolerance_vs_time(df_B)
# kmeans_vis.plot_cap_percentage_vs_inertia(df_C)
# kmeans_vis.plot_cap_percentage_vs_time(df_C)
kmeans_vis.plot_expD(df_D)
kmeans_vis.plot_expE(df_E)
kmeans_vis.plot_expF(df_F)


print(os.getcwd())

