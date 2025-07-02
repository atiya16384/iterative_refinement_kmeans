import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd
from aoclda.sklearn import skpatch
skpatch() # Apply AOCL patch before any KMeans usage
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score, silhouette_score, davies_bouldin_score
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore")
from scipy.spatial import Voronoi, voronoi_plot_2d
import sys
import os

# CONFIGURATION PARAMETERS 
dataset_sizes = [100000, 200000]
n_clusters_list = [5, 8]
n_features_list = [3, 30]  # We keep 2 here for proper plotting 
max_iter = 120

tol_fixed_A = 1e-16
max_iter_A = 300
cap_grid = [0, 50, 100, 150, 200, 250, 300]

max_iter_B = 1000
tol_double_B = 1e-7
tol_single_grid = [1e-1, 1e-3, 1e-5, 1e-7, 1e-9]

n_repeats = 3
rng_global = np.random.default_rng(0)

load_3D_Road_Network_data = '3D_spatial_network.txt'
load_3D_Road_Network= pd.read_csv(load_3D_Road_Network_data)


# Real-dataset
real_datasets = {
    "3D_ROAD": load_3d_road,
    "SUSY":    load_susy,
}



# Define dictionary of precisions
precisions = {
    "Single Precision": np.float32,
    "Double Precision": np.float64
}

def generate_data(n_samples, n_features, n_clusters, random_state):
    X, y_true = make_blobs(n_samples=n_samples, n_features=n_features, centers=n_clusters, random_state=random_state)
    
    return X.astype(np.float64), y_true

def evaluate_metrics(X, labels, y_true, inertia):
    ari = adjusted_rand_score(y_true, labels)
    silhouette_avg = silhouette_score(X, labels)
    db_index = davies_bouldin_score(X, labels)
    return ari, silhouette_avg, db_index, inertia

# ── helpers to read the two local CSVs ───────────────────────────────
import pathlib
DATA_DIR = pathlib.Path(".")          # change if the CSVs live elsewhere

def load_3d_road(n_rows=1_000_000):

    path = DATA_DIR / "3D_spatial_network.csv"
    X = pd.read_csv(path, sep=r"\s+|,", engine="python",  # auto detect space or comma
                    header=None, usecols=[0, 1, 2],
                    nrows=n_rows, dtype=np.float64).to_numpy()
    return X, None
    
def load_susy(n_rows=1_000_000):

    path = DATA_DIR / "SUSY.csv"
    df = pd.read_csv(path, header=None, nrows=n_rows,
                     dtype=np.float64, names=[f"c{i}" for i in range(9)])
    X = df.iloc[:, 1:].to_numpy()     # drop label col
    return X, None

def plot_clusters(X, labels, centers, title="", do_plot = True):
    if not do_plot or X.shape[1] != 2:
        print("⤷  Skipping plot (data not 2-D)")
        return

    # decision boundary mesh
    h = 0.02
    x_min, x_max = X[:,0].min()-1, X[:,0].max()+1
    y_min, y_max = X[:,1].min()-1, X[:,1].max()+1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = KMeans(n_clusters=len(centers), init=centers,
               n_init=1, max_iter=1).predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(6,5))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap="Pastel2")
    plt.scatter(X[:,0], X[:,1], c=labels, s=5, cmap="Dark2")
    plt.scatter(centers[:,0], centers[:,1], c="k", s=80, marker="x")
    plt.title(title)
    plt.tight_layout()
    plt.show()

# FULL DOUBLE PRECISION RUN
def run_full_double(X, initial_centers, n_clusters, max_iter, tol, y_true):
    start_time = time.time()
    kmeans = KMeans(n_clusters=n_clusters, init=initial_centers, n_init=1, max_iter=max_iter,tol=tol, algorithm='lloyd', random_state=0)
    kmeans.fit(X)
    elapsed = time.time() - start_time
   # speedup = elapsed / elapsed_hybrid
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    inertia = kmeans.inertia_
    ari, silhouette, dbi, inertia = evaluate_metrics(X, labels, y_true, inertia)
    mem_MB_double = X.astype(np.float64).nbytes / 1e6
    return centers, labels, inertia, elapsed, ari, silhouette, dbi, mem_MB_double

# Hybrid precison loop 
def run_adaptive_hybrid(X, initial_centers, n_clusters, max_iter_total, tol_single, tol_double, single_iter_cap, y_true, seed=0):
    # single floating point type
    start_time_single = time.time()
    X_single = X.astype(np.float32)
    # Define the initial centers
    initial_centers_32 = initial_centers.astype(np.float32)
    max_iter_single=max(1,min(single_iter_cap, max_iter_total))
    # K=means algorithm for single precision
    kmeans_single = KMeans(n_clusters=n_clusters, init=initial_centers_32, n_init=1, max_iter=max_iter_single, tol=tol_single,random_state=0, algorithm = 'lloyd'
    )
    # start time 
    kmeans_single.fit(X_single)
    end_time_single = time.time() - start_time_single

    iters_single = kmeans_single.n_iter_
    centers32 = kmeans_single.cluster_centers_

    remaining_iter = max_iter_total - iters_single
    remaining_iter = max(1, remaining_iter)
    start_time_double = time.time()
    X_double = X.astype(np.float64)
    initial_centers_64 = centers32.astype(np.float64)

    kmeans_double = KMeans( n_clusters=n_clusters, init=initial_centers_64, n_init=1, max_iter=remaining_iter, tol=tol_double, random_state=seed, algorithm= 'lloyd')
    kmeans_double.fit(X_double)
    end_time_double = time.time() - start_time_double

    labels_final = kmeans_double.labels_
    centers_final = kmeans_double.cluster_centers_
    inertia = kmeans_double.inertia_

    ari, sil, dbi, inertia = evaluate_metrics(X, labels_final, y_true, inertia)
    mem_MB_double = X_double.nbytes / 1e6
    mem_MB_total = mem_MB_double + X_single.nbytes / 1e6
    center_diff = np.linalg.norm(centers_final - initial_centers_64)

    total_time = end_time_single + end_time_double
    total_iters = iters_single + kmeans_double.n_iter_
    return (total_iters, total_time, mem_MB_total, ari, sil, dbi, inertia, center_diff, labels_final, centers_final, mem_MB_total)

def run_one_dataset(ds_name: str, n_clusters, X_full: np.ndarray, y_full):
    rows = []
    for n_samples in dataset_sizes:
        if n_samples > len(X_full):
            continue
        sel = rng_global.choice(len(X_full), n_samples, replace=False)
        X_ns = X_full[sel]
        y_ns = None if y_full is None else y_full[sel]
        for n_clusters in n_clusters_list:
            for n_features in n_features_list:
                if X_ns.shape[1] < n_features and ds_name != "SYNTH":
                    continue
                if ds_name == "SYNTH":
                     X, y_true = generate_data(n_samples, n_features, n_clusters, random_state = rep)           
                else:
                    X = X_ns[:, :n_features]
                    y=y_ns
                
                # Compute initial centers once
                init_kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=1, random_state=0,  max_iter =1)
                #init_Z_kmeans = init_kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
                #init_Z_kmeans = init_Z_kmeans.reshape(xx.shape)
                initial_fit = init_kmeans.fit(X)
                initial_centers = init_kmeans.cluster_centers_

                option = "A"
                for cap in cap_grid:
                    for rep in range(n_repeats):

                        # Full double precision run
                        centers_double, labels_double, inertia, elapsed, mem_MB_double, ari, silhouette, dbi = run_full_double(
                            X, initial_centers, n_clusters, max_iter, tol_fixed_A, y_true
                        )

                        rows.append([n_samples, n_clusters, n_features, "A", cap, "Double", max_iter_A, elapsed, mem_MB_double, 
                                        ari, silhouette, dbi, inertia, 0])

                        # Adaptive hybrid run
                        iter_num, elapsed_hybrid, mem_MB_hybrid, ari_hybrid, silhouette_hybrid, dbi_hybrid, inertia_hybrid, center_diff, labels_hybrid, centers_hybrid, mem_MB_hybrid = run_adaptive_hybrid(
                        X, initial_centers, n_clusters, max_iter_A, single_iter_cap=cap, tol_single=tol_fixed_A, tol_double=tol_fixed_A,y_true=y_true, seed = rep
                        )

                        rows.append([n_samples, n_clusters, n_features, "A", cap, "AdaptiveHybrid", iter_num, elapsed_hybrid, mem_MB_hybrid,
                                        ari_hybrid, silhouette_hybrid, dbi_hybrid, inertia_hybrid, center_diff])
                        
                        # plot clusters
                        if n_features == 2 and rep ==0:
                            title = (f"{ds_name} * n={n_samples} * k={n_clusters}"
                                     f"{'cap='+str(cap) if option =='A' else 'tol='+str(tol_s)}")
                            plot_clusters(X, labels_hybrid, centers_hybrid, title)
                    
                option = "B"
                for tol_s in tol_single_grid:
                    for rep in range(n_repeats):
                               # Full double precision run
                        centers_double, labels_double, inertia, elapsed, mem_MB_double, ari, silhouette, dbi = run_full_double(
                            X, initial_centers, n_clusters, max_iter_B, tol_double_B, y_true
                        )

                        rows.append([n_samples, n_clusters, n_features, "B", tol_s, "Double", max_iter_B, elapsed, mem_MB_double, max_iter_B,
                                        ari, silhouette, dbi, inertia, 0])

                        # Adaptive hybrid run
                        iter_num, elapsed_hybrid, mem_MB_hybrid, ari_hybrid, silhouette_hybrid, dbi_hybrid, inertia_hybrid, center_diff, labels_hybrid, centers_hybrid, mem_MB_hybrid = run_adaptive_hybrid(
                        X, initial_centers, n_clusters, max_iter_B, tol_single=tol_s, tol_double= tol_double_B,single_iter_cap=300, y_true=y_true, seed = rep
                        )

                        rows.append([n_samples, n_clusters, n_features, "B", tol_s, "AdaptiveHybrid", max_iter_B, iter_num, elapsed_hybrid, mem_MB_hybrid,
                                        ari_hybrid, silhouette_hybrid, dbi_hybrid, inertia_hybrid, center_diff])
                        
                        # plot clusters
                        if n_features == 2 and rep ==0:
                            title = (f"{ds_name} * n={n_samples} * k={n_clusters}"
                                     f"{'cap='+str(cap) if option =='B' else 'tol='+str(tol_s)}")
                            plot_clusters(X, labels_hybrid, centers_hybrid, title)
    return rows

all_rows = []
all_rows += run_one_dataset("SYNTH", 8, np.empty((1,1)), None)
all_rows += run_one_dataset("SYNTH", 30, np.empty((1,1)), None )

# real datasets
for tag, loader_fn in real_datasets.items():
    print("loading")
    X_real, y_real = loader_fn()
    all_rows += run_one_dataset(tag, X_real, y_real)

# Data frame and output
columns = ['DatasetSize', 'NumClusters', 'NumFeatures', 'Suite', 'SweepVal', 'Mode', 'SwitchIter',
           'Time', 'Memory_MB', 'ARI', 'Silhouette', 'DBI', 'Inertia', 'CenterDiff']

results_df = pd.DataFrame(all_rows, columns=columns)

# Print summary
print("\n==== SUMMARY ====")
print(results_df.groupby(['DatasetSize','NumClusters','NumFeatures','Mode', 'SwitchIter', 'CenterDiff' ])[['Time','Memory_MB','ARI','Silhouette', 'Inertia']].mean())

# Save results to CSV file
results_df.to_csv("hybrid_kmeans_results.csv", index=False)
for tag in results_df.DatasetName.unique():
    results_df[results_df.DatasetName == tag].to_csv(f"results_{tag}.csv", index=False)
print("CSVS written:", list(results_df.DatasetName.unique()))

print("\nResults saved to 'hybrid_kmeans_results.csv'")
print(os.getcwd())
