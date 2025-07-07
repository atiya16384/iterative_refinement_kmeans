import numpy as np
import time
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

DATA_DIR = pathlib.Path(".")         
RESULTS_DIR = pathlib.Path("Results")
RESULTS_DIR.mkdir(exist_ok=True)

PLOTS_DIR= pathlib.Path("ClusterPlots")
PLOTS_DIR.mkdir(exist_ok = True)

def load_3d_road(n_rows=1_000_000):

    path = DATA_DIR / "3D_spatial_network.csv"
    X = pd.read_csv(path, sep=r"\s+|,", engine="python",  
                    header=None, usecols=[0, 1, 2],
                    nrows=n_rows, dtype=np.float64).to_numpy()
    return X, None
    
def load_susy(n_rows=1_000_000):

    path = DATA_DIR / "SUSY.csv"
    df = pd.read_csv(path, header=None, nrows=n_rows,
                     dtype=np.float64, names=[f"c{i}" for i in range(9)])
    X = df.iloc[:, 1:].to_numpy()     # drop label col
    return X, None

# CONFIGURATION PARAMETERS 
dataset_sizes = [100000]
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
    if y_true is None:
        ari = np.nan                
    else:
        ari = adjusted_rand_score(y_true, labels)

    db_index       = davies_bouldin_score(X, labels)
    return ari, db_index, inertia

def plot_clusters(
        X, labels, centers,
        title="",
        do_plot=True,
        filename=None,
        max_grid_pts=4_000_000,   # upper bound for meshgrid points
        max_scatter=25_000        # max points to draw in scatter
):

    if not do_plot or X.shape[1] != 2:
        print("⤷  Skipping plot (not 2-D)")
        return

    # bounding box 
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    nx = int((x_max - x_min) / h) + 1
    ny = int((y_max - y_min) / h) + 1
    need_mesh = (nx * ny) <= max_grid_pts

    # figure 
    plt.figure(figsize=(6, 5))

    if need_mesh:
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, nx),
            np.linspace(y_min, y_max, ny)
        )
        Z = KMeans(
            n_clusters=len(centers),
            init=centers, n_init=1, max_iter=1
        ).predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        plt.contourf(xx, yy, Z, alpha=0.25, cmap="Pastel2")
    else:
        print("⤷  Mesh too large; plotting points only")

    # scatter
    if len(X) > max_scatter:
        sel = np.random.default_rng(0).choice(len(X), max_scatter, replace=False)
        X_plot, labels_plot = X[sel], labels[sel]
    else:
        X_plot, labels_plot = X, labels

    plt.scatter(X_plot[:, 0], X_plot[:, 1], c=labels_plot, s=5, cmap="Dark2")
    plt.scatter(centers[:, 0], centers[:, 1], c="k", s=80, marker="x")
    plt.title(title)
    plt.tight_layout()

    # save 
    if filename:
        out_path = PLOTS_DIR / f"{filename}.png"
        plt.savefig(out_path, dpi=150)
        print(f"⤷  Plot saved to {out_path}")
    plt.close()


def pca_2d_view(X_full, centers_full, random_state=0):
    pca = PCA(n_components=2, random_state=random_state)
    X_vis = pca.fit_transform(X_full)
    centers_vis = pca.transform(centers_full)
    return X_vis, centers_vis

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
    ari, dbi, inertia = evaluate_metrics(X, labels, y_true, inertia)
    mem_MB_double = X.astype(np.float64).nbytes / 1e6
    return centers, labels, inertia, elapsed, ari, dbi, mem_MB_double

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

    ari, dbi, inertia = evaluate_metrics(X, labels_final, y_true, inertia)
    mem_MB_double = X_double.nbytes / 1e6
    mem_MB_total = mem_MB_double + X_single.nbytes / 1e6
    center_diff = np.linalg.norm(centers_final - initial_centers_64)

    total_time = end_time_single + end_time_double
    total_iters = iters_single + kmeans_double.n_iter_
    return (total_iters, total_time, mem_MB_total, ari, dbi, inertia, center_diff, labels_final, centers_final, mem_MB_total)
    
def run_one_dataset(ds_name: str, X_full: np.ndarray, y_full):
    rows = []
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

        for n_clusters in n_clusters_list:
            for n_features in n_features_list:

                if X_ns.shape[1] < n_features and not ds_name.startswith("SYNTH"):
                    continue            

                X_cur = X_ns[:, :n_features]   
                y_true_cur = y_ns          

                print(f"→ n={n_samples:,}  k={n_clusters}  d={n_features}  "f"({ds_name})", flush=True)

                init_kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=1, random_state=0,  max_iter =1)
                initial_fit = init_kmeans.fit(X_cur)
                initial_centers = init_kmeans.cluster_centers_

                option = "A"
                for cap in cap_grid:
                    for rep in range(n_repeats):
                        # Full double precision run
                        centers_double, labels_double, inertia, elapsed, mem_MB_double, ari,  dbi = run_full_double(
                        X_cur, initial_centers, n_clusters, max_iter, tol_fixed_A, y_true_cur
                            )

                        rows.append([ds_name, n_samples, n_clusters, n_features, "A", cap, "Double", max_iter_A, elapsed, mem_MB_double, 
                                        ari,  dbi, inertia, 0])
                    
                        print(f" [Double] {rows}", flush=True) 
                        # Adaptive hybrid run
                        iter_num, elapsed_hybrid, mem_MB_hybrid, ari_hybrid, dbi_hybrid, inertia_hybrid, center_diff, labels_hybrid, centers_hybrid, mem_MB_hybrid = run_adaptive_hybrid(
                        X_cur, initial_centers, n_clusters, max_iter_total = max_iter_A, single_iter_cap=cap, tol_single = tol_fixed_A, tol_double=tol_fixed_A, y_true = y_true_cur, seed = rep
                            )

                        rows.append([ds_name, n_samples, n_clusters, n_features, "A", cap, "AdaptiveHybrid", iter_num, elapsed_hybrid, mem_MB_hybrid,
                                        ari_hybrid, dbi_hybrid, inertia_hybrid, center_diff])
                        
                        print(f" [Hybrid] {rows}", flush=True) 
                        # plot clusters
                        if rep == 0:
                            X_vis, centers_vis = pca_2d_view(X_cur, centers_hybrid)
                            filename = (f"{ds_name}_n{n_samples}_k{n_clusters}_A_{cap}")
                            title = (f"{ds_name}: n={n_samples}, k={n_clusters}, "f"cap={cap}")
                            plot_clusters(X_vis, labels_hybrid, centers_vis, title=title, filename=filename)

                    option = "B"
                    for tol_s in tol_single_grid:
                        for rep in range(n_repeats):
                               # Full double precision run
                            centers_double, labels_double, inertia, elapsed, mem_MB_double, ari, dbi = run_full_double(
                                X_cur, initial_centers, n_clusters, max_iter_B, tol_double_B, y_true_cur
                            )

                            rows.append([ds_name, n_samples, n_clusters, n_features, "B", tol_s, "Double", elapsed, mem_MB_double, max_iter_B,
                                        ari, dbi, inertia, 0])
                            print(f" [Double] {rows}", flush=True) 
                            # Adaptive hybrid run
                            iter_num, elapsed_hybrid, mem_MB_hybrid, ari_hybrid, dbi_hybrid, inertia_hybrid, center_diff, labels_hybrid, centers_hybrid, mem_MB_hybrid = run_adaptive_hybrid(
                            X_cur, initial_centers, n_clusters, max_iter_total=max_iter_B, tol_single = tol_s, tol_double = tol_double_B, single_iter_cap=300, y_true= y_true_cur, seed = rep
                            )

                            rows.append([ds_name, n_samples, n_clusters, n_features, "B", tol_s, "AdaptiveHybrid", iter_num, elapsed_hybrid, mem_MB_hybrid,
                                        ari_hybrid, dbi_hybrid, inertia_hybrid, center_diff])
                        
                            print(f" [Hybrid] {rows}", flush=True) 

                            # plot clusters
                            if rep == 0:
                                X_vis, centers_vis = pca_2d_view(X_cur, centers_hybrid)
                                filename = (f"{ds_name}_n{n_samples}_k{n_clusters}_B_tol{tol_s:g}")
                                title = (f"{ds_name}: n={n_samples}, k={n_clusters},  tol={tol_s:g}")
                                plot_clusters(X_vis, labels_hybrid, centers_vis, title=title, filename=filename)


    return rows

all_rows = []

synth_specs = [
    ("SYNTH_K5_n100k" , 100_000, 30,  5, 0),
    ("SYNTH_K30_n100k", 100_000, 30, 30, 1),
]

all_rows = []

for tag, n, d, k, seed in synth_specs:
    X, y = generate_data(n, d, k, random_state=seed)
    print(f"[SYNTH] {tag:14s}  shape={X.shape}  any_NaN={np.isnan(X).any()}",
          flush=True)
    all_rows += run_one_dataset(tag, X, y)

# real datasets
for tag, loader in real_datasets.items():
    print(f"loading {tag} …")
    X_real, y_real = loader()
    all_rows += run_one_dataset(tag, X_real, y_real)

# Data frame and output
columns = ['DatasetName', 'DatasetSize', 'NumClusters', 'NumFeatures', 'Suite', 'SweepVal', 'Mode', 'SwitchIter',
           'Time', 'Memory_MB', 'ARI',  'DBI', 'Inertia', 'CenterDiff']

results_df = pd.DataFrame(all_rows, columns=columns)

# Print summary
print("\n==== SUMMARY ====")
print(results_df.groupby(['DatasetSize','NumClusters','NumFeatures','Mode', 'SwitchIter', 'CenterDiff' ])[['Time','Memory_MB','ARI', 'Inertia']].mean())

# Save results to CSV file
results_df.to_csv(RESULTS_DIR / "hybrid_kmeans_results.csv", index=False)
for tag in results_df.DatasetName.unique():
    out_path = RESULTS_DIR / f"results_{tag}.csv"
    results_df[results_df.DatasetName == tag].to_csv(out_path, index=False)
print("CSVS written:", list(results_df.DatasetName.unique()))

print("\nResults saved to 'results/hybrid_kmeans_results.csv'")
print(os.getcwd())
