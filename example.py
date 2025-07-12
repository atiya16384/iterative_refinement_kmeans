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

CONV_DIR = pathlib.Path("ConvergencePlots")
CONV_DIR.mkdir(exist_ok=True)

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
    X = df.iloc[:, 1:].to_numpy()     
    return X, None

# CONFIGURATION PARAMETERS 
dataset_sizes = [100000]
n_clusters_list = [5]
n_features_list = [30]  
max_iter = 300

tol_fixed_A = 1e-16
max_iter_A = 300
cap_grid = [1, 50, 100, 150, 200, 250, 300]

max_iter_B = 1000
tol_double_B = 1e-7
tol_single_grid = [1e-1, 1e-3, 1e-5, 1e-7]

n_repeats = 1
rng_global = np.random.default_rng(0)

# Real-dataset
real_datasets = {
    "3D_ROAD": load_3d_road,
    # "SUSY":    load_susy,
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

    db_index = davies_bouldin_score(X, labels)
    return ari, db_index, inertia

def plot_clusters(
        X, labels, centers,
        title="",
        do_plot=True,
        filename=None,
        max_grid_pts=4_000_000,   
        max_scatter=25_000):

    if not do_plot or X.shape[1] != 2:
        print("Skipping plot (not 2-D)")
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
        print("Mesh too large; plotting points only")

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
        print(f"Plot saved to {out_path}")
    plt.close()

def plot_hybrid_cap_vs_inertia(results_path = "Results/hybrid_kmeans_results_expA.csv", output_dir = "Results"):
    output_dir = pathlib.Path(output_dir)
    df=pd.read_csv(results_path)
    df_hybrid = df[df["Suite"]== "AdaptiveHybrid"]

    plt.figure(figsize=(7,5))
    for name, group in df_hybrid.groupby("DatasetName"):
        group_sorted = group.sort_values("Cap")
        plt.plot(group_sorted["Cap"], group_sorted["Inertia"], marker = 'o', label=name)
    
    plt.title("Cap vs Intertia (Adaptive Hybrid)")
    plt.xlabel("Cap (Single Precision Iteration Cap)")
    plt.ylabel("Final Inertia")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    filename = output_dir / "cap_vs_inertia_hybrid.png"
    plt.savefig(filename)
    plt.close()
    print(f"saved: {filename}")

def plot_cap_vs_time(results_path="Results/hybrid_kmeans_results_expA.csv", output_dir="Results"):
    output_dir = pathlib.Path(output_dir)
    df = pd.read_csv(results_path)
    df_hybrid = df[df["Suite"] == "AdaptiveHybrid"]

    plt.figure(figsize=(7,5))
    for name, group in df_hybrid.groupby("DatasetName"):
        group_sorted = group.sort_values("Cap")
        plt.plot(group_sorted["Cap"], group_sorted["Time"], marker = 'o', label=name)
    
    plt.title("Cap vs Time (Adaptive Hybrid)")
    plt.xlabel("Cap (Single Precision Iteration Cap)")
    plt.ylabel("Total Time (seconds)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    filename = output_dir / "cap_vs_time_hybrid.png"
    plt.savefig(filename)
    plt.close()
    print(f"Saved: {filename}")

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
    
def run_one_dataset(ds_name: str, X_full: np.ndarray, y_full, rows_A, rows_B):

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

                init_kmeans = KMeans(n_clusters=n_clusters, init='random', n_init=1, random_state=0,  max_iter =1)
                initial_fit = init_kmeans.fit(X_cur)
                initial_centers = init_kmeans.cluster_centers_

                for rep in range(n_repeats):

                    # Full double precision run
                    centers_double, labels_double, iters_double_tot, iters_single_tot, elapsed, mem_MB_double, ari,  dbi, inertia = run_full_double(
                    X_cur, initial_centers, n_clusters, max_iter, tol_fixed_A, y_true_cur
                    )

                    rows_A.append([ds_name, n_samples, n_clusters, n_features, "A", 0, 1e-16,  iters_single_tot, iters_double_tot,"Double",  elapsed, mem_MB_double, 
                                        ari,  dbi, inertia])
                                 
                    print(f" [Double] {rows_A}", flush=True) 

                option = "A"
                for cap in cap_grid:
                    for rep in range(n_repeats):

                        # Adaptive hybrid run
                        labels_hybrid, centers_hybrid, iters_single, iters_double, elapsed_hybrid, mem_MB_hybrid, ari_hybrid, dbi_hybrid, inertia_hybrid = run_adaptive_hybrid(
                        X_cur, initial_centers, n_clusters, max_iter_total = max_iter_A, single_iter_cap=cap, tol_single = tol_fixed_A, tol_double=tol_fixed_A, y_true = y_true_cur, seed = rep
                            )
                        
                        print(f"Cap: {cap}, Iter Single: {iters_single}, Iter Double: {iters_double}, Toal: {iters_single + iters_double}")
                    
                        rows_A.append([ds_name, n_samples, n_clusters, n_features, "A", cap, tol_fixed_A, iters_single, iters_double, "AdaptiveHybrid", elapsed_hybrid, mem_MB_hybrid,
                                        ari_hybrid, dbi_hybrid, inertia_hybrid ])
                        
                        print(f" [Hybrid] {rows_A}", flush=True) 
                        # plot clusters
                        # if rep == 0:
                        #     X_vis, centers_vis = pca_2d_view(X_cur, centers_hybrid)
                        #     filename = (f"{ds_name}_n{n_samples}_k{n_clusters}_A_{cap}")
                        #     title = (f"{ds_name}: n={n_samples}, k={n_clusters}, "f"cap={cap}")
                        #     plot_clusters(X_vis, labels_hybrid, centers_vis, title=title, filename=filename)
                    


                    # Full double precision baseline for Experiment B
                for rep in range(n_repeats):
                        centers_double, labels_double, iters_double_tot, iters_single_tot, elapsed, mem_MB_double, ari, dbi, inertia = run_full_double(
                            X_cur, initial_centers, n_clusters, max_iter_B, tol_double_B, y_true_cur
                            )

                        rows_B.append([ ds_name, n_samples, n_clusters, n_features, "B", 0, tol_double_B,  iters_single_tot, iters_double_tot, "Double", elapsed, mem_MB_double,
                                        ari, dbi, inertia])
                        print(f"[Double Baseline - Exp B] tol={tol_double_B} | iter_double={iters_double_tot}")

                option = "B"
                for tol_s in tol_single_grid:
                    for rep in range(n_repeats):
           
                            # Adaptive hybrid run
                            labels_hybrid, centers_hybrid, iters_single, iters_double, elapsed_hybrid, mem_MB_hybrid, ari_hybrid, dbi_hybrid, inertia_hybrid = run_adaptive_hybrid(
                                X_cur, initial_centers, n_clusters, max_iter_total=max_iter_B, tol_single = tol_s, tol_double = tol_double_B, single_iter_cap=max_iter_B, y_true= y_true_cur, seed = rep
                            )

                            print(f"Tol_single: {tol_s}, Iter Single: {iters_single}, Iter Double: {iters_double}, Total: {iters_single + iters_double}")
                           

                            rows_B.append([ds_name, n_samples, n_clusters, n_features, "B", max_iter_B, tol_s,  iters_single, iters_double, "AdaptiveHybrid", elapsed_hybrid, mem_MB_hybrid,
                                        ari_hybrid, dbi_hybrid, inertia_hybrid])
                            
                            print(f" [Hybrid] {rows_B}", flush=True) 

                            # plot clusters
                            # if rep == 0:
                                # X_vis, centers_vis = pca_2d_view(X_cur, centers_hybrid)
                                # filename = (f"{ds_name}_n{n_samples}_k{n_clusters}_B_tol{tol_s:g}")
                                # title = (f"{ds_name}: n={n_samples}, k={n_clusters},  tol={tol_s:g}")
                                # plot_clusters(X_vis, labels_hybrid, centers_vis, title=title, filename=filename)
    return rows_A, rows_B

all_rows = []

synth_specs = [
    ("SYNTH_K5_n100k" , 100_000, 30,  5, 0),
    # ("SYNTH_K30_n100k", 100_000, 30, 30, 1),
]

rows_A = []
rows_B= []

columns_A = [
    'DatasetName', 'DatasetSize', 'NumClusters', 'NumFeatures',
    'Mode', 'Cap', 'tolerance_single', 'iter_single', 'iter_double', 'Suite',
    'Time', 'Memory_MB', 'ARI', 'DBI', 'Inertia'
]

columns_B = [
    'DatasetName', 'DatasetSize', 'NumClusters', 'NumFeatures',
    'Mode', 'tolerance_single', 'iter_single', 'iter_double', 'Suite',
    'Time', 'Memory_MB', 'ARI', 'DBI', 'Inertia'
]

df_A = pd.DataFrame(rows_A, columns=columns_A)
df_B = pd.DataFrame(rows_B, columns=columns_B)

df_A.to_csv(RESULTS_DIR / "hybrid_kmeans_results_expA.csv", index=False)
df_B.to_csv(RESULTS_DIR / "hybrid_kmeans_results_expB.csv", index=False)

print("Saved:")
print("- hybrid_kmeans_results_expA.csv")
print("- hybrid_kmeans_results_expB.csv")


# real datasets
#for tag, loader in real_datasets.items():
 #   print(f"loading {tag} …")
  #  X_real, y_real = loader()
   # all_rows += run_one_dataset(tag, X_real, y_real)


results_df = pd.DataFrame(all_rows, columns=columns)

# === SUMMARY: Experiment A ===
print("\n==== SUMMARY: EXPERIMENT A ====")
print(df_A.groupby([
    'DatasetSize', 'NumClusters', 'NumFeatures', 'Mode', 'Cap',
    'tolerance_single', 'iter_single', 'iter_double', 'Suite'
])[['Time', 'Memory_MB', 'ARI', 'DBI', 'Inertia']].mean())

# === SUMMARY: Experiment B ===
print("\n==== SUMMARY: EXPERIMENT B ====")
print(df_B.groupby([
    'DatasetSize', 'NumClusters', 'NumFeatures', 'Mode',
    'tolerance_single', 'iter_single', 'iter_double', 'Suite'
])[['Time', 'Memory_MB', 'ARI', 'DBI', 'Inertia']].mean())

plot_hybrid_cap_vs_inertia()
plot_cap_vs_time()


print("\nResults saved to 'hybrid_kmeans_results.csv'")
print(os.getcwd())
