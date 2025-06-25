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
dataset_sizes = [10000]
n_clusters_list = [4, 5,6, 7, 8]
n_features_list = [2, 4, 6, 8]  # We keep 2 here for proper plotting 
max_iter = 120
n_repeats = 3

# Define dictionary of precisions
precisions = {
    "Single Precision": np.float32,
    "Double Precision": np.float64
}

def generate_data(n_samples, n_features, n_clusters, random_state):
    X, y_true = make_blobs(n_samples=n_samples, n_features=n_features, centers=n_clusters, random_state=random_state)
    return X, y_true

def evaluate_metrics(X, labels, y_true, inertia):
    ari = adjusted_rand_score(y_true, labels)
    silhouette_avg = silhouette_score(X, labels)
    db_index = davies_bouldin_score(X, labels)
    return ari, silhouette_avg, db_index, inertia

# FULL DOUBLE PRECISION RUN
def run_full_double(X, initial_centers, n_clusters, max_iter, repeat, y_true):
    start_time = time.time()
    kmeans = KMeans(n_clusters=n_clusters, init=initial_centers, n_init=1, max_iter=max_iter, random_state=repeat)
    kmeans.fit(X.astype(precisions["Double Precision"]))
    elapsed = time.time() - start_time
   # speedup = elapsed / elapsed_hybrid
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    inertia = kmeans.inertia_
    ari, silhouette, dbi, inertia = evaluate_metrics(X, labels, y_true, inertia)
    mem_MB_double = X.astype(np.float64).nbytes / 1e6
    return centers, labels, inertia, elapsed, ari, silhouette, dbi, mem_MB_double

# Hybrid precison loop 

def run_adaptive_hybrid( X, initial_centers, n_clusters, max_iter, repeat, y_true, tol_single=1e-7, single_iter_cap=300):
    # single floating point type
    X_single = X.astype(np.float32)
    # Define the initial centers
    initial_centers_32 = initial_centers.astype(np.float32)
    
    # K=means algorithm for single precision
    kmeans_single = KMeans(
        n_clusters=n_clusters,
        init=initial_centers_32,
        n_init=1,
        max_iter=min(single_iter_cap, max_iter),
        tol=tol_single,
        random_state=repeat,
    )
    # start time 
    start_time_single = time.time()
    kmeans_single.fit(X_single)
    end_time_single = time.time() - start_time_single
    iters_single = kmeans_single.n_iter_
    centers32 = kmeans_single.cluster_centers_

    # stop early if we already converged or we have reached the total tolerance
    if iters_single >= max_iter or kmeans_single.n_iter_ < tol_single:
        labels_final = kmeans_single.labels_
        centers_final = centers32.astype(np.float64)  # cast for consistency
        inertia = kmeans_single.inertia_
        ari, sil, dbi, inertia = evaluate_metrics(X, labels_final, y_true, inertia)
        mem_MB = X_single.nbytes / 1e6
        return (iters_single, start_time_single, mem_MB, ari, sil, dbi,
                inertia, 0.0, labels_final, centers_final, mem_MB)

    remaining_iter = max_iter - iters_single
    X_double = X.astype(np.float64)
    initial_centers_64 = centers32.astype(np.float64)

    kmeans_double = KMeans(
        n_clusters=n_clusters,
        init=initial_centers_64,
        n_init=1,
        max_iter=remaining_iter,
        tol=1e-7,  # tighter tol for final polish
        random_state=repeat,
    )
    start_time_double = time.time()
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

# Main loop
results = []

for n_samples in dataset_sizes:
    for n_clusters in n_clusters_list:
        for n_features in n_features_list:
            for repeat in range(n_repeats):
                print(f"Samples={n_samples}, Clusters={n_clusters}, Features={n_features}, Repeat={repeat+1}")

                # Generate data
                X, y_true = generate_data(n_samples, n_features, n_clusters, repeat)

                # Define the decision boundaries of the clusters.
                x_min, x_max = X[:, 0].min() -1 , X[:, 0].max() + 1
                y_min, y_max = X[:, 1].min() -1 , X[:, 1].max() + 1

                xx, yy =np.meshgrid(np.arange(x_min, x_max,0.02 ), np.arange(y_min, y_max, 0.02 ))
                # Visualize raw data (only if n_features=2)
                if n_features == 2:
                    plt.figure(figsize=(8,6))
                    plt.scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis', s=10)
                    plt.title(f"Generated Data - Samples {n_samples}")
                    plt.show()

                # Compute initial centers once
                init_kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=1, random_state=repeat)
                init_kmeans.fit(X.astype(precisions["Double Precision"]))
                #init_Z_kmeans = init_kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
                #init_Z_kmeans = init_Z_kmeans.reshape(xx.shape)
                initial_centers = init_kmeans.cluster_centers_

                # Full double precision run
                centers_double, labels_double, inertia, elapsed, mem_MB_double, ari, silhouette, dbi = run_full_double(
                    X, initial_centers, n_clusters, max_iter, repeat, y_true
                )

                results.append([n_samples, n_clusters, n_features, "Double", 0, elapsed, mem_MB_double,
                                ari, silhouette, dbi, inertia, 0])

                if (n_features == 2):
                    plt.figure(figsize=(8,6))
                    # plt.contour(xx, yy, init_Z_kmeans, alpha = 0.6)
                    plt.scatter(X[:, 0], X[:, 1], c=labels_double, cmap='viridis', s=10)
                    plt.scatter(centers_double[:, 0], centers_double[:, 1], c='red', marker='x', s=100)
                    plt.title("Double Precision Clusters")
                    plt.show()
                
                # Normalize X before processing
                X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
                

                 # Adaptive hybrid run
                iter_num, elapsed_hybrid, mem_MB_hybrid, ari_hybrid, silhouette_hybrid, dbi_hybrid, inertia_hybrid, center_diff, labels_hybrid, centers_hybrid, mem_MB_hybrid = run_adaptive_hybrid(
                    X, initial_centers, n_clusters, max_iter, repeat, y_true
                )

                results.append([n_samples, n_clusters, n_features, "AdaptiveHybrid", iter_num, elapsed_hybrid, mem_MB_hybrid,
                                ari_hybrid, silhouette_hybrid, dbi_hybrid, inertia_hybrid, center_diff])

# Data frame and output
columns = ['DatasetSize', 'NumClusters', 'NumFeatures', 'Mode', 'SwitchIter',
           'Time', 'Memory_MB', 'ARI', 'Silhouette', 'DBI', 'Inertia', 'CenterDiff']

results_df = pd.DataFrame(results, columns=columns)

# Print summary
print("\n==== SUMMARY ====")
print(results_df.groupby(['DatasetSize','NumClusters','NumFeatures','Mode', 'SwitchIter', 'CenterDiff' ])[['Time','Memory_MB','ARI','Silhouette', 'Inertia']].mean())

# Save results to CSV file
results_df.to_csv("hybrid_kmeans_results.csv", index=False)
print("\nResults saved to 'hybrid_kmeans_results.csv'")
print(os.getcwd())
