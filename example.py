import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score, silhouette_score, davies_bouldin_score
from sklearn.cluster import KMeans
from aoclda.sklearn import skpatch
import warnings
warnings.filterwarnings("ignore")

# Apply AOCL patch before any KMeans usage
skpatch()
# CONFIGURATION PARAMETERS

dataset_sizes = [10000, 20000, 50000, 100000, 200000, 500000]
n_clusters_list = [5, 8]
n_features_list = [2, 4]  # We keep 2 here for proper plotting
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
def run_adaptive_hybrid(X, initial_centers, n_clusters, max_iter, repeat, y_true):
    # Define tolerance to decide when to switch to double precision
    tol_single_precision = 1e-7
    # Convert data to single precision for fast computation
    X_single = X.astype(np.float32)
    # convert dataset and initial centroids to single precision
    centers = initial_centers.astype(np.float32) 

   # kmeans_single= KMeans(n_clusters=n_clusters, init=centers, n_init=1,
                       #     max_iter=max_iter - iter_num, random_state=repeat)
    #kmeans_single.fit(X_single)

    for iter_num in range(max_iter):
        # Assignment step (distance calculation in single precision)
        # calcuate the distance of each point to centroid using euclidean distance and assign to nearest cluster.
        distances = np.linalg.norm(X_single[:, None, :] - centers[None, :, :], axis=2)
        labels = np.argmin(distances, axis=1)

        # Update step (centroid update in double precision)
        # for each cluster, compute the mean of all assigned 
        new_centers = []
        for k in range(n_clusters):
            cluster_points = X_single[labels == k]
            if len(cluster_points) == 0:
                # Reinitialize empty cluster to a random data point
                random_idx = np.random.choice(len(X_single))
                new_centers.append(X_single[random_idx].astype(np.float64))
            else:
                new_centers.append(cluster_points.astype(np.float64).mean(axis=0))

        new_centers = np.array(new_centers, dtype=np.float64).astype(np.float32)
        # Calculate how much centroids moved (shift)
        shift = np.linalg.norm(new_centers - centers)/ new_centers
        centers = new_centers

        # # If the shift is below the tolerance, we assume we're close enough to final convergence.
        if shift < tol_single_precision:
            print(f"Switching to double after {iter_num+1} iterations. Shift={shift}")
            break

    # Switch to double precision refinement
    # Convert both the data and the centroids to float64
    centers_double = centers.astype(np.float64)
    X_double = X.astype(np.float64)

    start_time = time.time()
    print(np.isnan(X_double))
    print(np.isnan(centers_double))
    
    # Run standard sckit-learn k-means starting from the current centroids.
    # This helps refine the solution in full double precision for high accuracy
    kmeans_refine = KMeans(n_clusters=n_clusters, init=centers_double, n_init=1,
                            max_iter=max_iter - iter_num, random_state=repeat)
    kmeans_refine.fit(X_double)
    elapsed = time.time() - start_time
   # speedup = elapsed / elapsed_hybrid

    # Extract final labels, final cluster centers, final inertia.
    labels_final = kmeans_refine.labels_
    centers_final = kmeans_refine.cluster_centers_
    inertia = kmeans_refine.inertia_

    # Evaluate with these metrics
    ari, silhouette, dbi, inertia = evaluate_metrics(X, labels_final, y_true, inertia)
    mem_MB = X_double.nbytes / 1e6
    
    # Memory used: full double + partial single precision copies
    mem_MB_hybrid = (X.astype(np.float64).nbytes + X.astype(np.float32).nbytes) / 1e6

    center_diff = np.linalg.norm(centers_final - centers_double)

    return iter_num, elapsed, mem_MB, ari, silhouette, dbi, inertia, center_diff, labels_final, centers_final, mem_MB_hybrid,

# Main loop
results = []

for n_samples in dataset_sizes:
    for n_clusters in n_clusters_list:
        for n_features in n_features_list:
            for repeat in range(n_repeats):
                print(f"Samples={n_samples}, Clusters={n_clusters}, Features={n_features}, Repeat={repeat+1}")

                # Generate data
                X, y_true = generate_data(n_samples, n_features, n_clusters, repeat)

                # Visualize raw data (only if n_features=2)
                if n_features == 2:
                    plt.scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis', s=10)
                    plt.title(f"Generated Data - Samples {n_samples}")
                    plt.show()
                    plt.savefig()

                # Compute initial centers once
                init_kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=1, random_state=repeat)
                init_kmeans.fit(X.astype(precisions["Double Precision"]))
                initial_centers = init_kmeans.cluster_centers_

                # Full double precision run
                centers_double, labels_double, inertia, elapsed, mem_MB_double, ari, silhouette, dbi = run_full_double(
                    X, initial_centers, n_clusters, max_iter, repeat, y_true
                )

                results.append([n_samples, n_clusters, n_features, "Double", 0, elapsed, mem_MB_double,
                                ari, silhouette, dbi, inertia, 0])

                if n_features == 2:
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
print(results_df.groupby(['DatasetSize','NumClusters','NumFeatures','Mode'])[['Time','ARI','Silhouette']].mean())

# Save results to CSV file
results_df.to_csv("hybrid_kmeans_results.csv", index=False)
print("\nResults saved to 'hybrid_kmeans_results.csv'")
