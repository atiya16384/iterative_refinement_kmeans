from statistics import mode
import numpy as np
from aoclda.sklearn import skpatch
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
import time
import pandas as pd
import matplotlib.pyplot as plt

def compare_precision():
    dataset_sizes = [5000, 8000, 10000] # list of different dataset sizes
    max_iter = 300
    n_clusters = 4
    n_features = 2
    n_repeats = 3
    switch_iters_list = [10, 20, 30, 50, 100, 150, 200] # switch points to test

    # Create dictionary of precisions
    precisions = {
        "Single Precision": np.float32,
        "Double Precision": np.float64
    }
    # Store results
    results = []

    for n_samples in dataset_sizes:
        for repeat in range(n_repeats):
            print(f"Dataset size{n_samples}, Repeat {repeat + 1}")

            # Generate synthetic data
            X , y_true = make_blobs(n_samples=n_samples, n_features=n_features,centers= n_clusters, random_state=repeat )

            # Compute initial cluster centers using double precision
            # Ensures all 

            init_kmeans = KMeans(n_clusters=n_clusters, init = "k-means++", n_init = 1, random_state=repeat)
            init_kmeans.fit(X.astype(np.float64))
            initial_centers = init_kmeans.cluster_centers_

            for precison_name, dtype in precisions.items():
                X_precision = X.astype(dtype) # cast dataset to proper precision
                init_precision = initial_centers.astype(dtype) # cast initial centers to proper precision
                
                start_time = time.time()
                kmeans = KMeans(n_clusters=n_clusters, init=init_precision, n_init=1, max_iter = max_iter, random_state=repeat)
                kmeans.fit(X_precision)
                elapsed = time.time() - start_time

                labels = kmeans.labels_
                centers = kmeans.cluster_centers_
                
                # Compute clustering quality metrics
                ari = adjusted_rand_score(y_true, labels)
                silhoutte_avg = silhouette_score(X_precision, labels)
                db_index = davies_bouldin_score(X_precision, labels)
                inertia = kmeans.inertia_
                mem_MB = X_precision.nbytes/ 1e6

                results.append([n_samples, max_iter, mode, 0, elapsed, mem_MB, ari, silhoutte_avg, db_index, inertia ])

            # run full double precision again to store the final centers for center different calculation
            centers_double = initial_centers.astype(np.float64)
            kmeans_double_full = KMeans(n_clusters=n_clusters, init=centers_double, n_init=1, max_iter=max_iter, random_state=repeat)
            kmeans_double_full.fit(X.astype(np.float64))
            final_centers_double = kmeans_double_full.clusters_centers_

            # loop over different switch points
            for switch_iter in switch_iters_list:
                if switch_iter >= max_iter:
                    continue

                # Begin with single precision
                X_single = X.astype(np.float32)
                init_single = initial_centers.astype(np.float32)

                kmeans_partial = KMeans(n_clusters=n_clusters, init=init_single, n_init=1, max_iter=switch_iter, random_state=repeat)
                kmeans_partial.fit(X_single)
                centers_partial = kmeans_partial.clusters_centers_

                # switch to double precision
                centers_partial_double = centers_partial.astype(np.float64)
                X_double = X.astype(np.float64)
                start_time = time.time()
                kmeans_refine = KMeans(n_clusters = n_clusters, init=centers_partial_double, n_init=1, max_iter = max_iter - switch_iter, random_state=repeat)
                kmeans_refine.fit(X_double)
                elapsed_hybrid = time.time() - start_time

                labels_hybrid = kmeans_refine.labels_
                centers_hybrid = kmeans_refine.cluster_centers_

                ari_hybrid = adjusted_rand_score(y_true, labels_hybrid)
                silhouette_avg_hybrid = silhouette_score(X_double, labels_hybrid)
                db_index_hybrid = davies_bouldin_score(X_precision, labels)
                inertia_hybrid = kmeans_refine.inertia_
                mem_MB_hybrid = X_double.nbytes/ 1e6

                # calculate center difference
                center_diff = np.linalg.norm(centers_hybrid - final_centers_double)

                label = f"Hybrid_Switch@{switch_iter}"
                results.append([
                    n_samples, max_iter, label, switch_iter, elapsed_hybrid, mem_MB_hybrid,
                    ari_hybrid, silhouette_avg_hybrid, db_index_hybrid, inertia_hybrid, center_diff
                ])

        columns = [
        'DatasetSize', 'MaxIter', 'Mode', 'SwitchIter', 'Time', 'Memory_MB',
        'ARI', 'Silhouette', 'DBI', 'Inertia', 'CenterDiff'
        ]

        # Extract centers separately since they cannot be put in dataframe directly
        flattened_results = []
        for row in results:
            if len(row) == 11:  # Hybrid row with center diff
                flattened_results.append(row)
            else:  # Single or Double row (no center diff yet)
                center_diff_placeholder = np.nan
                flattened_results.append(row + [center_diff_placeholder])

        results_df = pd.DataFrame(flattened_results, columns=columns)
        print("\n==== Final Summary Table ====")
        print(results_df.groupby(['DatasetSize','Mode'])[['Time','ARI','Silhouette']].mean())

        # PLOTS

        # Timing vs Dataset Size
        plt.figure(figsize=(10, 5))
        for mode in results_df['Mode'].unique():
            mean_times = results_df[results_df['Mode'] == mode].groupby('DatasetSize')['Time'].mean()
            plt.plot(mean_times.index, mean_times.values, marker='o', label=mode)
        plt.xlabel("Dataset Size")
        plt.ylabel("Time (s)")
        plt.title("Timing vs Dataset Size")
        plt.legend()
        plt.grid()
        plt.show()

        # ARI vs Dataset Size
        plt.figure(figsize=(10, 5))
        for mode in results_df['Mode'].unique():
            mean_ari = results_df[results_df['Mode'] == mode].groupby('DatasetSize')['ARI'].mean()
            plt.plot(mean_ari.index, mean_ari.values, marker='o', label=mode)
        plt.xlabel("Dataset Size")
        plt.ylabel("ARI")
        plt.title("ARI vs Dataset Size")
        plt.legend()
        plt.grid()
        plt.show()

        # Hybrid Switch Analysis (ARI vs Switch Iter)
        hybrids = results_df[results_df['Mode'].str.contains('Hybrid')]
        plt.figure(figsize=(10, 5))
        for ds in dataset_sizes:
            df_ds = hybrids[hybrids['DatasetSize'] == ds].groupby('SwitchIter').mean()
            plt.plot(df_ds.index, df_ds['ARI'], marker='o', label=f'Size={ds}')
        plt.xlabel("Switch Iteration")
        plt.ylabel("ARI")
        plt.title("Hybrid: ARI vs Switch Iteration")
        plt.legend()
        plt.grid()
        plt.show()

        # Timing vs Switch Iteration
        plt.figure(figsize=(10, 5))
        for ds in dataset_sizes:
            df_ds = hybrids[hybrids['DatasetSize'] == ds].groupby('SwitchIter').mean()
            plt.plot(df_ds.index, df_ds['Time'], marker='o', label=f'Size={ds}')
        plt.xlabel("Switch Iteration")
        plt.ylabel("Time (s)")
        plt.title("Hybrid: Time vs Switch Iteration")
        plt.legend()
        plt.grid()
        plt.show()

        # Memory Usage (Single vs Double only)
        plt.figure(figsize=(10, 5))
        for mode in ["Single Precision", "Double Precision"]:
            mem = results_df[results_df['Mode'] == mode].groupby('DatasetSize')['Memory_MB'].mean()
            plt.plot(mem.index, mem.values, marker='o', label=mode)
        plt.xlabel("Dataset Size")
        plt.ylabel("Memory Usage (MB)")
        plt.title("Memory Usage vs Dataset Size")
        plt.legend()
        plt.grid()
        plt.show()

        # Cluster visualization (optional sample example)
        X_vis, y_vis = make_blobs(n_samples=500, n_features=2, centers=n_clusters, random_state=42)
        plt.scatter(X_vis[:, 0], X_vis[:, 1], c=y_vis, s=30, cmap='viridis')
        plt.title("Example synthetic KMeans data")
        plt.show()


skpatch()       
compare_precision()

