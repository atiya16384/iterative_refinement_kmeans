import numpy as np
from aoclda.sklearn import skpatch
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
import time
import pandas as pd
import matplotlib.pyplot as plt
from statistics import mode

def compare_precision():
    dataset_sizes = [10000, 15000, 20000, 25000, 30000] # list of different dataset sizes
    max_iter = 120
    n_clusters = 5
    n_features = 8
    n_repeats = 5
    switch_iters_list = [10, 20, 30, 40, 50, 100] # switch points to test

    # Create dictionary of precisions
    precisions = {
        "Single Precision": np.float32,
        "Double Precision": np.float64
    }
    # Store results
    results = []

    for n_samples in dataset_sizes:
        for repeat in range(n_repeats):
            print(f"Dataset size {n_samples}, Repeat {repeat + 1}")

            # Generate synthetic data
            X , y_true = make_blobs(n_samples=n_samples, n_features=n_features,centers= n_clusters, random_state=repeat )

            # Compute initial cluster centers using double precision
            init_kmeans = KMeans(n_clusters=n_clusters, init = "k-means++", n_init = 1, random_state=repeat)
            init_kmeans.fit(X.astype(np.float64))
            initial_centers = init_kmeans.cluster_centers_

            for mode, dtype in precisions.items():
                X_precision = X.astype(dtype) # cast dataset to proper precision
                init_precision = initial_centers.astype(dtype) # cast initial centers to proper precision
                
                start_time = time.time()
                kmeans = KMeans(n_clusters=n_clusters, init=init_precision, n_init=1, max_iter = max_iter, random_state=repeat, tol=1e-15)
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
            kmeans_double_full = KMeans(n_clusters=n_clusters, init=centers_double, n_init=1, max_iter=max_iter, random_state=repeat, tol=1e-15)
            kmeans_double_full.fit(X.astype(np.float64))
            final_centers_double = kmeans_double_full.cluster_centers_

            # loop over different switch points
            for switch_iter in switch_iters_list:
                if switch_iter >= max_iter:
                    continue

                # Begin with single precision
                X_single = X.astype(np.float32)
                init_single = initial_centers.astype(np.float32)

                kmeans_partial = KMeans(n_clusters=n_clusters, init=init_single, n_init=1, max_iter=switch_iter, random_state=repeat,tol=1e-15 )
                kmeans_partial.fit(X_single)
                centers_partial = kmeans_partial.cluster_centers_

                # switch to double precision
                centers_partial_double = centers_partial.astype(np.float64)
                X_double = X.astype(np.float64)
                start_time = time.time()
                kmeans_refine = KMeans(n_clusters = n_clusters, init=centers_partial_double, n_init=1, max_iter = max_iter - switch_iter, random_state=repeat, tol = 1e-15)
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

                label = f"firstSingle_thenDouble@{switch_iter}"
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
        print(results_df.groupby(['DatasetSize','Mode'])[['Time','ARI','Silhouette', 'Inertia','Memory_MB']].mean())

skpatch()       
compare_precision()

