from statistics import mode
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from aoclda.sklearn import skpatch
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
import time

def compare_precision():
    dataset_sizes = [1000, 3000, 5000]
    max_iter = 300
    n_clusters = 4
    n_features = 2
    n_repeats = 3

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

            # Generate the data
            X , y_true = make_blobs(n_samples=n_samples, n_features=n_features,centers= n_clusters, random_state=repeat )

            # Common initilisation centers

            init_kmeans = KMeans(n_clusters=n_clusters, init = "k-means++", n_init = 1, random_state=repeat)
            init_kmeans.fit(X.astype(np.float64))
            initial_centers = init_kmeans.cluster_centers_


            for precison_name, dtype in precisions.items():
                X_precision = X.astype(dtype)
                init_precision = initial_centers.astype(dtype)
                
                start_time = time.time()
                kmeans = KMeans(n_clusters=n_clusters, init=init_precision, n_init=1, max_iter = max_iter, random_state=repeat)
                kmeans.fit(X_precision)
                elapsed = time.time() - start_time

                labels = kmeans.labels_
                centers = kmeans.cluster_centers_

                ari = adjusted_rand_score(y_true, labels)
                silhoutte_avg = silhouette_score(X_precision, labels)
                db_index = davies_bouldin_score(X_precision, labels)
                inertia = kmeans.inertia_
                mem_MB = X_precision.nbytes/ 1e6

                results.append([n_samples, max_iter, mode, 0, elapsed, mem_MB, ari, silhoutte_avg, db_index, inertia ])


skpatch()       
compare_precision()

