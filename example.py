import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from aoclda.sklearn import skpatch
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score



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

skpatch()       
compare_precision()

