import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from aoclda.sklearn import skpatch
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score

def compare_precision():
    X, y_true = make_blobs(n_samples=10000, centers=3, cluster_std=1.0, random_state=40)

    # Create dictionary of precisions
    precisions = {
        "Single Precision": np.float32,
        "Double Precision": np.float64
    }
    # Store results
    results = {}

    # Loop over precisions
    for precision_name, dtype in precisions.items():
        print(f"\nRunning KMeans with {precision_name}:")

        # Convert data to correct precision
        X_precision = X.astype(dtype)

        # Apply KMeans
        kmeans = KMeans(n_clusters=3, random_state=40)
        kmeans.fit(X_precision)

        # Get labels and centers
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_

        # Evaluate clustering accuracy
        ari = adjusted_rand_score(y_true, labels)
        silhouette_avg = silhouette_score(X_precision, labels)
        db_index = davies_bouldin_score(X_precision, labels)

        # Store results for comparison 
        results[precision_name] = {
            'labels': labels,
            'centers': centers,
            'ari': ari,
            'silhoutte_avg': silhouette_avg,
            'db_index': db_index
        } 

        print("Adjusted Rand Index:", ari)
        print("Cluster Centers:\n", centers)
        print("silhoutte_avg", silhouette_avg)
        print("Db_index", db_index)
        
    # Absolute difference between centers 
    center_diff = np.abs(results["Single Precision"]['centers'] -
                        results["Double Precision"]['centers'])
    print("\nAbsolute difference between cluster centers (float32 vs float64):\n", center_diff)

skpatch()       
compare_precision()

