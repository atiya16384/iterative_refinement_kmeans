import time
import numpy as np
from aoclda.sklearn import skpatch
skpatch()
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score

def evaluate_metrics(X, labels, y_true, inertia):
    if y_true is None:
        ari = np.nan                
    else:
        ari = adjusted_rand_score(y_true, labels)

    db_index = davies_bouldin_score(X, labels)
    return ari, db_index, inertia

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
def run_hybrid(X, initial_centers, n_clusters, max_iter_total, tol_single, tol_double, single_iter_cap, y_true, seed=0):
    # single floating point type
    start_time_single = time.time()
    X_single = X.astype(np.float32)
    # Define the initial centers
    initial_centers_32 = initial_centers.astype(np.float32)
    # this
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
    
def run_adaptive_hybrid(X, initial_centers, n_clusters, max_iter, tol_final, y_true=None,
                        switch_tol=1e-5, switch_shift=1e-4, seed=0):

    # Store dataset in both single and double precision forms
    X_float32 = X.astype(np.float32, copy=False)
    X_float64 = X.astype(np.float64, copy=False)

    # We will always track centers in float64 to retain accuracy during updates
    centers = initial_centers.astype(np.float64)

    # For convergence tracking
    prev_inertia = None
    mode = "single"  # Start in single precision
    labels = None
    iters_single = 0
    iters_double = 0
    elapsed_total = 0

    for i in range(max_iter):
        t0 = time.perf_counter()

        # === Distance computation ===
        # Compute distance matrix using current precision mode
        if mode == "single":
            # Compute in float32 for speed
            dists = np.linalg.norm(X_float32[:, None, :] - centers[None, :, :].astype(np.float32), axis=2)
        else:
            # Compute in float64 for final refinement
            dists = np.linalg.norm(X_float64[:, None, :] - centers[None, :, :], axis=2)

        # === Assignment step ===
        new_labels = np.argmin(dists, axis=1)

        # === Update step ===
        new_centers = np.zeros_like(centers)  # Keep in float64
        for k in range(n_clusters):
            members = X_float64[new_labels == k]  # Always use float64 for accuracy
            if len(members) > 0:
                new_centers[k] = np.mean(members, axis=0)

        # === Convergence metrics ===
        # Calculate total inertia (sum of squared errors) in float64
        inertia = np.sum((X_float64 - new_centers[new_labels])**2)

        # Compute how much centroids moved
        center_shift = np.linalg.norm(new_centers - centers)

        # Compute change in inertia from last iteration
        inertia_diff = abs(prev_inertia - inertia) if prev_inertia is not None else np.inf

        # Update trackers
        centers = new_centers
        labels = new_labels
        prev_inertia = inertia
        elapsed_total += time.perf_counter() - t0
        if mode == "single":
            iters_single += 1
        else:
            iters_double += 1

        # === Check for convergence ===
        if center_shift < tol_final or inertia_diff < tol_final:
            break  # We have converged

        # === Check for adaptive switch to double ===
        if mode == "single" and (inertia_diff < switch_tol or center_shift < switch_shift):
            mode = "double"  # Upgrade to full precision

    # === Final Evaluation ===
    # Optionally evaluate clustering quality if true labels are known
    ari, dbi = None, None
    if y_true is not None:
        ari = adjusted_rand_score(y_true, labels)
    if len(np.unique(labels)) > 1:  # DBI requires at least 2 clusters
        dbi = davies_bouldin_score(X, labels)

    # Estimate memory usage: data + centers
    mem_MB = (X_float32.nbytes + centers.nbytes) / 2**20

    return labels, centers, iters_single, iters_double, elapsed_total, mem_MB, ari, dbi, inertia
