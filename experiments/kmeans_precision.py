import time
import numpy as np
from aoclda.sklearn import skpatch
skpatch()
from sklearn.cluster import KMeans

def evaluate_metrics(inertia):
    return inertia

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

    inertia = evaluate_metrics(inertia)
    mem_MB_double = X.astype(np.float64).nbytes / 1e6
    return centers, labels, iters_double_tot, iters_single_tot,  elapsed, mem_MB_double, inertia, 

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

    inertia = evaluate_metrics(inertia)
    mem_MB_double = X.astype(np.float64).nbytes / 1e6
    mem_MB_total = mem_MB_double + X_single.nbytes / 1e6
    iters_double = kmeans_double.n_iter_
    print(f"This is iters_double: {iters_double}")
    total_time = end_time_single + end_time_double
    
    return ( labels_final, centers_final, iters_single, iters_double,total_time, mem_MB_total, inertia)
    
def run_adaptive_hybrid(X, initial_centers, n_clusters,
                        max_iter=300,
                        initial_precision='single',
                        stability_threshold=0.02,
                        inertia_improvement_threshold=0.02,
                        refine_iterations=2,
                        tol_shift=1e-3,
                        seed=0,
                        y_true=None):


    np.random.seed(seed)
    X_f32 = X.astype(np.float32, copy=False)
    X_f64 = X.astype(np.float64, copy=False)
    centers = initial_centers.astype(np.float64)
    labels = np.zeros(X.shape[0], dtype=int)

    prev_inertia = np.inf
    precision = initial_precision
    stable_count = 0
    switched = False
    switch_iter = None

    start = time.perf_counter()

    for it in range(max_iter):
        data = X_f32 if precision == 'single' else X_f64
        cent = centers.astype(np.float32) if precision == 'single' else centers

        dists = np.linalg.norm(data[:, None, :] - cent, axis=2)
        new_labels = np.argmin(dists, axis=1)

        stability = np.sum(labels != new_labels) / len(labels)
        data_mean = X_f32 if precision == 'single' else X_f64
        new_centers = np.array([
            data_mean[new_labels == k].mean(axis=0) if np.any(new_labels == k) else centers[k]
            for k in range(n_clusters)
        ])

        inertia = np.sum((X_f64 - new_centers[new_labels])**2)
        inertia_improve = (prev_inertia - inertia) / prev_inertia if prev_inertia != np.inf else np.inf
        shift = np.linalg.norm(new_centers - centers)

        print(f"[{precision.upper()}] Iter {it+1}: inertia={inertia:.4e}, "
              f"Δinertia={inertia_improve:.4f}, shift={shift:.4e}, stability={stability:.4f}")

        # Track convergence
        if (inertia_improve < inertia_improvement_threshold or
            shift < tol_shift or
            stability < stability_threshold):
            stable_count += 1
        else:
            stable_count = 0

        # Switch to double
        if not switched and stable_count >= refine_iterations:
            precision = 'double'
            switched = True
            switch_iter = it + 1
            stable_count = 0
            print("Switched to DOUBLE precision")

        # Final convergence
        if switched and stable_count >= refine_iterations:
            print("Converged in DOUBLE precision")
            break

        labels = new_labels
        centers = new_centers
        prev_inertia = inertia

    total_iters = it + 1
    iters_single = switch_iter if switched else total_iters
    iters_double = total_iters - iters_single if switched else 0
    elapsed_time = time.perf_counter() - start
    mem_MB = (X_f32.nbytes + centers.nbytes) / 2**20

    return {
    'labels': labels,
    'centers': centers,
    'iters_single': iters_single,
    'iters_double': iters_double,
    'switched': switched,
    'total_iters': total_iters,
    'elapsed_time': elapsed_time,
    'mem_MB': mem_MB,
    'inertia': inertia
    }

