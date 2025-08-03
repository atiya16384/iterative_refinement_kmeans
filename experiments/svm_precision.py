from aoclda.sklearn import skpatch
skpatch()
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import time 
import psutil
import numpy as np

def measure_memory():
    process = psutil.Process()
    return process.memory_info().rss / (1024 ** 2)

def svm_double_precision(tag, X, y, max_iter, tol, cap=0, C=1.0, kernel='rbf', test_size=0.2, seed=0):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
    mem_before = measure_memory()
    start_time = time.time()
    svm = SVC(C=C, kernel=kernel, gamma='scale', tol=tol, max_iter=int(max_iter), random_state=seed)
    svm.fit(X_train, y_train)
    elapsed = time.time() - start_time
    mem_after = measure_memory()
    y_pred = svm.predict(X_test)

    return (
        tag, len(X), 0, tol, cap, 0, svm.n_iter_, 'Double',
        elapsed, mem_after - mem_before, accuracy_score(y_test, y_pred)
    )


def run_general_adaptive_hybrid(X, initial_centers, n_clusters,
                                max_iter=300, initial_precision='single',
                                stability_threshold=0.01, inertia_improvement_threshold=0.01,
                                refine_iterations=3, seed=0):

    np.random.seed(seed)

    # Dual precision copies
    X_single = X.astype(np.float32, copy=False)
    X_double = X.astype(np.float64, copy=False)

    centers = initial_centers.astype(np.float64)
    labels = np.zeros(X.shape[0], dtype=int)
    prev_inertia = np.inf
    precision = initial_precision
    iter_in_stable_state = 0
    total_iters = 0
    precision_switch = False
    start_time = time.perf_counter()

    for iter_num in range(max_iter):

        # Distance computation
        data_used = X_single if precision == 'single' else X_double
        current_centers = centers.astype(np.float32) if precision == 'single' else centers
        dists = np.linalg.norm(data_used[:, None, :] - current_centers, axis=2)
        new_labels = np.argmin(dists, axis=1)

        # Early stopping if cluster labels stabilize
        label_changes = np.sum(labels != new_labels)
        stability_ratio = label_changes / len(X)

        # Update step (always use double for accuracy)
        new_centers = np.array([
            X_double[new_labels == k].mean(axis=0) if np.any(new_labels == k) else centers[k]
            for k in range(n_clusters)
        ])

        inertia = np.sum((X_double - new_centers[new_labels]) ** 2)
        inertia_improvement = (prev_inertia - inertia) / prev_inertia if prev_inertia != np.inf else np.inf

        # Logging convergence signals
        print(f"[{precision.upper()}] Iter {iter_num+1}: "
              f"Inertia={inertia:.4e}, Improvement={inertia_improvement:.4e}, "
              f"Label Stability={stability_ratio:.4e}")

        # Checking stability criteria
        if inertia_improvement < inertia_improvement_threshold or stability_ratio < stability_threshold:
            iter_in_stable_state += 1
        else:
            iter_in_stable_state = 0

        if iter_in_stable_state >= refine_iterations and not precision_switch:
            print(f"Precision increased from {precision.upper()} to DOUBLE for refinement.")
            precision = 'double'
            precision_switch = True
            iter_in_stable_state = 0  # Reset stability counter for refinement phase

        # Check full convergence after refinement
        if precision == 'double' and iter_in_stable_state >= refine_iterations:
            print("Final convergence reached after refinement.")
            total_iters = iter_num + 1
            break

        centers, labels, prev_inertia = new_centers, new_labels, inertia

    elapsed_time = time.perf_counter() - start_time
    mem_MB = (X_single.nbytes + centers.nbytes) / (2**20)

    return labels, centers, precision_switch, total_iters, elapsed_time, mem_MB, inertia
