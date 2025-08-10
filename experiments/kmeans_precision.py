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
    
# kmeans_precision.py
import time
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans

def _mem_megabytes(*arrays) -> float:
    return sum(getattr(a, "nbytes", 0) for a in arrays) / (2**20)

def run_expD_adaptive_sklearn(
    X,
    initial_centers,
    n_clusters,
    *,
    max_iter=300,
    chunk_single=20,
    chunk_double=20,
    stability_threshold=0.02,
    improve_threshold=1e-3,
    shift_tol=1e-3,
    seed=0,
    algorithm="lloyd",
):
    """
    Adaptive *sklearn-only* KMeans:
      - Do short bursts in float32 (single) using KMeans with warm-start centers.
      - When progress stalls OR chunk stops early => switch to float64 (double) bursts.
      - Finish in double.

    Returns a dict with labels/centers/iters/time/memory/inertia.
    """
    X32 = X.astype(np.float32, copy=False)
    X64 = X.astype(np.float64, copy=False)
    centers = initial_centers.astype(np.float64, copy=True)

    prev_labels = None
    prev_inertia = np.inf
    it_single = 0
    it_double = 0
    total = 0
    switched = False

    t0 = time.perf_counter()

    while total < max_iter:
        remaining = max_iter - total

        # -------- SINGLE PRECISION BURST --------
        if not switched:
            step = min(chunk_single, remaining)
            if step <= 0:
                switched = True
                continue

            km = KMeans(
                n_clusters=n_clusters,
                init=centers.astype(np.float32),
                n_init=1,
                max_iter=step,
                tol=0.0,                    # don't let tol stop us early (use chunk)
                algorithm=algorithm,
                random_state=seed,
            )
            labels = km.fit_predict(X32)
            new_centers = km.cluster_centers_.astype(np.float64)
            inertia = float(km.inertia_)
            n_done = int(km.n_iter_)

            # signals
            shift = np.linalg.norm(new_centers - centers)
            improve = (prev_inertia - inertia) / prev_inertia if np.isfinite(prev_inertia) else np.inf
            stability = (np.mean(labels != prev_labels) if prev_labels is not None else 1.0)

            # update
            centers = new_centers
            prev_labels = labels
            prev_inertia = inertia
            it_single += n_done
            total += n_done

            # switch (if chunk ended early, or weak progress, or small shift, or labels stable)
            if (n_done < step) or (improve < improve_threshold) or (shift < shift_tol) or (stability < stability_threshold):
                switched = True
                continue

            if n_done == 0:
                break

        # -------- DOUBLE PRECISION BURST --------
        else:
            step = min(chunk_double, remaining)
            if step <= 0:
                break

            km = KMeans(
                n_clusters=n_clusters,
                init=centers,
                n_init=1,
                max_iter=step,
                tol=0.0,
                algorithm=algorithm,
                random_state=seed,
            )
            labels = km.fit_predict(X64)
            new_centers = km.cluster_centers_.astype(np.float64)
            inertia = float(km.inertia_)
            n_done = int(km.n_iter_)

            shift = np.linalg.norm(new_centers - centers)
            improve = (prev_inertia - inertia) / prev_inertia if np.isfinite(prev_inertia) else np.inf

            centers = new_centers
            prev_inertia = inertia
            it_double += n_done
            total += n_done

            # stop if chunk ended early or no real movement
            if (n_done < step) or (improve < 1e-7) or (shift < 1e-7) or (n_done == 0):
                break

    return {
        "labels": labels,
        "centers": centers,
        "iters_single": it_single,
        "iters_double": it_double,
        "switched": switched,
        "total_iters": total,
        "elapsed_time": time.perf_counter() - t0,
        "mem_MB": _mem_megabytes(X32, centers),
        "inertia": float(prev_inertia),
    }


def run_expE_minibatch_then_full(
    X,
    initial_centers,
    n_clusters,
    *,
    mb_iter=100,
    mb_batch=2048,
    max_refine_iter=100,
    seed=0,
    algorithm="lloyd",
):
    """
    MiniBatchKMeans (float32) -> warm-start full KMeans (float64).
    Entirely sklearn.
    """
    X32 = X.astype(np.float32, copy=False)
    X64 = X.astype(np.float64, copy=False)
    init32 = initial_centers.astype(np.float32, copy=False)

    t0 = time.perf_counter()
    mb = MiniBatchKMeans(
        n_clusters=n_clusters,
        init=init32,
        n_init=1,
        max_iter=mb_iter,
        batch_size=mb_batch,
        random_state=seed,
    )
    mb.fit(X32)
    warm_centers = mb.cluster_centers_.astype(np.float64)

    km = KMeans(
        n_clusters=n_clusters,
        init=warm_centers,
        n_init=1,
        max_iter=max_refine_iter,
        tol=0.0,
        algorithm=algorithm,
        random_state=seed,
    )
    labels = km.fit_predict(X64)

    elapsed = time.perf_counter() - t0

    return {
        "labels": labels,
        "centers": km.cluster_centers_.astype(np.float64),
        "iters_single": int(mb.n_iter_),
        "iters_double": int(km.n_iter_),
        "switched": True,
        "total_iters": int(mb.n_iter_ + km.n_iter_),
        "elapsed_time": elapsed,
        "mem_MB": _mem_megabytes(X32, km.cluster_centers_),
        "inertia": float(km.inertia_),
    }


def mixed_precision_per_cluster(
    X,
    initial_centers,
    *,
    max_iter_total=300,
    tol_single=1e-4,
    tol_double=1e-4,
    single_iter_cap=None,
    seed=0,
    freeze_stable=False,
    freeze_patience=1,
    evaluate_metrics_fn=None,   # defaults to identity if None
):
    """
    Phase 1 (float32): per-cluster Lloyd with per-cluster shift; optional freeze of stable centers.
    Phase 2 (float64): sklearn.KMeans refinement initialized from Phase 1 centers.
    Returns a dict compatible with your row builder for Experiment G.
    """
    t0 = time.time()

    # ---------- Phase 1: single-precision, per-cluster loop ----------
    X32 = X.astype(np.float32, copy=False)
    centers32 = np.asarray(initial_centers, dtype=np.float32, order="C")
    k = centers32.shape[0]

    max_iter_single = max(
        1,
        min(single_iter_cap if single_iter_cap is not None else max_iter_total, max_iter_total)
    )

    shift_history = []
    frozen_history = []
    below_tol_streak = np.zeros(k, dtype=np.int32)
    frozen_mask = np.zeros(k, dtype=bool)

    iters_single = 0
    for it in range(max_iter_single):
        prev_centers = centers32.copy()

        # Assign
        d2 = ((X32[:, None, :] - centers32[None, :, :]) ** 2).sum(axis=2, dtype=np.float32)
        labels = np.argmin(d2, axis=1)

        # Update (respect per-cluster freezing)
        new_centers = prev_centers.copy()
        for j in range(k):
            if freeze_stable and frozen_mask[j]:
                continue
            mask = (labels == j)
            if mask.any():
                new_centers[j] = X32[mask].mean(axis=0, dtype=np.float32)

        centers32 = new_centers

        # Track per-cluster shifts
        per_cluster_shift = np.linalg.norm(centers32 - prev_centers, axis=1).astype(np.float32)
        shift_history.append(per_cluster_shift)

        # Freezing logic
        if freeze_stable:
            below_tol_streak = np.where(per_cluster_shift < tol_single, below_tol_streak + 1, 0)
            newly_frozen = (below_tol_streak >= freeze_patience) & (~frozen_mask)
            frozen_mask |= newly_frozen

        frozen_history.append(frozen_mask.copy())
        iters_single = it + 1

        # Global stop for Phase 1
        if per_cluster_shift.max() < tol_single:
            break

    time_single = time.time() - t0
    last_shift_per_cluster = shift_history[-1] if shift_history else np.zeros(k, dtype=np.float32)

    # ---------- Phase 2: double-precision sklearn refinement ----------
    remaining_iter = max(1, max_iter_total - iters_single)
    t1 = time.time()
    km = KMeans(
        n_clusters=k,
        init=centers32.astype(np.float64, copy=False),
        n_init=1,
        max_iter=remaining_iter,
        tol=tol_double,
        random_state=seed,
        algorithm="lloyd",
    )
    km.fit(X)  # sklearn prefers float64 internally
    time_double = time.time() - t1

    labels_final = km.labels_
    centers_final = km.cluster_centers_
    inertia = float(km.inertia_)
    if evaluate_metrics_fn is not None:
        inertia = evaluate_metrics_fn(inertia)

    # Rough memory (float64 + float32 copies of X)
    mem_MB_total = _mem_mb(X.astype(np.float64, copy=False), X32)

    return {
        "labels": labels_final,
        "centers": centers_final,
        "iters_single": iters_single,
        "iters_double": int(km.n_iter_),
        "elapsed_time": time_single + time_double,
        "mem_MB": mem_MB_total,
        "inertia": inertia,
        "last_shift_per_cluster": last_shift_per_cluster,
        "frozen_mask": frozen_mask,
    }
