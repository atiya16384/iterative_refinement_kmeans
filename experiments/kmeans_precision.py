import time
import numpy as np
from sklearn.cluster import KMeans
import time
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import pairwise_distances_argmin_min

def evaluate_metrics(inertia):
    return inertia


def run_full_single(X, initial_centers, n_clusters, max_iter, tol, y_true,
                    algorithm='lloyd', random_state=0):
    
    #Full single-precision run (treat as the 'single' baseline).
    #Returns the same tuple layout as run_full_double:
    #  centers, labels, iters_single_tot, iters_double_tot, elapsed, mem_MB, inertia

    # cast to float32 for memory/speed (sklearn may upcast internally on some versions)
    X32 = np.asarray(X, dtype=np.float32)
    init32 = np.asarray(initial_centers, dtype=np.float32)

    start_time = time.time()
    kmeans = KMeans(
        n_clusters=n_clusters,
        init=init32,
        n_init=1,
        max_iter=max_iter,
        tol=tol,
        algorithm=algorithm,
        random_state=random_state,
    )
    kmeans.fit(X32)
    elapsed = time.time() - start_time

    iters_single_tot = int(kmeans.n_iter_)
    iters_double_tot = 0  # full single run: no double-precision phase

    labels = kmeans.labels_
    # return centers in float64 for consistency with other paths
    centers = kmeans.cluster_centers_.astype(np.float64, copy=False)
    inertia = evaluate_metrics(kmeans.inertia_)

    mem_MB_single = X32.nbytes / 1e6

    print(f"This is total for single run: {iters_single_tot}")

    return centers, labels, iters_single_tot, iters_double_tot, elapsed, mem_MB_single, inertia

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
def run_hybrid(
    X,
    initial_centers,
    n_clusters,
    max_iter_total,
    tol_single,
    tol_double,
    single_iter_cap,
    y_true,                      # unused, kept for signature parity
    seed=0,
    algorithm="lloyd",
):
    """
    Float32 (capped) -> Float64 refinement.
    Optimized to cast once up-front and reuse arrays (X32/X64, init32/init64).
    """

    # Cast ONCE and reuse (avoids repeated .astype allocations)
    X32  = np.asarray(X, dtype=np.float32)
    X64  = np.asarray(X, dtype=np.float64)
    init32 = np.asarray(initial_centers, dtype=np.float32)
    init64 = np.asarray(initial_centers, dtype=np.float64)

    # Cap sanitization
    cap = int(max(0, min(int(single_iter_cap), int(max_iter_total))))

    # ----- Phase 1: float32 (optional if cap > 0) -----
    iters_single = 0
    t_single = 0.0
    centers64 = init64

    if cap > 0:
        t0 = time.perf_counter()
        km_s = KMeans(
            n_clusters=n_clusters,
            init=init32,
            n_init=1,
            max_iter=cap,
            tol=tol_single,
            algorithm=algorithm,
            random_state=seed,
        ).fit(X32)
        t_single = time.perf_counter() - t0
        iters_single = int(km_s.n_iter_)
        centers64 = km_s.cluster_centers_.astype(np.float64, copy=False)

    # ----- Phase 2: float64 refinement -----
    remaining = max(1, int(max_iter_total) - iters_single)
    t1 = time.perf_counter()
    km_d = KMeans(
        n_clusters=n_clusters,
        init=centers64,          # warm-start via init array (legal)
        n_init=1,
        max_iter=remaining,
        tol=tol_double,
        algorithm=algorithm,
        random_state=seed,
    ).fit(X64)
    t_double = time.perf_counter() - t1

    # Outputs
    labels_final   = km_d.labels_
    centers_final  = km_d.cluster_centers_
    iters_double   = int(km_d.n_iter_)
    inertia        = evaluate_metrics(km_d.inertia_)
    total_time     = t_single + t_double
    mem_MB_total   = (X32.nbytes + X64.nbytes) / 1e6

    return (
        labels_final,
        centers_final,
        iters_single,
        iters_double,
        total_time,
        mem_MB_total,
        inertia,
    )
    
# kmeans_precision.py — Experiments D, E, F (simple, sklearn-first, AOCL-compatible)



# -----------------------------------------------------------------------------
# small helper: estimate memory use in MiB for any numpy arrays we keep around
# -----------------------------------------------------------------------------
def _mem_megabytes(*arrays) -> float:
    return sum(getattr(a, "nbytes", 0) for a in arrays) / (2**20)

# Experiment D — Adaptive Hybrid (global switch)
# Idea:
#   1) Run a short "burst" in single precision (float32) using sklearn.KMeans.
#   2) Check simple global signals (center shift, relative inertia improvement,
#      label stability). If progress is weak OR the burst finishes early, switch.
#   3) Finish remaining iterations in double precision (float64) using sklearn.KMeans.
#

from sklearn.cluster import KMeans
import numpy as np, time

def run_expD_adaptive_sklearn(
    X,
    initial_centers,
    n_clusters,
    *,
    max_iter=300,
    chunk_single=20,            # size of each f32 burst
    improve_threshold=1e-3,     # relative inertia improvement threshold
    shift_tol=1e-3,             # L2(center shift) threshold
    stability_threshold=0.02,   # fraction of labels changed threshold
    seed=0,
    algorithm="lloyd",
):
    X32 = X.astype(np.float32, copy=False)
    X64 = X.astype(np.float64, copy=False)
    centers64 = initial_centers.astype(np.float64, copy=True)

    it_single = it_double = total = 0
    prev_inertia = np.inf
    prev_labels = None
    switched = False

    t0 = time.perf_counter()

    # --------- Phase 1: keep taking f32 chunks until “stalled” or we hit the budget ---------
    while total < max_iter and not switched:
        step = int(min(chunk_single, max_iter - total))
        km_s = KMeans(
            n_clusters=n_clusters,
            init=centers64.astype(np.float32, copy=False),
            n_init=1,
            max_iter=step,
            tol=0.0,                 # run the full chunk
            algorithm=algorithm,
            random_state=seed,
        )
        labels_s = km_s.fit_predict(X32)
        it = int(km_s.n_iter_)
        total += it
        it_single += it

        new_centers64 = km_s.cluster_centers_.astype(np.float64, copy=False)
        inertia_s = float(km_s.inertia_)
        shift = float(np.linalg.norm(new_centers64 - centers64))
        improve = 0.0 if not np.isfinite(prev_inertia) else (prev_inertia - inertia_s) / max(prev_inertia, 1e-12)
        stability = 1.0 if prev_labels is None else float(np.mean(labels_s != prev_labels))

        # update state
        centers64 = new_centers64
        prev_inertia = inertia_s
        prev_labels = labels_s

        # decide to switch to f64
        stalled = (it < step) or (improve < improve_threshold) or (shift < shift_tol) or (stability < stability_threshold)
        switched = stalled  # switch when progress is small/labels stable/centers not moving

        # if budget exhausted, we’ll exit the loop anyway

    # --------- Phase 2: finish in f64 for the remaining budget ---------
    remaining = max(0, max_iter - total)
    if remaining > 0:
        km_d = KMeans(
            n_clusters=n_clusters,
            init=centers64,          # continue from the last f32 centers
            n_init=1,
            max_iter=remaining,
            tol=0.0,
            algorithm=algorithm,
            random_state=seed,
        )
        labels_d = km_d.fit_predict(X64)
        it = int(km_d.n_iter_)
        it_double += it
        total += it
        centers64 = km_d.cluster_centers_.astype(np.float64, copy=False)
        inertia = float(km_d.inertia_)
    else:
        # never entered double; report last f32 state
        labels_d = prev_labels
        inertia = float(prev_inertia)

    return {
        "labels": labels_d,
        "centers": centers64,
        "iters_single": it_single,
        "iters_double": it_double,
        "switched": switched,
        "total_iters": total,
        "elapsed_time": time.perf_counter() - t0,
        "mem_MB": _mem_megabytes(X32, centers64.astype(np.float32, copy=False)),
        "inertia": inertia,
        "shift_after_single": shift if 'shift' in locals() else np.nan,
        "stability_after_single": stability if 'stability' in locals() else np.nan,
    }


# Experiment E — Mini-batch Hybrid K-Means
# Idea:
#   1) Stage 1: MiniBatchKMeans on float32 to move quickly with small memory.
#   2) Stage 2: Full KMeans on float64 to refine to high precision.
#
# Why this is AOCL-friendly:
#   - Uses sklearn.MiniBatchKMeans + sklearn.KMeans; AOCL patches both paths.
# =============================================================================
def run_expE_minibatch_then_full(
    X,
    initial_centers,
    n_clusters,
    *,
    mb_iter=100,            # number of MiniBatchKMeans iterations
    mb_batch=2048,          # mini-batch size (≈ 10–20% of data if feasible)
    max_refine_iter=100,    # final full-batch refinement iters
    seed=0,
    algorithm="elkan",
):
    # prepare views
    X32 = X.astype(np.float32, copy=False)
    X64 = X.astype(np.float64, copy=False)
    init32 = initial_centers.astype(np.float32, copy=False)

    t0 = time.perf_counter()

    # ---- Stage 1: Mini-batch on float32 ----
    mb = MiniBatchKMeans(
        n_clusters=n_clusters,
        init=init32,
        n_init=1,
        max_iter=mb_iter,
        batch_size=mb_batch,
        random_state=seed,
    ).fit(X32)

    # warm-start centers in float64 for precise refinement
    warm64 = mb.cluster_centers_.astype(np.float64, copy=False)

    # ---- Stage 2: Full KMeans on float64 ----
    km = KMeans(
        n_clusters=n_clusters,
        init=warm64,
        n_init=1,
        max_iter=max_refine_iter,
        tol=0.0,
        algorithm=algorithm,
        random_state=seed,
    ).fit(X64)

    elapsed = time.perf_counter() - t0

    return {
        "labels": km.labels_,
        "centers": km.cluster_centers_.astype(np.float64, copy=False),
        "iters_single": int(mb.n_iter_),          # we treat MiniBatchKMeans iters as "single"
        "iters_double": int(km.n_iter_),          # full KMeans iters as "double"
        "switched": True,
        "total_iters": int(mb.n_iter_ + km.n_iter_),
        "elapsed_time": elapsed,
        "mem_MB": _mem_megabytes(X32, km.cluster_centers_),
        "inertia": float(km.inertia_),
    }


def run_expF_percluster_mixed(
    X,
    initial_centers,
    *,
    max_iter_total=300,
    single_iter_cap=3,         # tiny warm-start
    tol_single=0.05,           # freeze quickly
    tol_double=1e-3,           # sklearn tol
    freeze_stable=True,
    freeze_patience=1,
    seed=0,
    algorithm="elkan",
    # NEW knobs that make it fast:
    phase1_sample=50000,       # run Phase-1 on a small subsample
    phase1_time_cap=0.8,       # seconds; hard wall-time for Phase-1
    refine_iter_cap=5,         # max sklearn iterations after warm-start
):
    import time
    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.metrics import pairwise_distances_argmin_min

    rs = np.random.RandomState(seed)
    n = X.shape[0]

    # ----- data views -----
    X32 = X.astype(np.float32, copy=False)
    X64 = X.astype(np.float64, copy=False)

    # ----- choose subsample for the Phase-1 warm start -----
    if phase1_sample and phase1_sample < n:
        idx = rs.choice(n, phase1_sample, replace=False)
        X32_p1 = X32[idx]
    else:
        X32_p1 = X32

    # ----- init -----
    centers32 = np.asarray(initial_centers, dtype=np.float32, order="C")
    k = centers32.shape[0]

    iters1_max = max(1, min(single_iter_cap, max_iter_total))
    frozen = np.zeros(k, dtype=bool)
    below_tol_streak = np.zeros(k, dtype=np.int32)
    iters_single = 0

    t0 = time.perf_counter()

    # ----- Phase 1: FAST warm-start on subsample with hard wall-time -----
    for it in range(iters1_max):
        prev = centers32.copy()

        labels, _ = pairwise_distances_argmin_min(
            X32_p1, centers32, metric='euclidean', metric_kwargs={'squared': True}
        )

        # vectorized centroid update
        counts = np.bincount(labels, minlength=k).astype(np.int32)
        sums = np.zeros_like(centers32, dtype=np.float32)
        np.add.at(sums, labels, X32_p1)
        new_centers = np.divide(
            sums, counts[:, None],
            out=np.copy(centers32), where=counts[:, None] != 0
        )

        # keep frozen clusters unchanged
        if freeze_stable and frozen.any():
            new_centers[frozen] = prev[frozen]

        centers32 = new_centers

        move = np.linalg.norm(centers32 - prev, axis=1).astype(np.float32)

        if freeze_stable:
            below_tol_streak = np.where(move < tol_single, below_tol_streak + 1, 0)
            frozen |= (below_tol_streak >= freeze_patience) & (~frozen)

        iters_single = it + 1
        if move.max() < tol_single:
            break
        if (time.perf_counter() - t0) >= phase1_time_cap:  # hard wall-time
            break

    # ----- Phase 2: clamp sklearn refine iterations -----
    remaining = max(1, min(max_iter_total - iters_single, refine_iter_cap))

    km = KMeans(
        n_clusters=k,
        init=centers32.astype(np.float64, copy=False),
        n_init=1,
        max_iter=remaining,
        tol=tol_double,
        random_state=seed,
        algorithm=algorithm,
    ).fit(X64)

    elapsed = time.perf_counter() - t0

    # rough memory: float32 view + final centers
    mem_MB = (X32.nbytes + km.cluster_centers_.nbytes) / (2**20)

    return {
        "labels": km.labels_,
        "centers": km.cluster_centers_.astype(np.float64, copy=False),
        "iters_single": iters_single,
        "iters_double": int(km.n_iter_),
        "elapsed_time": elapsed,
        "mem_MB": mem_MB,
        "inertia": float(km.inertia_),
        "frozen_mask": frozen,
    }
