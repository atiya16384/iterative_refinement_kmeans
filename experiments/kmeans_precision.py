import time
import numpy as np
from sklearn.cluster import KMeans
import time
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans

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


# =============================================================================
# Experiment D — Adaptive Hybrid (global switch)
# Idea:
#   1) Run a short "burst" in single precision (float32) using sklearn.KMeans.
#   2) Check simple global signals (center shift, relative inertia improvement,
#      label stability). If progress is weak OR the burst finishes early, switch.
#   3) Finish remaining iterations in double precision (float64) using sklearn.KMeans.
#
# Why this is AOCL-friendly:
#   - We only use sklearn.KMeans; AOCL patches it when skpatch() has been applied.
# =============================================================================
def run_expD_adaptive_sklearn(
    X,
    initial_centers,
    n_clusters,
    *,
    max_iter=300,           # total iteration budget (single + double)
    chunk_single=20,        # length of the initial single-precision burst
    improve_threshold=1e-3, # min relative inertia improvement to consider "good progress"
    shift_tol=1e-3,         # min L2 center shift to consider "movement"
    stability_threshold=0.02,# max label-change rate to consider "stable" (lower = more stable)
    seed=0,
    algorithm="lloyd",
):
    # ------------------------------
    # prepare data and state
    # ------------------------------
    X32 = X.astype(np.float32, copy=False)  # single-precision view
    X64 = X.astype(np.float64, copy=False)  # double-precision view
    centers64 = initial_centers.astype(np.float64, copy=True)

    # book-keeping
    prev_inertia = np.inf
    prev_labels = None
    it_single = 0
    it_double = 0
    total = 0
    switched = False

    t0 = time.perf_counter()

    # ------------------------------
    # Phase 1: single-precision burst
    # ------------------------------
    # cap the burst to remaining budget
    step1 = max(1, min(chunk_single, max_iter))
    km_s = KMeans(
        n_clusters=n_clusters,
        init=centers64.astype(np.float32, copy=False),
        n_init=1,
        max_iter=step1,
        tol=0.0,                # we control stopping via chunk size, not tol
        algorithm=algorithm,
        random_state=seed,
    )
    labels_s = km_s.fit_predict(X32)
    it1 = int(km_s.n_iter_)
    it_single += it1
    total += it1

    # measure signals
    new_centers64 = km_s.cluster_centers_.astype(np.float64, copy=False)
    inertia_s = float(km_s.inertia_)
    shift = float(np.linalg.norm(new_centers64 - centers64))
    # relative improvement vs previous run; here prev is inf → treat as ∞
    improve = np.inf if not np.isfinite(prev_inertia) else (prev_inertia - inertia_s) / prev_inertia
    # label stability = fraction of points whose label changed vs prior labels
    stability = 1.0 if prev_labels is None else float(np.mean(labels_s != prev_labels))

    # update state
    centers64 = new_centers64
    prev_inertia = inertia_s
    prev_labels = labels_s

    # decide if we should switch now (POC: always switch after this single burst,
    # but we also keep the logic here to be explicit about *why*)
    if (it1 < step1) or (improve < improve_threshold) or (shift < shift_tol) or (stability < stability_threshold):
        switched = True
    else:
        switched = True  # even if progress is good, we *want* to finish in double

    # ------------------------------
    # Phase 2: finish in double precision
    # ------------------------------
    remaining = max(1, max_iter - total)
    km_d = KMeans(
        n_clusters=n_clusters,
        init=centers64,
        n_init=1,
        max_iter=remaining,
        tol=0.0,
        algorithm=algorithm,
        random_state=seed,
    )
    labels_d = km_d.fit_predict(X64)
    it2 = int(km_d.n_iter_)
    it_double += it2
    total += it2

    # ------------------------------
    # build result
    # ------------------------------
    return {
        "labels": labels_d,
        "centers": km_d.cluster_centers_.astype(np.float64, copy=False),
        "iters_single": it_single,
        "iters_double": it_double,
        "switched": switched,
        "total_iters": total,
        "elapsed_time": time.perf_counter() - t0,
        "mem_MB": _mem_megabytes(X32, km_d.cluster_centers_),
        "inertia": float(km_d.inertia_),
        # optional diagnostics
        "shift_after_single": shift,
        "stability_after_single": stability,
    }


# =============================================================================
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


# =============================================================================
# Experiment F — Mixed Precision Per-Cluster
# Idea:
#   1) Phase 1: A light, numpy-based Lloyd loop in float32 that lets you optionally
#      "freeze" individual clusters that appear stable (small centroid movement).
#   2) Phase 2: Hand off to sklearn.KMeans in float64 for final refinement.
#
# Why this is mostly AOCL-friendly:
#   - Per-cluster freezing is not exposed in sklearn APIs, so Phase 1 is a tiny
#     numpy loop. Phase 2 uses sklearn.KMeans, so AOCL still accelerates the
#     heavy refinement stage.
# =============================================================================
def run_expF_percluster_mixed(
    X,
    initial_centers,
    *,
    max_iter_total=300,     # total budget = (phase1 iters) + (phase2 iters)
    single_iter_cap=100,    # cap Phase 1 iterations
    tol_single=1e-3,        # per-cluster movement tolerance to consider "stable"
    tol_double=1e-4,        # sklearn refinement tolerance
    freeze_stable=True,     # if True, stop updating clusters that are stable
    freeze_patience=1,      # require N consecutive "below tol" moves to freeze
    seed=0,
    algorithm="elkan",
):
    t0 = time.perf_counter()

    # ----- Phase 1: per-cluster float32 Lloyd with optional freezing -----
    X32 = X.astype(np.float32, copy=False)
    centers32 = np.asarray(initial_centers, dtype=np.float32, order="C")
    k = centers32.shape[0]

    # number of iterations we will do in Phase 1
    iters1_max = max(1, min(single_iter_cap, max_iter_total))

    # keep some simple state for freezing
    frozen = np.zeros(k, dtype=bool)
    below_tol_streak = np.zeros(k, dtype=np.int32)

    iters_single = 0
    for it in range(iters1_max):
        prev = centers32.copy()

        # assignment step (squared Euclidean distances, all in float32)
        # shape: (n_samples, k)
        d2 = ((X32[:, None, :] - centers32[None, :, :]) ** 2).sum(axis=2, dtype=np.float32)
        labels = np.argmin(d2, axis=1)

        # update step, with optional per-cluster freezing
        for j in range(k):
            if freeze_stable and frozen[j]:
                continue
            mask = (labels == j)
            if mask.any():
                centers32[j] = X32[mask].mean(axis=0, dtype=np.float32)

        # track how much each cluster moved this iteration
        move = np.linalg.norm(centers32 - prev, axis=1).astype(np.float32)

        if freeze_stable:
            # increment streak for clusters that moved less than tol_single
            below_tol_streak = np.where(move < tol_single, below_tol_streak + 1, 0)
            # freeze newly stable clusters (patience reached)
            newly = (below_tol_streak >= freeze_patience) & (~frozen)
            frozen |= newly

        iters_single = it + 1

        # global early stop for Phase 1: if *all* moves are tiny, break
        if move.max() < tol_single:
            break

    # ----- Phase 2: sklearn.KMeans refinement in float64 -----
    remaining = max(1, max_iter_total - iters_single)
    X64 = X.astype(np.float64, copy=False)
    km = KMeans(
        n_clusters=k,
        init=centers32.astype(np.float64, copy=False),  # warm start from Phase 1
        n_init=1,
        max_iter=remaining,
        tol=tol_double,
        random_state=seed,
        algorithm=algorithm,
    ).fit(X64)

    elapsed = time.perf_counter() - t0

    return {
        "labels": km.labels_,
        "centers": km.cluster_centers_.astype(np.float64, copy=False),
        "iters_single": iters_single,
        "iters_double": int(km.n_iter_),
        "elapsed_time": elapsed,
        "mem_MB": _mem_megabytes(X32, km.cluster_centers_),  # rough: float32 + final centers
        "inertia": float(km.inertia_),
        "frozen_mask": frozen,  # which clusters ended Phase 1 frozen
    }

