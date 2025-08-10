from visualisations.kmeans_visualisations import KMeansVisualizer
from experiments.kmeans_precision import run_full_double, run_hybrid, run_adaptive_hybrid,  run_expD_adaptive_sklearn, run_expE_minibatch_then_full,run_expF_percluster_mixed,      
import numpy as np
import time
import numpy as np
from math import ceil

def run_experiment_A(ds_name, X, y_true, n_clusters, initial_centers, config):
    rows_A = []
    n_samples = len(X)
    n_features = X.shape[1]
    X_cur = X
    y_true_cur = y_true

    max_iter = config["max_iter_A"]
    tol_fixed_A = config["tol_fixed_A"]
    cap_grid = config["cap_grid"]
    n_repeats = config["n_repeats"]

    for rep in range(n_repeats):
        # Full double precision run
        centers_double, labels_double, iters_double_tot, iters_single_tot, elapsed, mem_MB_double, inertia = run_full_double(
            X_cur, initial_centers, n_clusters, max_iter, tol_fixed_A, y_true_cur
        )

        rows_A.append([
            ds_name, n_samples, n_clusters, "A", 0, 0,
            iters_single_tot, iters_double_tot, "Double", elapsed, mem_MB_double,
             inertia
        ])

    for cap in cap_grid:
        for rep in range(n_repeats):
            # Hybrid run
            labels_hybrid, centers_hybrid, iters_single, iters_double, elapsed_hybrid, mem_MB_hybrid, inertia_hybrid = run_hybrid(
                X_cur, initial_centers, n_clusters,
                max_iter_total=max_iter,
                single_iter_cap=cap,
                tol_single=tol_fixed_A,
                tol_double=tol_fixed_A,
                y_true=y_true_cur,
                seed=rep
            )

            rows_A.append([
                ds_name, n_samples, n_clusters, "A", cap, tol_fixed_A,
                iters_single, iters_double, "Hybrid", elapsed_hybrid, mem_MB_hybrid,
                inertia_hybrid
            ])

            # Only plot for the first repeat
            if rep == 0:
                X_vis, centers_vis, xx, yy, labels_grid = KMeansVisualizer.pca_2d_view(X_cur, centers_hybrid)
                filename = f"{ds_name}_n{n_samples}_c{n_clusters}_A_{cap}"
                title = f"{ds_name}: n={n_samples}, c={n_clusters}, cap={cap}"
                KMeansVisualizer.plot_clusters(X_vis, labels_hybrid, centers_vis, xx, yy, labels_grid, title=title, filename=filename)
                
    return rows_A

def run_experiment_B(ds_name, X, y_true, n_clusters, initial_centers, config):
    rows_B = []
    n_samples = len(X)
    n_features = X.shape[1]
    X_cur = X
    y_true_cur = y_true

    max_iter_B = config["max_iter_B"]
    tol_double_B = config["tol_double_B"]
    tol_single_grid = config["tol_single_grid"]
    n_repeats = config["n_repeats"]

    for rep in range(n_repeats):
        centers_double, labels_double, iters_double_tot, iters_single_tot, elapsed, mem_MB_double, inertia = run_full_double(
        X_cur, initial_centers, n_clusters, max_iter_B, tol_double_B, y_true_cur
        )

        rows_B.append([ ds_name, n_samples, n_clusters, "B", tol_double_B,  iters_single_tot, iters_double_tot, "Double", elapsed, mem_MB_double,
               inertia])
        print(f"[Double Baseline - Exp B] tol={tol_double_B} | iter_double={iters_double_tot}")
        print(f"The total number of features is : F={n_features}")


    for tol_s in tol_single_grid:
        for rep in range(n_repeats):
        #  hybrid run
            labels_hybrid, centers_hybrid, iters_single, iters_double, elapsed_hybrid, mem_MB_hybrid, inertia_hybrid = run_hybrid(
                X_cur, initial_centers, n_clusters, max_iter_total=max_iter_B, tol_single = tol_s, tol_double = tol_double_B, single_iter_cap=max_iter_B, y_true= y_true_cur, seed = rep
            )

            print(f"Tol_single: {tol_s}, Iter Single: {iters_single}, Iter Double: {iters_double}, Total: {iters_single + iters_double}")
            print(f"The total number of features is : F={n_features}")


            rows_B.append([ds_name, n_samples, n_clusters, "B", tol_s,  iters_single, iters_double, "Hybrid", elapsed_hybrid, mem_MB_hybrid,
                     inertia_hybrid])
        
            print(f" [Hybrid] {rows_B}", flush=True) 
            print(f"The total number of features is : F={n_features}")


            # plot clusters
            if rep == 0:
                X_vis, centers_vis, xx, yy, labels_grid = KMeansVisualizer.pca_2d_view(X_cur, centers_hybrid)
                filename = f"{ds_name}_n{n_samples}_c{n_clusters}_B_{tol_s}"
                title = f"{ds_name}: n={n_samples}, c={n_clusters}, tol={tol_s}"
                KMeansVisualizer.plot_clusters(X_vis, labels_hybrid, centers_vis, xx, yy, labels_grid, title=title, filename=filename)
                
    return rows_B




def run_experiment_C(ds_name, X, y_true, n_clusters, initial_centers, config):
    rows_C = []
    n_samples = len(X)

    max_iter_C = config["max_iter_C"]
    tol_fixed_C = config["tol_fixed_C"]
    cap_percentages = config.get("cap_percentages", [0.0, 0.1, 0.2, 0.4, 0.6, 0.8])
    n_repeats = config.get("n_repeats", 1)

    for rep in range(n_repeats):
        # Full double precision baseline
        centers_double, labels_double, iters_double_tot, iters_single_tot, elapsed_d, mem_MB_d, inertia_d = run_full_double(
            X, initial_centers, n_clusters, max_iter_C, tol_fixed_C, y_true
        )

        rows_C.append([
            ds_name, n_samples, n_clusters, "C", "full", tol_fixed_C,
            0, iters_double_tot, "Double", elapsed_d, mem_MB_d, inertia_d
        ])

        for pct in cap_percentages:
            single_cap = ceil(max_iter_C * pct)

            labels_h, centers_h, iters_s, iters_d, elapsed_h, mem_MB_h, inertia_h = run_hybrid(
                X, initial_centers, n_clusters,
                max_iter_total=max_iter_C,
                single_iter_cap=single_cap,
                tol_single=tol_fixed_C,
                tol_double=tol_fixed_C,
                y_true=y_true,
                seed=rep
            )

            rows_C.append([
                ds_name, n_samples, n_clusters, "C", pct, tol_fixed_C,
                iters_s, iters_d, "Hybrid", elapsed_h, mem_MB_h, inertia_h
            ])

            # Optional: 2D PCA cluster plot for first rep
            if rep == 0:
                X_vis, centers_vis, xx, yy, labels_grid = KMeansVisualizer.pca_2d_view(X, centers_h)
                filename = f"{ds_name}_C_cap{int(pct*100)}"
                title = f"{ds_name}: cap = {int(pct*100)}% iter"
                KMeansVisualizer.plot_clusters(X_vis, labels_h, centers_vis, xx, yy, labels_grid, title=title, filename=filename)

    return rows_C

def run_experiment_D(ds_name, X, y_true, n_clusters, initial_centers, config):
    """
    Experiment D — Adaptive Hybrid (global switch)
    Baseline: full double with same max_iter.
    Variant:  short float32 burst -> finish in float64 (sklearn-only).
    """
    rows_D = []
    n = len(X)
    reps = int(config["n_repeats"])

    # Fixed POC parameters (read like A/B/C)
    max_iter = int(config["max_iter_D"])
    tol_double_baseline = float(config["tol_double_baseline_D"])

    # Variant knobs (single values; not sweeping)
    chunk_single = int(config["chunk_single_D"])
    improve_threshold = float(config["improve_threshold_D"])
    shift_tol = float(config["shift_tol_D"])
    stability_threshold = float(config["stability_threshold_D"])

    for rep in range(reps):
        # Baseline: pure double (same budget)
        c_d, l_d, it_d, it_s, t, mem, J = run_full_double(
            X, initial_centers, n_clusters, max_iter, tol_double_baseline, y_true
        )
        rows_D.append([
            ds_name, n, n_clusters, "D",
            chunk_single, improve_threshold,     # store the variant params for reference
            it_s, it_d,
            "Double", t, mem, J
        ])

        # Variant: adaptive single burst -> double finish
        res = run_expD_adaptive_sklearn(
            X, initial_centers, n_clusters,
            max_iter=max_iter,
            chunk_single=chunk_single,
            improve_threshold=improve_threshold,
            shift_tol=shift_tol,
            stability_threshold=stability_threshold,
            seed=rep
        )
        rows_D.append([
            ds_name, n, n_clusters, "D",
            chunk_single, improve_threshold,
            res["iters_single"], res["iters_double"],
            "Adaptive", res["elapsed_time"], res["mem_MB"], res["inertia"]
        ])

    return rows_D


def run_experiment_E(ds_name, X, y_true, n_clusters, initial_centers, config):
    """
    Experiment E — Mini-batch Hybrid K-Means
    Baseline: full double with budget = mb_iter + refine_iter.
    Variant:  MiniBatchKMeans(float32) -> KMeans(float64).
    """
    rows_E = []
    n = len(X)
    reps = int(config["n_repeats"])

    # Fixed POC parameters
    mb_iter = int(config["mb_iter_E"])
    mb_batch = int(config["mb_batch_E"])
    refine_iter = int(config["max_refine_iter_E"])
    tol_double_baseline = float(config["tol_double_baseline_E"])  # 0.0 => stop by budget

    budget = mb_iter + refine_iter

    for rep in range(reps):
        # Baseline: pure double with same total iteration budget
        c_d, l_d, it_d, it_s, t, mem, J = run_full_double(
            X, initial_centers, n_clusters, budget, tol_double_baseline, y_true
        )
        rows_E.append([
            ds_name, n, n_clusters, "E",
            mb_iter, mb_batch, refine_iter,
            it_s, it_d,
            "Double", t, mem, J
        ])

        # Variant: Mini-batch -> Full
        res = run_expE_minibatch_then_full(
            X, initial_centers, n_clusters,
            mb_iter=mb_iter, mb_batch=mb_batch, max_refine_iter=refine_iter,
            seed=rep
        )
        rows_E.append([
            ds_name, n, n_clusters, "E",
            mb_iter, mb_batch, refine_iter,
            res["iters_single"], res["iters_double"],
            "MiniBatch+Full", res["elapsed_time"], res["mem_MB"], res["inertia"]
        ])

    return rows_E


def run_experiment_F(ds_name, X, y_true, n_clusters, initial_centers, config):
    """
    Experiment F — Mixed Precision Per-Cluster
    Baseline: full double with same total max_iter_F.
    Variant:  per-cluster float32 Lloyd (optional freezing) -> sklearn KMeans(double).
    """
    rows_F = []
    n = len(X)
    reps = int(config["n_repeats"])

    # Fixed POC parameters
    max_iter_total = int(config["max_iter_F"])
    tol_double = float(config["tol_double_F"])
    tol_single = float(config["tol_single_F"])
    single_iter_cap = int(config["single_iter_cap_F"])
    freeze_stable = bool(config["freeze_stable_F"])
    freeze_patience = int(config["freeze_patience_F"])

    for rep in range(reps):
        # Baseline: pure double
        c_d, l_d, it_d, it_s, t, mem, J = run_full_double(
            X, initial_centers, n_clusters, max_iter_total, tol_double, y_true
        )
        rows_F.append([
            ds_name, n, n_clusters, "F",
            tol_single, tol_double, single_iter_cap, freeze_stable, freeze_patience,
            it_s, it_d,
            "Double", t, mem, J
        ])

        # Variant: per-cluster mixed precision
        res = run_expF_percluster_mixed(
            X, initial_centers,
            max_iter_total=max_iter_total,
            single_iter_cap=single_iter_cap,
            tol_single=tol_single,
            tol_double=tol_double,
            freeze_stable=freeze_stable,
            freeze_patience=freeze_patience,
            seed=rep
        )
        rows_F.append([
            ds_name, n, n_clusters, "F",
            tol_single, tol_double, single_iter_cap, freeze_stable, freeze_patience,
            res["iters_single"], res["iters_double"],
            "MixedPerCluster", res["elapsed_time"], res["mem_MB"], res["inertia"]
        ])

    return rows_F
