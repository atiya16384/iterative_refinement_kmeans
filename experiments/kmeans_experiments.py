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
    rows_D = []

    max_iter = config["max_iter_D"]
    tol_shift = config.get("tol_shift_D", 1e-3)
    seed = config.get("seed", 0)

    # Baseline (Double Precision)
    base = run_adaptive_hybrid(
        X, initial_centers, n_clusters,
        max_iter=max_iter,
        initial_precision='double',
        stability_threshold=0.0,
        inertia_improvement_threshold=0.0,
        refine_iterations=0,
        tol_shift=tol_shift,
        seed=seed,
        y_true=y_true
    )

    rows_D.append([
        ds_name, len(X), n_clusters,
        "D", "-", "-",  # Mode, tol_single, Cap
        base["iters_single"], base["iters_double"],
        "Double", base["elapsed_time"], base["mem_MB"], base["inertia"]
    ])

    # Sweep multiple configurations for adaptive
    for stab_thresh in [0.01, 0.02, 0.05]:
        for inertia_thresh in [0.005, 0.01, 0.02]:
            for refine_iters in [1, 2]:

                adv = run_adaptive_hybrid(
                    X, initial_centers, n_clusters,
                    max_iter=max_iter,
                    initial_precision='single',
                    stability_threshold=stab_thresh,
                    inertia_improvement_threshold=inertia_thresh,
                    refine_iterations=refine_iters,
                    tol_shift=tol_shift,
                    seed=seed,
                    y_true=y_true
                )

                rows_D.append([
                    ds_name, len(X), n_clusters,
                    "D", stab_thresh, "-",  # Mode, tol_single, Cap
                    adv["iters_single"], adv["iters_double"],
                    "Adaptive", adv["elapsed_time"], adv["mem_MB"], adv["inertia"]
                ])

    return rows_D


