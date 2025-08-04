from visualisations.kmeans_visualisations import KMeansVisualizer
from experiments.kmeans_precision import run_full_double, run_hybrid, run_adaptive_hybrid
import numpy as np
import time

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
    cap_C_pct = config["cap_C_pct"]  # e.g., 0.8 means 80% of max_iter_C
    n_repeats = config["n_repeats"]

    iter_cap = int(max_iter_C * cap_C_pct)

    for rep in range(n_repeats):
        # Baseline full double precision
        centers_double, labels_double, iters_double_tot, iters_single_tot, elapsed, mem_MB_double, inertia = run_full_double(
            X, initial_centers, n_clusters, max_iter_C, tol_fixed_C, y_true
        )
        rows_C.append([
            ds_name, n_samples, n_clusters, "C", cap_C_pct, tol_fixed_C,
            0, iters_double_tot, "Double", elapsed, mem_MB_double, inertia
        ])

        # Single hybrid run with capped single precision
        labels_hybrid, centers_hybrid, iters_single, iters_double, elapsed_hybrid, mem_MB_hybrid, inertia_hybrid = run_hybrid(
            X, initial_centers, n_clusters,
            max_iter_total=max_iter_C,
            single_iter_cap=iter_cap,
            tol_single=tol_fixed_C,
            tol_double=tol_fixed_C,
            y_true=y_true,
            seed=rep
        )

        rows_C.append([
            ds_name, n_samples, n_clusters, "C", cap_C_pct, tol_fixed_C,
            iters_single, iters_double, "Hybrid", elapsed_hybrid, mem_MB_hybrid, inertia_hybrid
        ])

        # Optional plot (just once)
        if rep == 0:
            X_vis, centers_vis, xx, yy, labels_grid = KMeansVisualizer.pca_2d_view(X, centers_hybrid)
            filename = f"{ds_name}_n{n_samples}_c{n_clusters}_C_cap{int(cap_C_pct*100)}"
            title = f"{ds_name}: n={n_samples}, c={n_clusters}, cap={int(cap_C_pct*100)}%"
            KMeansVisualizer.plot_clusters(X_vis, labels_hybrid, centers_vis, xx, yy, labels_grid, title=title, filename=filename)

    return rows_C

def run_experiment_D(ds_name, X, y_true, n_clusters, initial_centers, config):
    rows_D = []

    max_iter = config["max_iter_D"]
    tol_shift = config.get("tol_shift_D", 1e-3)
    stability_threshold = config.get("stability_threshold_D", 0.02)
    inertia_improvement_threshold = config.get("inertia_improvement_threshold_D", 0.02)
    refine_iterations = config.get("refine_iterations_D", 2)
    seed = config.get("seed", 0)

    # === Baseline (Double Precision) ===
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
    labels_b, centers_b, switched_b, iters_b, time_b, mem_b, inertia_b, iters_single_b, iters_double_b = base

    rows_D.append([
        ds_name, len(X), n_clusters,
        "Double", "-", "-",  # Mode, tol_single, Cap
        iters_single_b, iters_double_b,
        "D", time_b, mem_b, inertia_b
    ])

    # === Adaptive-Hybrid ===
    adv = run_adaptive_hybrid(
        X, initial_centers, n_clusters,
        max_iter=max_iter,
        initial_precision='single',
        stability_threshold=stability_threshold,
        inertia_improvement_threshold=inertia_improvement_threshold,
        refine_iterations=refine_iterations,
        tol_shift=tol_shift,
        seed=seed,
        y_true=y_true
    )
    labels_h, centers_h, switched_h, iters_h, time_h, mem_h, inertia_h, iters_single_h, iters_double_h = adv

    rows_D.append([
        ds_name, len(X), n_clusters,
        "Adaptive", stability_threshold, "-",  # Mode, tol_single, Cap
        iters_single_h, iters_double_h,
        "D", time_h, mem_h, inertia_h
    ])

    return rows_D
