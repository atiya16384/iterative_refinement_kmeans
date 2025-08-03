from visualisations.kmeans_visualisations import KMeansVisualizer
from experiments.kmeans_precision import run_full_double, run_hybrid
import numpy as np

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
        centers_double, labels_double, iters_double_tot, iters_single_tot, elapsed, mem_MB_double, ari, dbi, inertia = run_full_double(
            X_cur, initial_centers, n_clusters, max_iter, tol_fixed_A, y_true_cur
        )

        rows_A.append([
            ds_name, n_samples, n_clusters, "A", 0, 0,
            iters_single_tot, iters_double_tot, "Double", elapsed, mem_MB_double,
            ari, dbi, inertia
        ])

    for cap in cap_grid:
        for rep in range(n_repeats):
            # Hybrid run
            labels_hybrid, centers_hybrid, iters_single, iters_double, elapsed_hybrid, mem_MB_hybrid, ari_hybrid, dbi_hybrid, inertia_hybrid = run_hybrid(
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
                ari_hybrid, dbi_hybrid, inertia_hybrid
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
        centers_double, labels_double, iters_double_tot, iters_single_tot, elapsed, mem_MB_double, ari, dbi, inertia = run_full_double(
        X_cur, initial_centers, n_clusters, max_iter_B, tol_double_B, y_true_cur
        )

        rows_B.append([ ds_name, n_samples, n_clusters, "B", tol_double_B,  iters_single_tot, iters_double_tot, "Double", elapsed, mem_MB_double,
                    ari, dbi, inertia])
        print(f"[Double Baseline - Exp B] tol={tol_double_B} | iter_double={iters_double_tot}")
        print(f"The total number of features is : F={n_features}")


    for tol_s in tol_single_grid:
        for rep in range(n_repeats):
        #  hybrid run
            labels_hybrid, centers_hybrid, iters_single, iters_double, elapsed_hybrid, mem_MB_hybrid, ari_hybrid, dbi_hybrid, inertia_hybrid = run_hybrid(
                X_cur, initial_centers, n_clusters, max_iter_total=max_iter_B, tol_single = tol_s, tol_double = tol_double_B, single_iter_cap=max_iter_B, y_true= y_true_cur, seed = rep
            )

            print(f"Tol_single: {tol_s}, Iter Single: {iters_single}, Iter Double: {iters_double}, Total: {iters_single + iters_double}")
            print(f"The total number of features is : F={n_features}")


            rows_B.append([ds_name, n_samples, n_clusters, "B", tol_s,  iters_single, iters_double, "Hybrid", elapsed_hybrid, mem_MB_hybrid,
                    ari_hybrid, dbi_hybrid, inertia_hybrid])
        
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
    X_cur = X
    y_true_cur = y_true

    max_iter_C = config["max_iter_C"]
    tol_fixed_C = config["tol_fixed_C"]
    cap_C_pct = config["cap_C"]  # e.g., 0.8 (i.e., 80% of max_iter_C)
    n_repeats = config["n_repeats"]

    cap_range = range(0, int(max_iter_C * cap_C_pct) + 1)

    for rep in range(n_repeats):
        # Baseline double
        centers_double, labels_double, iters_double_tot, iters_single_tot, elapsed, mem_MB_double, ari, dbi, inertia = run_full_double(
            X_cur, initial_centers, n_clusters, max_iter_C, tol_fixed_C, y_true_cur
        )
        rows_C.append([
            ds_name, n_samples, n_clusters, "C", cap_C_pct, tol_fixed_C,
            0, iters_double_tot, "Double", elapsed, mem_MB_double, ari, dbi, inertia
        ])

        # Sweep over cap values
        for cap in cap_range:
            labels_hybrid, centers_hybrid, iters_single, iters_double, elapsed_hybrid, mem_MB_hybrid, ari_hybrid, dbi_hybrid, inertia_hybrid = run_hybrid(
                X_cur, initial_centers, n_clusters,
                max_iter_total=max_iter_C,
                single_iter_cap=cap,
                tol_single=tol_fixed_C,
                tol_double=tol_fixed_C,
                y_true=y_true_cur,
                seed=rep
            )

            rows_C.append([
                ds_name, n_samples, n_clusters, "C", cap_C_pct, tol_fixed_C,
                iters_single, iters_double, "Hybrid", elapsed_hybrid, mem_MB_hybrid,
                ari_hybrid, dbi_hybrid, inertia_hybrid
            ])

            if rep == 0 and cap == cap_range[-1]:  # Plot only once at final cap
                X_vis, centers_vis, xx, yy, labels_grid = KMeansVisualizer.pca_2d_view(X_cur, centers_hybrid)
                filename = f"{ds_name}_n{n_samples}_c{n_clusters}_C_cap{cap_C_pct}"
                title = f"{ds_name}: n={n_samples}, c={n_clusters}, cap={cap_C_pct}"
                KMeansVisualizer.plot_clusters(X_vis, labels_hybrid, centers_vis, xx, yy, labels_grid, title=title, filename=filename)

    return rows_C

def run_experiment_D(ds_name, X, y_true, n_clusters, initial_centers, config):

    rows_D = []
    n_samples = len(X)

    # Load hyperparameters
    max_iter = config["max_iter_D"]
    tol_final = config["tol_double_D"]
    tol_single_switch = config.get("tol_single_D", 1e-3)
    switch_tol = config.get("switch_tol", 1e-5)        # Switch if inertia doesn't improve by this
    switch_shift = config.get("switch_shift", 1e-4)    # Switch if centroid movement is small
    n_repeats = config["n_repeats"]

    for rep in range(n_repeats):
        # === Double precision baseline ===
        centers_double, labels_double, iters_double_tot, iters_single_tot, elapsed_double, mem_MB_double, ari_double, dbi_double, inertia_double = run_full_double(
            X, initial_centers, n_clusters, max_iter, tol_final, y_true
        )
        rows_D.append([
            ds_name, n_samples, n_clusters, "D", tol_final, 0,
            iters_single_tot, iters_double_tot, "Double", elapsed_double, mem_MB_double,
            ari_double, dbi_double, inertia_double
        ])

        # === Adaptive hybrid run ===
        labels_hybrid, centers_hybrid, iters_single, iters_double, elapsed_hybrid, mem_MB_hybrid, ari_hybrid, dbi_hybrid, inertia_hybrid = run_adaptive_hybrid(
            X, initial_centers, n_clusters,
            max_iter=max_iter,
            tol_final=tol_final,
            y_true=y_true,
            switch_tol=switch_tol,
            switch_shift=switch_shift,
            seed=rep
        )
        rows_D.append([
            ds_name, n_samples, n_clusters, "D", tol_single_switch, 0,
            iters_single, iters_double, "Hybrid", elapsed_hybrid, mem_MB_hybrid,
            ari_hybrid, dbi_hybrid, inertia_hybrid
        ])

        # Optional: Save visualization of clustering
        if rep == 0:
            X_vis, centers_vis, xx, yy, labels_grid = KMeansVisualizer.pca_2d_view(X, centers_hybrid)
            filename = f"{ds_name}_n{n_samples}_c{n_clusters}_D_adaptive"
            title = f"{ds_name}: n={n_samples}, c={n_clusters}, Adaptive Hybrid"
            KMeansVisualizer.plot_clusters(X_vis, labels_hybrid, centers_vis, xx, yy, labels_grid, title=title, filename=filename)

    return rows_D

