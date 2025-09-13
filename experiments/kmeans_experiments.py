from visualisations.kmeans_visualisations import KMeansVisualizer
from experiments.kmeans_precision import (
    run_full_single, run_full_double, run_hybrid,
    run_expD_adaptive_sklearn, run_expE_minibatch_then_full, run_expF_percluster_mixed
)
import numpy as np
from math import ceil

def run_experiment_A(ds_name, X, y_true, n_clusters, initial_centers, config):
    """
    Columns produced (A):
      DatasetName, DatasetSize, NumClusters, Mode, Cap, tolerance_single,
      ItersSingle, ItersDouble, Suite, Time, PeakMB, Inertia
    """
    rows_A = []
    n_samples = int(len(X))

    max_iter = int(config["max_iter_A"])
    tol_fixed_A = float(config["tol_fixed_A"])
    cap_grid = list(config["cap_grid"])
    n_repeats = int(config["n_repeats"])
    algorithm = str(config.get("algorithm", "lloyd"))
    late_cast = bool(config.get("late_cast", False))

    for cap in cap_grid:
        for rep in range(n_repeats):
            # --- Single baseline (float32) ---
            c_s, l_s, it_s_s, it_d_s, t_s, peak_s, J_s = run_full_single(
                X, initial_centers, n_clusters, max_iter, tol_fixed_A, y_true,
                algorithm=algorithm, random_state=rep
            )
            rows_A.append({
                "DatasetName": ds_name, "DatasetSize": n_samples, "NumClusters": n_clusters,
                "Mode": "A", "Cap": int(cap), "tolerance_single": tol_fixed_A,
                "ItersSingle": it_s_s, "ItersDouble": it_d_s, "Suite": "Single",
                "Time": t_s, "PeakMB": peak_s, "Inertia": J_s,
            })

            # --- Double baseline (float64) ---
            c_d, l_d, it_d, it_s, t_d, peak_d, J_d = run_full_double(
                X, initial_centers, n_clusters, max_iter, tol_fixed_A, y_true,
                algorithm=algorithm, random_state=rep
            )
            rows_A.append({
                "DatasetName": ds_name, "DatasetSize": n_samples, "NumClusters": n_clusters,
                "Mode": "A", "Cap": int(cap), "tolerance_single": tol_fixed_A,
                "ItersSingle": it_s, "ItersDouble": it_d, "Suite": "Double",
                "Time": t_d, "PeakMB": peak_d, "Inertia": J_d,
            })

            # --- Hybrid (cap) ---
            labels_h, centers_h, it_s_h, it_d_h, t_h, peak_h, J_h = run_hybrid(
                X, initial_centers, n_clusters,
                max_iter_total=max_iter,
                single_iter_cap=int(cap),
                tol_single=tol_fixed_A,
                tol_double=tol_fixed_A,
                y_true=y_true,
                seed=rep,
                algorithm=algorithm,
                late_cast=late_cast,
            )
            rows_A.append({
                "DatasetName": ds_name, "DatasetSize": n_samples, "NumClusters": n_clusters,
                "Mode": "A", "Cap": int(cap), "tolerance_single": tol_fixed_A,
                "ItersSingle": it_s_h, "ItersDouble": it_d_h, "Suite": "Hybrid",
                "Time": t_h, "PeakMB": peak_h, "Inertia": J_h,
            })
    return rows_A


def run_experiment_B(ds_name, X, y_true, n_clusters, initial_centers, config):
    """
    Columns produced (B):
      DatasetName, DatasetSize, NumClusters, Mode, tolerance_single,
      ItersSingle, ItersDouble, Suite, Time, PeakMB, Inertia
    """
    rows_B = []
    n_samples = int(len(X))

    max_iter_B      = int(config["max_iter_B"])
    tol_double_B    = float(config["tol_double_B"])
    tol_single_grid = list(config["tol_single_grid"])
    n_repeats       = int(config["n_repeats"])
    algorithm       = str(config.get("algorithm", "lloyd"))
    late_cast       = bool(config.get("late_cast", False))

    for tol_s in tol_single_grid:
        tol_s = float(tol_s)
        for rep in range(n_repeats):
            # --- Single baseline at tol_s ---
            c_s, l_s, it_s_s, it_d_s, t_s, peak_s, J_s = run_full_single(
                X, initial_centers, n_clusters, max_iter_B, tol_s, y_true,
                algorithm=algorithm, random_state=rep
            )
            rows_B.append({
                "DatasetName": ds_name, "DatasetSize": n_samples, "NumClusters": n_clusters,
                "Mode": "B", "tolerance_single": tol_s,
                "ItersSingle": it_s_s, "ItersDouble": it_d_s, "Suite": "Single",
                "Time": t_s, "PeakMB": peak_s, "Inertia": J_s,
            })

            # --- Double baseline at tol_double_B ---
            c_d, l_d, it_d, it_s, t_d, peak_d, J_d = run_full_double(
                X, initial_centers, n_clusters, max_iter_B, tol_double_B, y_true,
                algorithm=algorithm, random_state=rep
            )
            rows_B.append({
                "DatasetName": ds_name, "DatasetSize": n_samples, "NumClusters": n_clusters,
                "Mode": "B", "tolerance_single": tol_s,
                "ItersSingle": it_s, "ItersDouble": it_d, "Suite": "Double",
                "Time": t_d, "PeakMB": peak_d, "Inertia": J_d,
            })

            # --- Hybrid: tol_s for Phase A, tol_double_B for Phase B ---
            labels_h, centers_h, it_s_h, it_d_h, t_h, peak_h, J_h = run_hybrid(
                X, initial_centers, n_clusters,
                max_iter_total=max_iter_B,
                single_iter_cap=max_iter_B,    # no early cap; tol drives stopping
                tol_single=tol_s,
                tol_double=tol_double_B,
                y_true=y_true,
                seed=rep,
                algorithm=algorithm,
                late_cast=late_cast,
            )
            rows_B.append({
                "DatasetName": ds_name, "DatasetSize": n_samples, "NumClusters": n_clusters,
                "Mode": "B", "tolerance_single": tol_s,
                "ItersSingle": it_s_h, "ItersDouble": it_d_h, "Suite": "Hybrid",
                "Time": t_h, "PeakMB": peak_h, "Inertia": J_h,
            })
    return rows_B

def run_experiment_C(ds_name, X, y_true, n_clusters, initial_centers, config):
    """
    Columns produced (C):
      DatasetName, DatasetSize, NumClusters, Mode, Cap, tolerance_single,
      ItersSingle, ItersDouble, Suite, Time, PeakMB, Inertia
    """
    rows_C = []
    n_samples = int(len(X))

    max_iter_C       = int(config["max_iter_C"])
    tol_fixed_C      = float(config["tol_fixed_C"])
    cap_percentages  = list(config.get("cap_percentages", [0.0, 0.2, 0.4, 0.6, 0.8]))
    n_repeats        = int(config.get("n_repeats", 1))
    algorithm        = str(config.get("algorithm", "lloyd"))
    late_cast        = bool(config.get("late_cast", False))

    for rep in range(n_repeats):
        # ---- baselines once per repeat ----
        c_s, l_s, it_s_s, it_d_s, t_s, peak_s, J_s = run_full_single(
            X, initial_centers, n_clusters, max_iter_C, tol_fixed_C, y_true,
            algorithm=algorithm, random_state=rep
        )
        rows_C.append({
            "DatasetName": ds_name, "DatasetSize": n_samples, "NumClusters": n_clusters,
            "Mode": "C", "Cap": "full", "tolerance_single": tol_fixed_C,
            "ItersSingle": it_s_s, "ItersDouble": it_d_s, "Suite": "Single",
            "Time": t_s, "PeakMB": peak_s, "Inertia": J_s,
        })

        c_d, l_d, it_d_d, it_s_d, t_d, peak_d, J_d = run_full_double(
            X, initial_centers, n_clusters, max_iter_C, tol_fixed_C, y_true,
            algorithm=algorithm, random_state=rep
        )
        rows_C.append({
            "DatasetName": ds_name, "DatasetSize": n_samples, "NumClusters": n_clusters,
            "Mode": "C", "Cap": "full", "tolerance_single": tol_fixed_C,
            "ItersSingle": it_s_d, "ItersDouble": it_d_d, "Suite": "Double",
            "Time": t_d, "PeakMB": peak_d, "Inertia": J_d,
        })

        # ---- hybrid sweep over cap fractions ----
        for pct in cap_percentages:
            pct = float(pct)
            single_cap = int(ceil(max_iter_C * pct))

            labels_h, centers_h, it_s_h, it_d_h, t_h, peak_h, J_h = run_hybrid(
                X, initial_centers, n_clusters,
                max_iter_total=max_iter_C,
                single_iter_cap=single_cap,
                tol_single=tol_fixed_C,
                tol_double=tol_fixed_C,
                y_true=y_true,
                seed=rep,
                algorithm=algorithm,
                late_cast=late_cast,
            )
            rows_C.append({
                "DatasetName": ds_name, "DatasetSize": n_samples, "NumClusters": n_clusters,
                "Mode": "C", "Cap": pct, "tolerance_single": tol_fixed_C,
                "ItersSingle": it_s_h, "ItersDouble": it_d_h, "Suite": "Hybrid",
                "Time": t_h, "PeakMB": peak_h, "Inertia": J_h,
            })

            # Optional PCA snapshot (first repeat only)
            if rep == 0:
                X_vis, centers_vis, xx, yy, labels_grid = KMeansVisualizer.pca_2d_view(X, centers_h)
                KMeansVisualizer.plot_clusters(
                    X_vis, labels_h, centers_vis, xx, yy, labels_grid,
                    title=f"{ds_name}: cap = {int(pct*100)}% iter",
                    filename=f"{ds_name}_C_cap{int(pct*100)}"
                )

    return rows_C

    
def run_experiment_D(ds_name, X, y_true, n_clusters, initial_centers, config):
    """
    Experiment D — Adaptive Hybrid (global switch)
      • Baselines: full Single and full Double (same max_iter & tol)
      • Variant : repeated f32 bursts (size = chunk_single) until stall → finish in f64
    We sweep over a grid of `chunk_single` values so the plots show curves rather than a dot.
    """

    rows_D = []
    n_samples = len(X)

    # --- config (with safe defaults) ---
    reps                  = int(config.get("n_repeats", 1))
    max_iter              = int(config.get("max_iter_D", 300))
    tol_double_baseline   = float(config.get("tol_double_baseline_D", 1e-6))

    # sweep this:
    chunk_grid            = list(config.get("chunk_single_grid_D", [5, 10, 25, 50, 100, 150, 200]))

    # adaptive knobs (kept fixed across the sweep, but you can make them grids too)
    improve_threshold     = float(config.get("improve_threshold_D", 1e-3))
    shift_tol             = float(config.get("shift_tol_D", 1e-3))
    stability_threshold   = float(config.get("stability_threshold_D", 0.02))

    for rep in range(reps):
        for ch in chunk_grid:
            # ---------- Baseline: full SINGLE (duplicate per `ch` so plotting/joining is easy) ----------
            c_s, l_s, it_s_s, it_d_s, t_s, mem_s, J_s = run_full_single(
                X, initial_centers, n_clusters, max_iter, tol_double_baseline, y_true
            )
            rows_D.append([
                ds_name, n_samples, n_clusters,          # DatasetName, DatasetSize, NumClusters
                "D",                                     # Experiment label
                ch, improve_threshold, shift_tol, stability_threshold,  # params we used
                it_s_s, it_d_s,                          # iter_single, iter_double
                "Single",                                # Suite
                t_s, mem_s, J_s,                         # Time, Memory_MB, Inertia
                rep                                      # repeat id (optional)
            ])

            # ---------- Baseline: full DOUBLE ----------
            c_d, l_d, it_d, it_s, t_d, mem_d, J_d = run_full_double(
                X, initial_centers, n_clusters, max_iter, tol_double_baseline, y_true
            )
            rows_D.append([
                ds_name, n_samples, n_clusters,
                "D",
                ch, improve_threshold, shift_tol, stability_threshold,
                it_s, it_d,
                "Double",
                t_d, mem_d, J_d,
                rep
            ])

            # ---------- Variant: Adaptive (f32 bursts → f64 finish) ----------
            res = run_expD_adaptive_sklearn(
                X, initial_centers, n_clusters,
                max_iter=max_iter,
                chunk_single=int(ch),
                improve_threshold=improve_threshold,
                shift_tol=shift_tol,
                stability_threshold=stability_threshold,
                seed=rep,
            )
            rows_D.append([
                ds_name, n_samples, n_clusters,
                "D",
                ch, improve_threshold, shift_tol, stability_threshold,
                res["iters_single"], res["iters_double"],
                "Adaptive",
                res["elapsed_time"], res["mem_MB"], res["inertia"],
                rep
            ])

    return rows_D

def run_experiment_E(ds_name, X, y_true, n_clusters, initial_centers, config):
    rows_E = []
    n = len(X)
    reps = int(config["n_repeats"])

    mb_iter_grid = config.get("E_mb_iter_grid", [config["mb_iter_E"]])
    batch_grid   = config.get("E_batch_grid",   [config["mb_batch_E"]])
    refine_grid  = config.get("E_refine_grid",  [config["max_refine_iter_E"]])
    tol_double_baseline = float(config["tol_double_baseline_E"])

    for rep in range(reps):
        for mb_iter in mb_iter_grid:
            for mb_batch in batch_grid:
                for refine_iter in refine_grid:
                    budget = mb_iter + refine_iter

                    # Baseline: pure single with same total budget
                    c_s, l_s, it_s_s, it_d_s, t_s, mem_s, J_s = run_full_single(
                        X, initial_centers, n_clusters, budget, tol_double_baseline, y_true
                    )
                    rows_E.append([
                        ds_name, n, n_clusters, "E",
                        mb_iter, mb_batch, refine_iter,
                        it_s_s, it_d_s,
                        "Single", t_s, mem_s, J_s
                    ])

                    # Baseline: pure double with same total budget
                    c_d, l_d, it_d, it_s, t, mem, J = run_full_double(
                        X, initial_centers, n_clusters, budget, tol_double_baseline, y_true
                    )
                    rows_E.append([
                        ds_name, n, n_clusters, "E",
                        mb_iter, mb_batch, refine_iter,
                        it_s, it_d,
                        "Double", t, mem, J
                    ])

                    # Variant: MiniBatch -> Full
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
    rows_F = []
    n = len(X)
    reps = int(config["n_repeats"])

    max_iter_total = int(config["max_iter_F"])
    tol_double     = float(config["tol_double_F"])
    freeze_stable  = bool(config["freeze_stable_F"])
    freeze_patience= int(config["freeze_patience_F"])

    cap_grid       = config.get("F_cap_grid",       [config["single_iter_cap_F"]])
    tol_single_grid= config.get("F_tol_single_grid",[config["tol_single_F"]])

    for rep in range(reps):
        for single_iter_cap in cap_grid:
            for tol_single in tol_single_grid:
                # Baseline: pure double with same total budget
                # Baseline: pure single with same total budget
                c_s, l_s, it_s_s, it_d_s, t_s, mem_s, J_s = run_full_single(
                    X, initial_centers, n_clusters, max_iter_total, tol_double, y_true
                )
                rows_F.append([
                    ds_name, n, n_clusters, "F",
                    tol_single, tol_double, single_iter_cap, freeze_stable, freeze_patience,
                    it_s_s, it_d_s,
                    "Single", t_s, mem_s, J_s
                ])

                c_d, l_d, it_d, it_s, t, mem, J = run_full_double(
                    X, initial_centers, n_clusters, max_iter_total, tol_double, y_true
                )
                rows_F.append([
                    ds_name, n, n_clusters, "F",
                    tol_single, tol_double, single_iter_cap, freeze_stable, freeze_patience,
                    it_s, it_d,
                    "Double", t, mem, J
                ])

                # Variant: per‑cluster mixed precision
                res = run_expF_percluster_mixed(
                    X, initial_centers,
                    max_iter_total=max_iter_total,
                    single_iter_cap=int(single_iter_cap),
                    tol_single=float(tol_single),
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











