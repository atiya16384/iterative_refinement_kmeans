import pathlib
import pandas as pd
from aoclda.sklearn import skpatch
skpatch()
from datasets.utils import generate_synthetic_data, synth_specs, lr_columns_A, lr_columns_B
from experiments.logreg_experiments import run_experiment_A, run_experiment_B
from visualisations.LOGREG_visualisations import LogisticVisualizer

RESULTS_DIR = pathlib.Path("Results")
RESULTS_DIR.mkdir(exist_ok=True)

# ===================
# SINGLE SOURCE CONFIG
# ===================
# logreg_main.py (config)
config = {
    "n_repeats": 3,

    # Cap sweep
    "cap_grid": [0, 5, 10, 25, 50, 100, 150],
    "tol_fixed_A": 1e-3,     # fp32 target tolerance (looser)

    "max_iter_A": 300,

    # Tolerance sweep (no cap)
    "max_iter_B": 300,
    "tol_double_B": 1e-6,    # fp64 stricter target
    "tol_single_grid": [1e-1, 5e-2, 1e-2, 5e-3, 1e-3],
}

rows_A, rows_B = [], []

# Synthetic datasets
for tag, n, d, k, seed in synth_specs:
    X, y = generate_synthetic_data(n, d, k, seed)
    n_classes = len(set(y))

    for _ in range(config["n_repeats"]):
        rows_A.extend(run_experiment_A(tag, X, y, n_classes, config))
        rows_B.extend(run_experiment_B(tag, X, y, n_classes, config))

# Save results
df_A = pd.DataFrame(rows_A, columns=lr_columns_A)
df_B = pd.DataFrame(rows_B, columns=lr_columns_B)
df_A.to_csv(RESULTS_DIR / "logistic_results_expA.csv", index=False)
df_B.to_csv(RESULTS_DIR / "logistic_results_expB.csv", index=False)


# Summaries
print("\n==== SUMMARY: EXPERIMENT A ====")
print(df_A.groupby(["DatasetName", "NumClasses", "Mode", "Cap", "tolerance_single"])[
    ["Time", "Memory_MB", "Accuracy"]
].mean())

print("\n==== SUMMARY: EXPERIMENT B ====")
print(df_B.groupby(["DatasetName", "NumClasses", "Mode", "tolerance_single"])[
    ["Time", "Memory_MB", "Accuracy"]
].mean())


# After created df_A and df_B
log_vis = LogisticVisualizer()

# Experiment A plots
log_vis.plot_cap_vs_time(df_A)
log_vis.plot_cap_vs_accuracy(df_A)

# Experiment B plots
log_vis.plot_tolerance_vs_time(df_B)
log_vis.plot_tolerance_vs_accuracy(df_B)
