# LOGREG_main.py
import pathlib
import pandas as pd
from aoclda.sklearn import skpatch
skpatch()

from datasets.utils import generate_synthetic_data, synth_specs, lr_columns_A, lr_columns_B
from experiments.logreg_experiments import run_experiment_A, run_experiment_B
from visualisations.LOGREG_visualisations import LogisticVisualizer

RESULTS_DIR = pathlib.Path("Results")
RESULTS_DIR.mkdir(exist_ok=True)

# Config
config = {
    "n_repeats": 1,
    "cap_grid": [1, 10, 20, 50, 100],
    "tol_fixed_A": 1e-2,
    "max_iter_A": 300,
    "max_iter_B": 200,
    "tol_double_B": 1e-11,
    "tol_single_grid": [1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-10],
}

rows_A, rows_B = [], []

# Run synthetic datasets
for tag, n, d, k, seed in synth_specs:
    X, y = generate_synthetic_data(n_samples=n, n_features=d, n_clusters=k, random_state=seed)
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
print(df_A.groupby(["DatasetName", "NumClasses", "Mode", "Cap", "tolerance_single"])[["Time", "Memory_MB", "Accuracy"]].mean())

print("\n==== SUMMARY: EXPERIMENT B ====")
print(df_B.groupby(["DatasetName", "NumClasses", "Mode", "tolerance_single"])[["Time", "Memory_MB", "Accuracy"]].mean())


