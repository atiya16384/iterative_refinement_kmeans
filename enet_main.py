# ENET_main.py
import pathlib
import pandas as pd
from aoclda.sklearn import skpatch
skpatch()  # enable AOCL patching before we instantiate sklearn estimators
import numpy as np

from datasets.utils import (
    generate_synthetic_data_en, enet_specs
)
from experiments.enet_experiments import (
    run_experiment_A, run_experiment_B,
    en_columns_A, en_columns_B
)

RESULTS_DIR = pathlib.Path("Results")
RESULTS_DIR.mkdir(exist_ok=True)

# =====================
# Config (tweak freely)
# =====================
config = {
    "n_repeats": 1,

    # ElasticNet/Lasso knobs
    "alpha": 1e-3,       # regularization strength (higher => sparser/faster)
    "l1_ratio": 0.5,     # 1.0 = Lasso, 0.0 = Ridge-like, (0,1) = ElasticNet

    # --- Exp A: cap sweep at fixed fp32 tol ---
    "cap_grid": [1, 5, 10, 20, 50, 100],
    "tol_fixed_A": 1e-3,
    "max_iter_A": 300,
    "tol_double_A": 1e-6,

    # --- Exp B: tol sweep (no cap) ---
    "max_iter_B": 300,
    "tol_double_B": 1e-6,
    "tol_single_grid": [1e-2, 5e-3, 1e-3, 5e-4],
}

rows_A, rows_B = [], []

# Synthetic datasets
for tag, n, d, sparsity, noise, seed in enet_specs:
    X, y = generate_synthetic_data_en(
        n_samples=n, n_features=d, seed=seed,
        sparsity=sparsity, noise=noise, rho=0.5
    )
    for _ in range(config["n_repeats"]):
        rows_A.extend(run_experiment_A(tag, X, y, d, config))
        rows_B.extend(run_experiment_B(tag, X, y, d, config))

# Save results
df_A = pd.DataFrame(rows_A, columns=en_columns_A)
df_B = pd.DataFrame(rows_B, columns=en_columns_B)
df_A.to_csv(RESULTS_DIR / "enet_results_expA.csv", index=False)
df_B.to_csv(RESULTS_DIR / "enet_results_expB.csv", index=False)

# Quick summaries
print("\n==== SUMMARY: EXPERIMENT A ====")
print(df_A.groupby(["DatasetName", "NumFeatures", "Mode", "Cap", "tolerance_single"])[
    ["Time","Memory_MB","R2","MSE"]
].mean())

print("\n==== SUMMARY: EXPERIMENT B ====")
print(df_B.groupby(["DatasetName", "NumFeatures", "Mode", "tolerance_single"])[
    ["Time","Memory_MB","R2","MSE"]
].mean())

print("\nSaved CSVs to:", RESULTS_DIR.resolve())
