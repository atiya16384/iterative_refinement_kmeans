import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
from aoclda.sklearn import skpatch
from experiments.svm_experiments import run_all
skpatch()
import pathlib
from visualisations.SVM_visualisations import (
    plot_svm_cap_vs_accuracy,
    plot_svm_cap_vs_time,
    plot_svm_tolerance_vs_accuracy,
    plot_svm_tolerance_vs_time,
)

from datasets.utils import (
    generate_synthetic_data, load_3d_road, load_susy, 
    synth_specs, real_datasets, columns_A, columns_B,
    columns_C, columns_D

)

from experiments.svm_precision import (
    svm_double_precision, svm_hybrid_precision

)


RESULTS_DIR = pathlib.Path("Results")
RESULTS_DIR.mkdir(exist_ok=True)

def print_summary(path, group_by):
    df = pd.read_csv(path)
    print(f"\n==== SUMMARY: {path.name.upper()} ====")
    summary = df.groupby(group_by)[['accuracy', 'Time', 'Memory_MB']].mean()
    print(summary)

def run_experiments():
    results_A, results_B, results_C, results_D = [], [], [], []

    config = {
        "n_repeats": 1,
        "tol_fixed_A": 1e-16,
        "tol_double_B": 1e-5,
        "caps": [0, 50, 100, 150, 200, 250, 300],
        "tolerances": [1e-1, 1e-2, 1e-3, 1e-4],
        "max_iter_C": 300,
        "perc_C": 0.8,
        "tol_D": 1e-3,
    }
    

    for tag, n, d, c, seed in synth_specs:
        X, y = generate_synthetic_data(n, d, c, seed)
        run_all(tag, X, y, config, results_A, results_B, results_C, results_D)
    
    for tag, loader in real_datasets:
        print(f"Loading {tag} …")
        X, y = loader()
        run_all(tag, X, y, config, results_A, results_B, results_C, results_D)
    

    df_A = pd.DataFrame(results_A, columns=columns_A)
    df_B = pd.DataFrame(results_B, columns=columns_B)
    df_C = pd.DataFrame(results_C, columns=columns_C)
    df_D = pd.DataFrame(results_D, columns=columns_D)
    
    df_A["Mode"] = "A"
    df_B["Mode"] = "B"
    df_C["Mode"] = "C"
    df_D["Mode"] = "D"


    df_A.to_csv(RESULTS_DIR / "svm_expA_caps.csv", index=False)
    df_B.to_csv(RESULTS_DIR / "svm_expB_tol.csv", index=False)
    df_C.to_csv(RESULTS_DIR / "svm_expC_80percent.csv", index=False)
    df_D.to_csv(RESULTS_DIR / "svm_expD_tol_fixed.csv", index=False)

    print("Saved:")
    print("- svm_expA_caps.csv")
    print("- svm_expB_tol.csv")
    print("- svm_expC_80percent.csv")
    print("- svm_expD_tol_fixed.csv")

    # Summaries
    print_summary(RESULTS_DIR / "svm_expA_caps.csv", ['DatasetName', 'Suite'])
    print_summary(RESULTS_DIR / "svm_expB_tol.csv", ['DatasetName', 'Suite'])
    print_summary(RESULTS_DIR / "svm_expC_80percent.csv", ['DatasetName', 'Suite'])
    print_summary(RESULTS_DIR / "svm_expD_tol_fixed.csv", ['DatasetName', 'Suite'])

    return df_A, df_B, df_C , df_D


if __name__ == "__main__":
    df_A, df_B, df_C, df_D = run_experiments()
    plot_svm_cap_vs_accuracy(df_A)
    plot_svm_cap_vs_time(df_A)
    plot_svm_tolerance_vs_accuracy(df_B)
    plot_svm_tolerance_vs_time(df_B)

