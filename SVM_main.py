import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
from aoclda.sklearn import skpatch
skpatch()
import pathlib
from experiments.svm_experiments import SVMExperimentRunner
from datasets.utils import (
    generate_synthetic_data, load_3d_road, load_susy, 
    synth_specs, real_datasets, svm_columns_A,  svm_columns_B
)


from visualisations.SVM_visualisations import SVMVisualizer

RESULTS_DIR = pathlib.Path("Results")
RESULTS_DIR.mkdir(exist_ok=True)

def print_summary(path, group_by):
    df = pd.read_csv(path)
    print(f"\n==== SUMMARY: {path.name.upper()} ====")
    summary = df.groupby(group_by)[['Accuracy', 'Time', 'Memory_MB']].mean()
    print(summary)

def run_experiments():
    results_A, results_B  = [], []
    config = {
        "n_repeats": 1,

        "tol_fixed_A": 1e-16,
        "caps": [0, 50, 100, 150, 200, 250, 300],

        "tol_double_B": 1e-5,
        "tolerances": [1e-1, 1e-2, 1e-3, 1e-4],

    }
    
    runner = SVMExperimentRunner(config)
    results_A, results_B = runner.get_results()
    # Synthetic datasets
    for tag, n, d, c, seed in synth_specs:
        X, y = generate_synthetic_data(n, d, c, seed)
        runner.run_all(tag, X, y)
    
    # Real-world datasets
    # for tag, loader in real_datasets:
    #     X, y = loader()
    #     runner.run_all(tag, X, y)
    
    

    df_A = pd.DataFrame(results_A, columns=svm_columns_A)
    df_B = pd.DataFrame(results_B, columns=svm_columns_B)

    df_A["Mode"] = "A"
    df_B["Mode"] = "B"

    df_A.to_csv(RESULTS_DIR / "svm_expA_caps.csv", index=False)
    df_B.to_csv(RESULTS_DIR / "svm_expB_tol.csv", index=False)

    print("Saved:")
    print("- svm_expA_caps.csv")
    print("- svm_expB_tol.csv")


    # Summaries
    print_summary(RESULTS_DIR / "svm_expA_caps.csv", ['DatasetName', 'Suite'])
    print_summary(RESULTS_DIR / "svm_expB_tol.csv", ['DatasetName', 'Suite'])


    return df_A, df_B


if __name__ == "__main__":
    df_A, df_B = run_experiments()
    visualizer = SVMVisualizer()
    visualizer.plot_cap_vs_accuracy(df_A)
    visualizer.plot_cap_vs_time(df_A)
    visualizer.plot_tolerance_vs_accuracy(df_B)
    visualizer.plot_tolerance_vs_time(df_B)

