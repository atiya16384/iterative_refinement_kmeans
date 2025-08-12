# SVM_main.py
import pathlib, pandas as pd
from aoclda.sklearn import skpatch
skpatch()
from visualisations.SVM_visualisations import SVMVisualizer; 
from experiments.svm_experiments import SVMExperimentRunner
from datasets.utils import generate_synthetic_data, synth_specs, svm_columns_A, svm_columns_B

RESULTS_DIR = pathlib.Path("Results")
RESULTS_DIR.mkdir(exist_ok=True)

def print_summary(path, group_by):
    df = pd.read_csv(path)
    print(f"\n==== SUMMARY: {path.name.upper()} ====")
    print(df.groupby(group_by)[['Accuracy','Time','Memory_MB']].mean())

def run_experiments():
    # Tuned so hybrid is typically faster than double
        # SVM_main.py (inside run_experiments)
    config = {
        "n_repeats": 1,
    
        # SVC knobs
        "C": 1.0,
        "kernel": "rbf",
        "gamma": "scale",     # or a float, e.g., 0.1
        "keep_frac": 0.40,    # how many "hard" points to keep for Stage-2
    
        # ===== Experiment A (subset percentage) =====
        "max_iter_A": 300,
        "tol_fixed_A": 1e-4,
        "caps": [0, 1, 2, 5, 10, 20],  # interpret as percent of training used in Stage-1
    
        # ===== Experiment B (Stage-1 tolerance) =====
        "max_iter_B": 300,
        "tolerances": [1e-2, 5e-3, 1e-3],
        "tol_double_B": 1e-4,
        "cap_B": 10,          # fixed Stage-1 subset percentage for B
    }

    runner = SVMExperimentRunner(config)
    results_A, results_B = [], []

    # Datasets
    for tag, n, d, c, seed in synth_specs:
        X, y = generate_synthetic_data(n, d, c, seed)
        runner.run_all(tag, X, y)

    results_A, results_B = runner.get_results()
    df_A = pd.DataFrame(results_A, columns=svm_columns_A)
    df_B = pd.DataFrame(results_B, columns=svm_columns_B)

    df_A.to_csv(RESULTS_DIR / "svm_expA_caps.csv", index=False)
    df_B.to_csv(RESULTS_DIR / "svm_expB_tol.csv", index=False)

    print("Saved:")
    print("- svm_expA_caps.csv")
    print("- svm_expB_tol.csv")

    print_summary(RESULTS_DIR / "svm_expA_caps.csv", ['DatasetName','Suite'])
    print_summary(RESULTS_DIR / "svm_expB_tol.csv", ['DatasetName','Suite'])

    return df_A, df_B

if __name__ == "__main__":
    df_A, df_B = run_experiments()
    visualizer = SVMVisualizer()
    visualizer.plot_cap_vs_accuracy(df_A)
    visualizer.plot_cap_vs_time(df_A)
    visualizer.plot_tolerance_vs_accuracy(df_B)
    visualizer.plot_tolerance_vs_time(df_B)

