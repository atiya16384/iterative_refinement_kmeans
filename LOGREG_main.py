# LOGREG_main.py
import pathlib, pandas as pd
from aoclda.sklearn import skpatch; skpatch()
from experiments.logreg_experiments import LogRegExperimentRunner
from sklearn.datasets import make_classification
from datasets.utils import logreg_columns_A, logreg_columns_B

RESULTS_DIR = pathlib.Path("Results")
RESULTS_DIR.mkdir(exist_ok=True)

def print_summary(path, group_by):
    df = pd.read_csv(path)
    print(f"\n==== SUMMARY: {path.name} ====")
    print(df.groupby(group_by)[['Accuracy', 'Time', 'Memory_MB']].mean())

def run_experiments():
    # Simple, readable config
    config = {
        "n_repeats": 1,
        "epochs_A_total": 50,        # total iters for Exp-A
        "epochs_B_total": 50,        # total iters for Exp-B
        "tol_fixed_A": 1e-4,         # Exp-A uses fixed tol
        "tolerances": [5e-3, 1e-3, 5e-4],  # Exp-B sweeps tol_single
        "tol_double_B": 1e-4,        # polish tol
        "cap_B": 10,                 # stage-1 iters for Exp-B
        "caps": [0, 5, 10, 20],      # stage-1 iters for Exp-A
        "C": 1.0,
        "solver": "lbfgs",
        "test_size": 0.2,
        "seed": 0,
    }

    # Synthetic classification dataset
    X, y = make_classification(
        n_samples=60_000, n_features=80, n_informative=40, n_redundant=10,
        n_classes=5, class_sep=1.5, random_state=0
    )

    runner = LogRegExperimentRunner(config)
    runner.run_all("LOGREG_SYNTH", X, y)
    rows_A, rows_B = runner.get_results()

    df_A = pd.DataFrame(rows_A, columns=logreg_columns_A)
    df_B = pd.DataFrame(rows_B, columns=logreg_columns_B)
    df_A.to_csv(RESULTS_DIR / "logreg_expA_caps.csv", index=False)
    df_B.to_csv(RESULTS_DIR / "logreg_expB_tol.csv", index=False)

    print("Saved: logreg_expA_caps.csv, logreg_expB_tol.csv")
    print_summary(RESULTS_DIR / "logreg_expA_caps.csv", ["Suite"])
    print_summary(RESULTS_DIR / "logreg_expB_tol.csv", ["Suite"])

    return df_A, df_B

if __name__ == "__main__":
    run_experiments()
