

import numpy as np
import time
from aoclda.sklearn import skpatch
skpatch() # Apply AOCL patch before any SM usage
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.cluster import SVC
import pathlib
import pandas as pd
import matplotlib.pyplot as plt

RUN_EXPERIMENT_A = True
RUN_EXPERIMENT_B = True
RESULTS_DIR = pathlib.Path("Results_SVM")

# CONFIGURATION PARAMETERS 
dataset_sizes = [100000]
# for the cluster size we are varying this for all datasets
n_clusters_list = [30]

max_iter = 300

# Understand what the experiment parameters mean
# Have the cap grid based on a percentage of the max iteration
tol_fixed_A = 0
# varying the capmax_percentage
max_iter_A = 300
    # start time 
cap_grid = [0, 50, 100, 150, 200, 250, 300]

# we may be assigning the max iterations to be the single iteration cap
max_iter_B = 1000
# tolerance at which we change from single to double 
tol_double_B = 1e-5
tol_single_grid = [1e-2, 1e-3, 1e-4]

n_repeats = 1
rng_global = np.random.default_rng(0)

def generate_data(n_samples, n_features, n_clusters, random_state):
    X, y_true = make_blobs(n_samples=n_samples, n_features=n_features, centers=n_clusters, random_state=random_state)
    return X.astype(np.float64), y_true

def svm_run_double(X, y, max_iter, tol, C=1.0, kernel='rbf', test_size=0.2, seed=0):
    # split into training and test dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
    start_time = time.time()
    
    svm = SVC(C=C, kernel=kernel, gamma='scale', tol=tol, 
              max_iter=max_iter, random_state=seed)
    
    svm.fit(X_train, y_train)
    elapsed_time = time.time() - start_time

    y_pred = svm.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    support_vectors_count = len(svm.support_)

    iterations = svm.n_iter_
    memory_MB = X_train.astype(np.float64).nbytes / 1e6

    return (iterations, elapsed_time, acc, f1, memory_MB, support_vectors_count)

def svm_run_hybrid(X, y, max_iter_total, tol_single, tol_double, single_iter_cap, 
                         C=1.0, kernel='rbf', test_size=0.2, seed=0):
    
    # Split dataset into train-test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

    # Convert to single precision for initial training
    X_train_single = X_train.astype(np.float32)
    
    # Single-precision phase
    start_single = time.time()
    svm_single = SVC(C=C, kernel=kernel, gamma='scale', tol=tol_single,
                     max_iter=single_iter_cap, random_state=seed)
    svm_single.fit(X_train_single, y_train)
    elapsed_single = time.time() - start_single

    # Evaluate single-precision residual (training error)
    residual_single = 1 - svm_single.score(X_train_single, y_train)

    elapsed_double = 0
    total_double_iters = 0

    # Adaptive switching based on residual
    if  (max_iter_total - svm_single.n_iter_) > 0:
        remaining_iters = max_iter_total - svm_single.n_iter_
        
        # Double precision refinement
        start_double = time.time()
        svm_double = SVC(C=C, kernel=kernel, gamma='scale', tol=tol_double,
                         max_iter=remaining_iters, random_state=seed)
        svm_double.fit(X_train, y_train)
        elapsed_double = time.time() - start_double
        final_model = svm_double
        total_double_iters = svm_double.n_iter_

    # Evaluate final model on test set
    y_pred = final_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    support_vectors_count = len(final_model.support_)

    total_time = elapsed_single + elapsed_double
    memory_MB = (X_train_single.nbytes + X_train.nbytes) / 1e6

    return (svm_single.n_iter_, total_double_iters, total_time, acc, f1, memory_MB, support_vectors_count)


