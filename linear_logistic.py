import sys
from aoclda.linear_model import linmod
import numpy as np

def linmod_example():
    """
    Linear regression with an intercept variable
    """
    X = np.array([[1, 1], [2, 3], [3, 5], [4, 8], [5, 7], [6, 9]], dtype=np.float32)
    y = np.array([3., 6.5, 10., 12., 13., 19.], dtype=np.float32)
    lmod = linmod("logistic", solver="lbfgs", max_iter=10, precision='single')
    lmod.fit(X, y , tol=1e-6)

    # Extract coefficients
    coef = lmod.coef
    print(f"coefficients: [{coef[0]:.3f}, {coef[1]:.3f}, {coef[2]:.3f}]")
    print('expected    : [2.350, 0.350, 0.433]\n')

    # # Evaluate model on new data
    # X_test = np.array([[1, 1.1], [2.5, 3], [7, 9]], dtype=np.float32)
    # pred = lmod.predict(X_test)
    # print(f"predictions: [{pred[0]:.3f}, {pred[1]:.3f}, {pred[2]:.3f}]")
    # print('expected   : [3.168  7.358 20.0333]')


if __name__ == "__main__":
    try:
        linmod_example()
    except RuntimeError:
        sys.exit(1)