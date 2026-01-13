import numpy as np

from svm.smo import smo


def fit(
    train_data: np.ndarray,
):
    alphas = smo(train_data)