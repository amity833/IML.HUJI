from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    # raise NotImplementedError()

    train_error = []
    validation_error = []
    values_for_fold = int(X.shape[0] / cv)

    for fold in range(cv):
        indices = np.append(np.arange(values_for_fold * fold), np.arange(values_for_fold * (fold + 1), X.shape[0]))
        X_train = np.take(deepcopy(X), indices,axis=0)
        y_train = np.take(deepcopy(y), indices,axis=0)
        fold_indices = np.arange(values_for_fold * fold, values_for_fold * (fold + 1))
        X_validate = np.take(deepcopy(X), fold_indices,axis=0)
        y_validate = np.take(deepcopy(y), fold_indices,axis=0)
        model = estimator.fit(X_train, y_train)
        train_error.append(scoring(model.predict(X_train), y_train))
        validation_error.append(scoring(model.predict(X_validate),y_validate))


    return np.mean(train_error), np.mean(validation_error)
