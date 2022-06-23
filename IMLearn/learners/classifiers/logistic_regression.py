from typing import NoReturn

import numpy as np

from IMLearn import BaseEstimator
from IMLearn.desent_methods import GradientDescent
from IMLearn.desent_methods.modules import LogisticModule, RegularizedModule, L1, L2
from IMLearn.metrics import misclassification_error
from IMLearn.desent_methods import GradientDescent, FixedLR

class LogisticRegression(BaseEstimator):

    def __init__(self,
                 include_intercept: bool = True,
                 solver: GradientDescent = GradientDescent(learning_rate=FixedLR(1e-4), max_iter=20000),
                 penalty: str = "none",
                 lam: float = 1,
                 alpha: float = .5):
        """
        Initialize a ridge regression model
        :param lam: scalar value of regularization parameter
        """
        super().__init__()
        self.include_intercept_ = include_intercept
        self.solver_ = solver
        self.lam_ = lam
        self.penalty_ = penalty
        self.alpha_ = alpha

        if penalty not in ["none", "l1", "l2"]:
            raise ValueError("Supported penalty types are: none, l1, l2")

        self.coefs_ = None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit Logistic regression model to given samples

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Notes
        -----
        Fits model using specified `self.optimizer_` passed when instantiating class and includes an intercept
        if specified by `self.include_intercept_
        """
        # raise NotImplementedError()
        if self.include_intercept_:
            X = np.insert(X, 0, 1, axis=1)
        self.coefs_ = np.random.normal(0, 1 / X.shape[1], X.shape[1])
        logistic_module = LogisticModule(self.coefs_)
        if self.penalty_ == 'none':
            model = logistic_module
        else:
            if self.penalty_ == 'l1':
                model = RegularizedModule(fidelity_module=logistic_module, lam=self.lam_,
                                          regularization_module=L1(self.coefs_), weights=self.coefs_)
            elif self.penalty_ == 'l2':
                model = RegularizedModule(fidelity_module=logistic_module, lam=self.lam_,
                                          regularization_module=L2(self.coefs_), weights=self.coefs_)

        self.coefs_ = self.solver_.fit(model, X=X, y=y)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        # raise NotImplementedError()
        return np.where(self.predict_proba(X) > self.alpha_, 1, 0)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities of samples being classified as `1` according to sigmoid(Xw)

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict probability for

        Returns
        -------
        probabilities: ndarray of shape (n_samples,)
            Probability of each sample being classified as `1` according to the fitted model
        """
        # raise NotImplementedError()
        if self.include_intercept_:
            X = np.insert(X, 0, 1, axis=1)
        a = X @ self.coefs_
        return np.exp(a) / (np.exp(a) + 1)

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification error

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under misclassification error
        """
        # raise NotImplementedError()
        return misclassification_error(y, self.predict(X))
