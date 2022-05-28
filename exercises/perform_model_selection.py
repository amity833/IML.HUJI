from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots

pio.renderers.default = "browser"


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions

    # raise NotImplementedError()
    response = lambda x: (x + 3) * (x + 2) * (x + 1) * (x - 1) * (x - 2)
    x = np.linspace(-1.2, 2, n_samples)
    y_ = response(x)
    y = y_ + np.random.normal(0, noise, n_samples)
    train_X, train_y, test_X, test_y = split_train_test(pd.DataFrame(x), pd.Series(y), 2 / 3)
    fig1 = go.Figure([go.Scatter(x=x, y=y_, mode="markers+lines", name="true (noiseless) model",
                                 marker=dict(color="green", opacity=.7)),
                      go.Scatter(x=list(train_X[0]), y=list(train_y), mode="markers", name="train set",
                                 marker=dict(color="red", opacity=.7)),
                      go.Scatter(x=list(test_X[0]), y=list(test_y), mode="markers", name="test set",
                                 marker=dict(color="blue", opacity=.7))])
    fig1.update_layout(
        title_text=rf"$\text{{Polynom samples, and samples with noise =  {noise}}}$",
        xaxis={"title": r"$samples$"},
        yaxis={"title": r"$values$", })

    fig1.show()

    # todo add axis names to graphs
    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    # raise NotImplementedError()

    train_error, validation_error = [], []
    X_train_update = np.array(train_X).reshape(-1)
    for k in range(11):
        model_k = PolynomialFitting(k)
        t_e, v_e = cross_validate(model_k, X_train_update, np.array(train_y),
                                  mean_square_error)  # reshape(train_X.shape[0])
        train_error.append(t_e)
        validation_error.append(v_e)
        fig1.add_trace(go.Scatter(x=x, y=list(model_k.predict(x)), mode="markers+lines", name=f"modle fitting{k}",
                                  marker=dict(colorscale='Electric', opacity=.7)))
    fig1.show()  # todo delete??

    fig2 = go.Figure([go.Scatter(x=np.arange(11), y=train_error, mode="markers+lines", name="train errors",
                                 marker=dict(color="red", opacity=.7)),
                      go.Scatter(x=np.arange(11), y=validation_error, mode="markers+lines", name="validation errors",
                                 marker=dict(color="blue", opacity=.7))
                      ])
    fig2.update_layout(
        title_text=rf"$\text{{The average train and validation error for each degree of the polynomial with noise =  {noise} }}$",
        xaxis={"title": r"Polynomial degree values"},
        yaxis={"title": r"Average error", })
    fig2.show()

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    # raise NotImplementedError()

    min_k = np.argmin(validation_error)
    poly_modle = PolynomialFitting(min_k)
    poly_modle.fit(np.array(train_X).reshape(-1), np.array(train_y))
    test_err = np.round(poly_modle.loss(np.array(test_X).reshape(-1), np.array(test_y)), 2)
    min_valid_error = np.round(validation_error[min_k ],2)
    print("Best fitted k (polinomial degree) = " ,min_k, ", noise = ", noise, ",\n Validation error = ", min_valid_error, ", Test error = ", test_err, "\n")
    # todo how to show the similarity / insimilarity between validation and test error


def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    raise NotImplementedError()

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    raise NotImplementedError()

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    select_polynomial_degree()
    select_polynomial_degree(noise=0)
    select_polynomial_degree(n_samples=1500, noise=10)
    # raise NotImplementedError()
