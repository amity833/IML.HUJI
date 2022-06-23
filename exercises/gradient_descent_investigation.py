import numpy as np
import pandas as pd
from typing import Tuple, List, Callable, Type

from IMLearn import BaseModule
from IMLearn.desent_methods import GradientDescent, FixedLR, ExponentialLR
from IMLearn.desent_methods.modules import L1, L2
from IMLearn.learners.classifiers.logistic_regression import LogisticRegression
from IMLearn.metrics import misclassification_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate

import plotly.graph_objects as go

import plotly.io as pio

pio.renderers.default = "browser"


def plot_descent_path(module: Type[BaseModule],
                      descent_path: np.ndarray,
                      title: str = "",
                      xrange=(-1.5, 1.5),
                      yrange=(-1.5, 1.5)) -> go.Figure:
    """
    Plot the descent path of the gradient descent algorithm

    Parameters:
    -----------
    module: Type[BaseModule]
        Module type for which descent path is plotted

    descent_path: np.ndarray of shape (n_iterations, 2)
        Set of locations if 2D parameter space being the regularization path

    title: str, default=""
        Setting details to add to plot title

    xrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    yrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    Return:
    -------
    fig: go.Figure
        Plotly figure showing module's value in a grid of [xrange]x[yrange] over which regularization path is shown

    Example:
    --------
    fig = plot_descent_path(IMLearn.desent_methods.modules.L1, np.ndarray([[1,1],[0,0]]))
    fig.show()
    """

    def predict_(w):
        return np.array([module(weights=wi).compute_output() for wi in w])

    from utils import decision_surface
    return go.Figure([decision_surface(predict_, xrange=xrange, yrange=yrange, density=70, showscale=False),
                      go.Scatter(x=descent_path[:, 0], y=descent_path[:, 1], mode="markers+lines",
                                 marker_color="black")],
                     layout=go.Layout(xaxis=dict(range=xrange),
                                      yaxis=dict(range=yrange),
                                      title=f"GD Descent Path {title}"))


def get_gd_state_recorder_callback() -> Tuple[Callable[[], None], List[np.ndarray], List[np.ndarray]]:
    """
    Callback generator for the GradientDescent class, recording the objective's value and parameters at each iteration

    Return:
    -------
    callback: Callable[[], None]
        Callback function to be passed to the GradientDescent class, recoding the objective's value and parameters
        at each iteration of the algorithm

    values: List[np.ndarray]
        Recorded objective values

    weights: List[np.ndarray]
        Recorded parameters
    """

    values = []
    weights = []

    def help_func(x):
        values.append(x[1])
        weights.append(x[0])

    return help_func, values, weights


def compare_fixed_learning_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                 etas: Tuple[float] = (1, .1, .01, .001)):
    models = [L1, L2]
    for eta in etas:
        for mod in models:
            callback = get_gd_state_recorder_callback()
            gd = GradientDescent(FixedLR(eta), callback=callback[0])
            weight = gd.fit(mod(np.copy(init)), np.zeros(0), np.zeros(0))
            fig = plot_descent_path(module=mod, descent_path=np.array(callback[2]),
                                    title="The model " + str(mod) + " with eta= " + str(eta))
            fig.show()
            fig1 = go.Figure([go.Scatter(x=np.arange(gd.max_iter_), y=callback[1], mode="markers",
                                         marker=dict(color="blue", opacity=.7))])
            fig1.update_layout(
                title_text=rf"$\text{{The norm of model {mod} with eta = {eta} as a function of the GD iterations }}$",
                xaxis={"title": r"GD iterations"},
                yaxis={"title": r"Norm values", })

            fig1.show()


def compare_exponential_decay_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                    eta: float = .1,
                                    gammas: Tuple[float] = (.9, .95, .99, 1)):
    graphs = []
    for gamma in gammas:
        callback = get_gd_state_recorder_callback()
        gd = GradientDescent(ExponentialLR(eta, decay_rate=gamma), callback=callback[0])
        gd.fit(L1(np.copy(init)), np.zeros(0), np.zeros(0))
        graphs.append(go.Scatter(x=np.arange(gd.max_iter_), y=callback[1], mode="markers", name=f'gamma= {gamma}'))

    # Optimize the L1 objective using different decay-rate values of the exponentially decaying learning rate
    fig = go.Figure(graphs)
    fig.update_layout(
        title_text=rf"$\text{{The convergence rate for different decay rates of L1}}$",
        xaxis={"title": r"GD iterations"},
        yaxis={"title": r"decay rates", })

    fig.show()

    # Plot algorithm's convergence for the different values of gamma

    # Plot descent path for gamma=0.95
    callback = get_gd_state_recorder_callback()
    l1_mod = L1(np.copy(init))
    gd = GradientDescent(ExponentialLR(eta, decay_rate=0.95), callback=callback[0])
    gd.fit(l1_mod, np.zeros(0), np.zeros(0))
    fig2 = plot_descent_path(module=L1, descent_path=np.array(callback[2]),
                             title="L1 model " + " with eta= " + str(eta))
    fig2.show()


def load_data(path: str = "../datasets/SAheart.data", train_portion: float = .8) -> \
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load South-Africa Heart Disease dataset and randomly split into a train- and test portion

    Parameters:
    -----------
    path: str, default= "../datasets/SAheart.data"
        Path to dataset

    train_portion: float, default=0.8
        Portion of dataset to use as a training set

    Return:
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples
    """
    df = pd.read_csv(path)
    df.famhist = (df.famhist == 'Present').astype(int)
    return split_train_test(df.drop(['chd', 'row.names'], axis=1), df.chd, train_portion)


def fit_logistic_regression():
    # Load and split SA Heard Disease dataset
    X_train, y_train, X_test, y_test = load_data()

    # Plotting convergence rate of logistic regression over SA heart disease data
    from sklearn.metrics import roc_curve, auc
    gd = GradientDescent(FixedLR(1e-4), max_iter=20000)
    model = LogisticRegression(solver=gd)
    model.fit(np.array(X_train), np.array(y_train))
    y_prob = model.predict_proba(np.array(X_train))
    fpr, tpr, thresholds = roc_curve(y_train, y_prob)

    fig = go.Figure(
        data=[go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="black", dash='dash'),
                         name="Random Class Assignment"),
              go.Scatter(x=fpr, y=tpr, mode='markers+lines', text=thresholds, name="", showlegend=False, marker_size=5,
                         # marker_color=c[1][1],
                         hovertemplate="<b>Threshold:</b>%{text:.3f}<br>FPR: %{x:.3f}<br>TPR: %{y:.3f}")],
        layout=go.Layout(title=rf"$\text{{ROC Curve Of Fitted Model - AUC}}={auc(fpr, tpr):.6f}$",
                         xaxis=dict(title=r"$\text{False Positive Rate (FPR)}$"),
                         yaxis=dict(title=r"$\text{True Positive Rate (TPR)}$")))
    fig.show()
    best_alpha = thresholds[np.argmax(tpr - fpr)]
    print("Best alpha is ", best_alpha)

    model_with_alpha = LogisticRegression(solver=gd, alpha=best_alpha)
    model_with_alpha.fit(np.array(X_train), np.array(y_train))
    print("Model's test error: ", model_with_alpha.loss(np.array(X_test), np.array(y_test)))

    # Fitting l1- and l2-regularized logistic regression models, using cross-validation to specify values
    # of regularization parameter

    lambdas = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
    for mod in ['l1', 'l2']:
        train_error, validation_error = [], []
        for lam in lambdas:
            model = LogisticRegression(alpha=0.5, penalty=mod, lam=lam)
            t_e, v_e = cross_validate(model, np.array(X_train), np.array(y_train), misclassification_error)
            train_error.append(t_e)
            validation_error.append(v_e)
        print(validation_error)
        best_lam = lambdas[np.argmin(np.array(validation_error))]
        best_model = LogisticRegression(alpha=0.5, penalty=mod, lam=best_lam)
        best_model.fit(np.array(X_train), np.array(y_train))
        print(mod, " regularized  model test error: ", best_model.loss(np.array(X_test), np.array(y_test)),
              " with lambda = ", best_lam)

    print("finished")


if __name__ == '__main__':
    np.random.seed(0)
    # compare_fixed_learning_rates()
    # compare_exponential_decay_rates()
    fit_logistic_regression()
