import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IMLearn.metrics.loss_functions import accuracy

pio.renderers.default = "browser"


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    adaboost = AdaBoost(wl=lambda: DecisionStump(), iterations=n_learners)
    adaboost.fit(train_X, train_y)
    train_loss = []
    test_loss = []
    for i in range(1, n_learners + 1):
        train_loss.append(adaboost.partial_loss(train_X, train_y, i))
        test_loss.append(adaboost.partial_loss(test_X, test_y, i))
    fig1 = go.Figure()
    fig1.add_traces([go.Scatter(x=list(range(1, n_learners + 1)), y=train_loss, name='Train error'),
                     go.Scatter(x=list(range(1, n_learners + 1)), y=test_loss, name='Test error')])
    fig1.update_layout(title=f"Training and test errors as a function of the number of fitted learners, noise ={noise} ",
                       xaxis_title="fitted learners",
                       yaxis_title="test errors")
    fig1.show()


    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])
    iterations = ['5 iterations', '50 iterations', '100 iterations', '250 iterations']
    if not noise :
        # raise NotImplementedError()
        fig2 = make_subplots(rows=2, cols=2, subplot_titles=[rf"$\textbf{{{t}}}$" for t in iterations],
                             horizontal_spacing=0.01, vertical_spacing=.03)
        for i, t in enumerate(T):
            adaboost_pred = lambda X: adaboost.partial_predict(X, t)
            fig2.add_traces([decision_surface(adaboost_pred, lims[0], lims[1], showscale=False),
                             go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                                        marker=dict(color=test_y, colorscale=[custom[0], custom[-1]],
                                                    line=dict(color="black", width=1)))],
                            rows=(i // 2) + 1, cols=(i % 2) + 1)


        fig2.update_layout(title=rf"$\textbf{{ Decision Boundaries Of different iterations}}$", margin=dict(t=100)) \
            .update_xaxes(visible=False).update_yaxes(visible=False)
        fig2.show()

        # Question 3: Decision surface of best performing ensemble
        min_err_ind = np.argmin(test_loss) + 1
        adaboost_pred = lambda X: adaboost.partial_predict(X, min_err_ind)
        fig3 = go.Figure()
        fig3.add_traces([decision_surface(adaboost_pred, lims[0], lims[1], showscale=False),
                         go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                                    marker=dict(color=test_y, colorscale=[custom[0], custom[-1]],
                                                line=dict(color="black", width=1)))])
        fig3.update_layout(
            title=rf"$\textbf{{The ensable that achieved the lowest test error, size {{{min_err_ind}}}, accuracy = {{{accuracy(test_y[:min_err_ind], adaboost.partial_predict(test_X, min_err_ind))}}}}} $",
            margin=dict(t=100)).update_xaxes(visible=False).update_yaxes(visible=False)
        fig3.show()

    # Question 4: Decision surface with weighted samples
    # raise NotImplementedError()
    fig4 = go.Figure()
    fig4.add_traces([decision_surface(adaboost.predict, lims[0], lims[1], showscale=False),
                     go.Scatter(x=train_X[:, 0], y=train_X[:, 1], mode='markers', showlegend=False,
                                marker=dict(color=train_y, colorscale=[custom[0], custom[-1]],
                                            size= adaboost.D_/np.max(adaboost.D_)*5),
                                line=dict(color="black", width=1))])

    fig4.update_layout(
        title=f"The training set with a point size proportional to it’s weight over the ensamble with 250 iterations, noise = {noise}",
        margin=dict(t=100)).update_xaxes(visible=False).update_yaxes(visible=False)
    fig4.show()


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(0)
    fit_and_evaluate_adaboost(0.4)
    # raise NotImplementedError()
