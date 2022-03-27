from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

SAMPLE_SIZE = 1000

LIKELIHOOD_RES_SIZE = 200

ROUND_SIZE = 3

pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    # raise NotImplementedError()
    mu, sigma = 10, 1
    samples = np.random.normal(mu, sigma, SAMPLE_SIZE)
    ug = UnivariateGaussian()
    ug.fit(samples)
    print("(" + str(ug.mu_) + ", " + str(ug.var_) + ")")

    # Question 2 - Empirically showing sample mean is consistent
    # raise NotImplementedError()

    sample_sizes = np.arange(10, 1001, 10)
    mu_diff_results = np.zeros(100)
    res_size = 0
    for i in sample_sizes:
        u = UnivariateGaussian()
        u.fit(samples[:i - 1])
        mu_diff_results[res_size] = np.abs(u.mu_ - mu)
        res_size += 1

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sample_sizes, y=mu_diff_results))
    fig.update_layout(title="Expectation Error between estimated and true value", xaxis_title="Sample size",
                      yaxis_title="Expectation difference")
    fig.show()

    # Question 3 - Plotting Empirical PDF of fitted model
    # raise NotImplementedError()

    pdf_results = ug.pdf(samples)
    pdf_fig = go.Figure()
    pdf_fig.add_trace(go.Scatter(x=samples, y=pdf_results, mode="markers"))
    pdf_fig.update_layout(title="PDF values calculated functioned to samples", xaxis_title="Samples",
                          yaxis_title="PDF value")
    pdf_fig.show()

    # print("moodle exam:")
    # res = np.array([1, 5, 2, 3, 8, -4, -2, 5, 1, 10, -10, 4, 5, 2, 7, 1, 1, 3, 2, -1, -3, 1, -4, 1, 2, 1,
    #                 -4, -4, 1, 3, 2, 6, -6, 8, 3, -6, 4, 1, -2, 3, 1, 4, 1, 4, -2, 3, -1, 0, 3, 5, 0, -2])
    # print(ug.log_likelihood(1, 1, res))
    # print(ug.log_likelihood(10, 1, res))


#     My expectation is to see a shape of a bell, since the mean of idd random variables should be a normal variable
#     according to Central Limit Theorem.


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    # raise NotImplementedError()
    mu = np.array([0, 0, 4, 0])
    cov_mat = np.array([[1, 0.2, 0, 0.5], [0.2, 2, 0, 0], [0, 0, 1, 0], [0.5, 0, 0, 1]])
    samples = np.random.multivariate_normal(mu, cov_mat, SAMPLE_SIZE)
    ug = MultivariateGaussian()
    ug.fit(samples)
    print(ug.mu_)
    print(ug.cov_)

    # Question 5 - Likelihood evaluation
    # raise NotImplementedError()
    log_likelihood_res = np.zeros((LIKELIHOOD_RES_SIZE, LIKELIHOOD_RES_SIZE))
    results_range = np.linspace(-10, 10, LIKELIHOOD_RES_SIZE)
    for f1 in range(results_range.size):
        for f3 in range(results_range.size):
            mu2 = np.array([results_range[f1], 0, results_range[f3], 0]).transpose()
            log_likelihood_res[f1, f3] = ug.log_likelihood(mu2, cov_mat, samples)

    fig = go.Figure().add_trace(go.Heatmap(x=results_range, y=results_range, z=log_likelihood_res))
    fig.update_layout(title="Log-likelihood for models with expectation µ = [ f1,0, f3,0]⊤ when f1 anf f3 are numbers\
    in range of (-10 ,10) ", xaxis_title="f1 range", yaxis_title="f3 range")
    fig.show()

    # I'm able to learn from the plot that there is one spot that is most likely to be the correct one, and as we get
    # farther from it the density goes down. The plot is a concave function, meaning is less likely that the true
    # value will be there, just like we learned that the argmax of the log likelihood will be the best value to solve
    # the problem.
    # Also we can see it acts like a normal distribution in several dimensions.

    # Question 6 - Maximum likelihood
    # raise NotImplementedError()
    i, j = np.unravel_index(log_likelihood_res.argmax(), log_likelihood_res.shape)
    print("(", np.round(results_range[i], ROUND_SIZE), ",", np.round(results_range[j], ROUND_SIZE), ")")


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
