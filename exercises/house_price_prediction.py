from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    df = pd.read_csv(filename)
    df = df.dropna().drop_duplicates()
    df = df[df['yr_renovated'] >= df['yr_built']]

    filtered_data = df.drop(['sqft_above', 'sqft_basement', 'yr_built', 'zipcode', 'lat', 'long',
                             'sqft_living15', 'sqft_lot15', 'date'], axis=1)
    filtered_data = filtered_data[filtered_data['price'] > 0]
    df['price'] = df['price'].apply(lambda x: float(x) / 100)
    filtered_data = filtered_data[filtered_data['bedrooms'] > 0]
    filtered_data = filtered_data[filtered_data['bathrooms'] > 0]
    filtered_data = filtered_data[
        (10 <= filtered_data['sqft_living']) & (filtered_data['sqft_living'] <= filtered_data['sqft_lot'])]
    filtered_data = filtered_data[filtered_data['floors'] > 0]
    filtered_data = filtered_data[(filtered_data['waterfront'] == 0) | (filtered_data['waterfront'] == 1)]
    price_vector = filtered_data['price']

    filtered_data.drop('price', inplace=True, axis=1)

    return filtered_data, price_vector


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    pearson_correlation_results = np.zeros((X.shape[1]))
    index = 0
    for feature in X.columns:
        p = (np.cov(y, X[feature])) / (np.std(X[feature]) * np.std(y))
        pearson_correlation_results[index] = p[0][1]
        index += 1
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=X[feature], y=y, mode="markers"))
        fig.update_layout(title=feature + "\n\t pearson correlation: " + str(pearson_correlation_results[index - 1]),
                          xaxis_title="feature values",
                          yaxis_title="response vector")
        fig.write_image(output_path + "/" + feature + ".jpeg")
    # print('exam Q1 : ',str((np.cov(y, X['grade']) / (np.std(X['grade']) * np.std(y)))[0][1]))


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    data, series = load_data('../datasets/house_prices.csv')

    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(data, series, "G:/My Drive/year 2/IML/ex2")

    # Question 3 - Split samples into training- and testing sets.
    train_X, train_y, test_X, test_y = split_train_test(data, series)

    # Question 4 - Fit model over increasing percentages of the overall training data
    linear_model = LinearRegression()
    mean_values = np.zeros(91)
    std_values = np.zeros(91)
    frames = []
    index = 0
    for p in range(10, 101):
        loss = np.zeros(10)
        for i in range(10):
            curr_train_X, curr_train_y, curr_test_X, curr_test_y = split_train_test(train_X, train_y, p / 100.0)
            linear_model.fit(curr_train_X.to_numpy(), curr_train_y.to_numpy())
            loss[i] = linear_model.loss(test_X.to_numpy(),
                                        test_y.to_numpy())
        mean_values[index] = np.mean(loss)
        std_values[index] = np.std(loss)
        index += 1
    fig = go.Figure([
        go.Scatter(
            name='Mean value',
            x=np.arange(10, 101),
            y=mean_values,
            mode='markers+lines',
            line=dict(color='rgb(31, 119, 180)'),
        ),
        go.Scatter(
            name='Upper Bound',
            x=np.arange(10, 101),
            y=mean_values + 2 * std_values,
            mode='lines',
            marker=dict(color="#444"),
            line=dict(width=0),
            showlegend=False
        ),
        go.Scatter(
            name='Lower Bound',
            x=np.arange(10, 101),
            y=mean_values - 2 * std_values,
            marker=dict(color="#444"),
            line=dict(width=0),
            mode='lines',
            fillcolor='rgba(68, 68, 68, 0.3)',
            fill='tonexty',
            showlegend=False
        )
    ])
    fig.update_layout(
        yaxis_title='Mean values',
        title='Mean loss as function of % of samples',
        hovermode="x"
    )
    fig.show()
    print("finished execution")
