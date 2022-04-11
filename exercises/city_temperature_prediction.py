import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio

MIN_TEMP = -30
MAX_TEMP = 55

pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    df = pd.read_csv(filename, parse_dates=['Date'], date_parser=pd.to_datetime)
    filter_data = df.dropna().drop_duplicates()
    filter_data = filter_data[(MIN_TEMP <= filter_data['Temp']) & (filter_data['Temp'] <= MAX_TEMP)]
    filter_data.insert(filter_data.shape[1], 'DayOfYear', np.zeros(filter_data.shape[0]))
    filter_data['DayOfYear'] = filter_data['Date'].dt.dayofyear

    filter_data.dropna(inplace=True)
    return filter_data


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    data = load_data('../datasets/City_Temperature.csv')

    # Question 2 - Exploring data for specific country
    only_israel = data[data['Country'] == 'Israel']
    response = only_israel['Temp']
    fig = px.scatter(only_israel, x=only_israel['DayOfYear'], y=response,
                     title="Temp change as function of Day of the year", color=only_israel['Year'].astype(str))
    fig.show()
    group_by_month = only_israel.groupby('Month').agg('Temp')

    std_values = (group_by_month.std())
    std_fig = px.bar(group_by_month, x=only_israel['Month'].unique(), y=std_values,
                     title="Standard deviation of the daily temperatures ",
                     labels={'x': 'Month', 'y': 'standard deviation'})
    std_fig.show()

    # Question 3 - Exploring differences between countries
    full_data_group_month_country = data.groupby(['Country', 'Month']).agg({'Temp': ['mean', 'std']})
    full_data_group_month_country.columns = ['Mean Temperature', 'Standard deviation']
    avg_std_values = full_data_group_month_country.reset_index()
    avg_std_fig = px.line(avg_std_values, x='Month', y='Mean Temperature',
                          title="Average monthly temperature, with error bars ", error_y='Standard deviation',
                          color='Country')
    avg_std_fig.show()

    # Question 4 - Fitting model for different values of `k`
    train_X, train_y, test_X, test_y = split_train_test(only_israel, response)
    loss_values = np.zeros(10)
    for k in range(1, 11):
        polynomial_model = PolynomialFitting(k)
        polynomial_model.fit(train_X['DayOfYear'].to_numpy(), train_y.to_numpy())
        loss_values[k - 1] = round(polynomial_model.loss(test_X['DayOfYear'].to_numpy(), test_y.to_numpy()), 2)
        print('Error for k = ', k, 'is ', loss_values[k - 1])
    error_poly_fig = px.bar(loss_values, x=range(1, 11), y=loss_values,
                            title="Test error for each degree value ",
                            labels={'x': 'Degree', 'y': 'Error'})
    error_poly_fig.show()

    # Question 5 - Evaluating fitted model on different countries

    polynomial_model_based_q4 = PolynomialFitting(5)
    polynomial_model_based_q4.fit(only_israel['DayOfYear'], only_israel['Temp'])
    country_loss = []
    countries = data['Country'].unique()
    countries = countries[countries != 'Israel']
    for country in countries:
        country_data = data[data['Country'] == country]
        country_loss.append(polynomial_model_based_q4.loss(country_data['DayOfYear'], country_data['Temp']))
    countries_error = px.bar(country_loss, x=countries, y=country_loss,
                             title="Test error for each country ",
                             labels={'x': 'Country', 'y': 'Error'})
    countries_error.show()

    print("finished execution")
