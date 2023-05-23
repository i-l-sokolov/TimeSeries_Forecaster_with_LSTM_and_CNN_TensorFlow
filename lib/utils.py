import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt


def df_modification(df):
    """
    Modification of primary dataframe with potential useful values
    input - pandas dataframe
    output - pandas dataframe with added columns
    """

    # choosing the columns with dates
    subset = [x for x in df.columns if 'Date' in x]

    # Deleting -0400 string and changing format to datetime
    df[subset] = df[subset].applymap(lambda x: x.replace(' -0400', ''))
    df[subset] = df[subset].apply(lambda x: pd.to_datetime(x), axis=1)

    # Selecting the category labeled as InBed by the device analyser
    df = df.query('value == "HKCategoryValueSleepAnalysisInBed"')

    # Column with sleeping hours as time difference of end and start
    df['sh'] = (df['endDate'] - df['startDate']).dt.total_seconds() / 3600

    # The difference between endDate and startDate in the next row. It could identify regions for sleeping hours combinations
    df['h_next'] = (np.roll(df['startDate'], -1) - df['endDate']).dt.total_seconds() / 3600
    df.loc[df.index[-1], 'h_next'] = 0

    # Label if next row has the same device. It could be useful with combination of short sleeping hours
    df['same_prev_dev'] = np.roll(df['sourceName'], 1) == df['sourceName']

    # Label regions that should be combined
    df['combine'] = False
    df.loc[df.query('sh <= 5 and same_prev_dev == True and h_next < 2 and h_next > 0').index, 'combine'] = True

    return df


def plotting_colors(df):
    """
    Simple function to observe the order of devices in unsorted dataframe.
    It could be useful for understanding if signal from two different devices came to analyser. In that case data should be filtered
    """

    plt.scatter(y=[_ for _ in range(df.shape[0])], x=[1] * df.shape[0],
                c=df['sourceName'].replace(dict(zip(df['sourceName'].unique(), ['yellow', 'black', 'blue']))).values)
    plt.show


def one_hot(df):
    """
    One hot encoding of not Data values
    """

    return pd.get_dummies(df.loc[:, ['Date' not in x for x in df.columns]])


def decompose(df, period):
    """
    Seasonal decomposition of sleeping hours
    """

    res = seasonal_decompose(df['sleep_hours'].values, period=period)
    res.plot()
    plt.show()


def topNcorr(df, N):
    """
    Function return top N correlations with 'sleeping hours' in one-hot dataframe
    """
    df_onehot = one_hot(df)
    df_corr = (df.join(df_onehot)).corr()
    return df_corr['sleep_hours'].sort_values(ascending=False, key=abs)[1:N + 1]


def cleaning_data(df):
    df['sleep_hours'] = df['sleep_hours'].apply(lambda x: x / 2 if x > 10 else x)
    q25, q75 = np.percentile(df['sleep_hours'],25), np.percentile(df['sleep_hours'],75)
    iqr = q75 - q25
    lower, upper = q25 - 1.5 * iqr, q75 + 1.5 * iqr
    df['outlier'] = df['sleep_hours'].apply(lambda x: x < lower or x > upper)
    mean, std = df.query('outlier == False')['sleep_hours'].describe()['mean'], df.query('outlier == True')['sleep_hours'].describe()['std']
    df.loc[df.query('outlier == True').index,'sleep_hours'] = np.random.normal(loc=mean,scale=std,size=df['outlier'].sum())
    df.drop('outlier',axis=1)
    return df

def cond_plot(prediction_generation, df_data, df_submission, window, out_vals, conv, shuffle, name):
    class_preds = prediction_generation(df_data, df_submission, window=window, out_vals=out_vals, conv=conv, shuffle=shuffle)
    class_preds.training(name)
    class_preds.prediction(name)
    class_preds.sub = class_preds.sub.assign(window = window, out_vals = out_vals, conv = conv, shuffle=shuffle)
    return class_preds.sub