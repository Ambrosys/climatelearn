import pandas as pd
import numpy as np
import scipy.stats as stats


def read_csv(path, names=None, sep=None, date_key=None, index=None):
    """
    Wraps the pandas.read_csv function adding extra features.

    :path: string
        Path to the csv file to be read.
    :names: list of strings
        Same argument as pandas.read_csv function.
    :sep: char
        Same argument as pandas.read_csv function.
    :date_key: string
        The name of the key representing the date in the dataset.
    :returns: Pandas DataFrame
        The pandas dataframe created from the data.
    """
    df = pd.read_csv(path, sep=sep, names=names)
    if date_key is not None:
        df['date'] = df[date_key]
        del df[date_key]
    if 'date' in df:
        df = df.set_index('date',drop=True)
    df = df.dropna(axis=1).dropna(axis=0)
    return df


def read_network(path, names=['mean', 'var', 'skew', 'kurt'], index=None):
    """
    A specific reader
    """
    data = np.loadtxt(path)
    mean = data.mean(axis=1)
    variance = data.var(axis=1)
    skew = stats.skew(data, axis=1)
    kustosis = stats.kurtosis(data, axis=1)
    df = pd.DataFrame({names[0] : mean, names[1]: variance, names[2]: skew, names[3]: kustosis}, index=index)
    return df


