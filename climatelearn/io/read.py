import pandas as pd
import numpy as np
import scipy.stats as stats


def read_csv(path, sep=None, feat_drop=None, date_key=None, dropna=False, drop_axis=None):
    """
    Wraps the pandas.read_csv function adding extra features.

    :path: string
        Path to the csv file to be read.
    :sep: char
        Same argument as pandas.read_csv function.
    :feat_drop: list of strings
        Features to drop
    :date_key: string
        The name of the key representing the date in the dataset.
    :returns: Pandas DataFrame
        The pandas dataframe created from the data.
    """
    df = pd.read_csv(path, sep=sep, index_col=date_key)
    if dropna:
        for d in drop_axis:
            df = df.dropna(axis=d)
    if feat_drop is not None:
        df = df.drop(feat_drop, axis=1)
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


