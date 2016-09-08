import pandas as pd
import numpy as np


def regression_set(df, target_key, initial_time, horizon, deltat=0):
    n_horizon = _calc_horizon(df.index, horizon)
    n_deltat = _calc_horizon(df.index, deltat)

    for n in range(n_deltat):
        df = _shift_features(df, target_key, n+1)
    df = _shift_features(df, key=target_key, shift=-n_horizon, target=True)
    df = df[df.index >= initial_time]

    y = np.array(df[target_key + '_tau'])
    X = np.asmatrix(df.drop(target_key + '_tau', axis=1))
    return X, y


def classification_set(df, target_key, t0, horizon, deltat, nominal=True):
    n_horizon = _calc_horizon(df.index, horizon)
    n_deltat = _calc_horizon(df.index, deltat)
    for n in range(n_deltat):
        df = _shift_features(df, target_key, n+1)
    df = _shift_features(df, key=target_key, shift=-n_horizon, target=True)
    df = classification_feature(df, target_key + '_tau', nominal=nominal)
    df = df[df.index >= t0]

    y = np.array(df[target_key + '_tau'])
    X = np.asmatrix(df.drop(target_key + '_tau', axis=1))
    return X, y


def _shift_features(df, key, shift, target=False):
    if target:
        df[key + '_tau'] = df[key].shift(shift)
    else:
        df[key + '_' + str(shift)] = df[key].shift(shift)
    df = df.dropna(axis=0)
    return df


def _calc_horizon(index, horizon):
    first = index[0]
    for i in range(len(index)):
        if index[i] >= first + horizon:
            return i


def classification_feature(df, key, width=0.417, threshold=0.5, nominal=False, drop_key=True):
    """
    Creates a classification of events based on the behavior of one of the features.

    df: Pandas Dataframe
        The dataframe of the data without classification index

    key: String
        A key of the dataframe according to which the classification is performed

    width: Float
        The width of the window, according to the index of the dataframe, for which
        an event is considered as such

    threshold: Float
        The thresholding value above which the event is considered as such

    :returns: Pandas Dataframe with a classification key added

    """
    classification_list = np.array([])
    k = 0
    while True:
        for i in range(k, len(df)):
            if df[key][df.index[i]] >= threshold:
                begin = i
                break
            begin = i
        for i in range(begin+1, len(df)):
            if df[key][df.index[i]] < threshold:
                end = i-1
                break
            end = i
        if df.index[end] - df.index[begin] > width:
            for i in range(k, begin):
                classification_list = np.append(classification_list, 'no' if nominal else int(0))
            for i in range(begin, end+1):
                classification_list = np.append(classification_list, 'yes' if nominal else int(1))
        else:
            for i in range(k, end+1):
                classification_list = np.append(classification_list, 'no' if nominal else int(0))
        k = end + 1
        if end == len(df) -1:
            break
        if begin > end:
            for i in range(k, len(df)):
                classification_list = np.append(classification_list, 'no' if nominal else int(0))
            break
    df[key + '_class'] = pd.Series(classification_list, index=df.index)
    if drop_key:
        del df[key]
    return df