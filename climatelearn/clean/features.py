import pandas as pd
import numpy as np


def regression_set(df, target_key, t0, horizon, deltat):
    n_horizon = _calc_horizon(df.index, horizon)
    n_deltat = _calc_horizon(df.index, deltat)
    #Allow the shift also for other keys than the only target key
    for n in range(n_deltat):
        df = _shift_features(df, target_key, n+1)
    df = _shift_features(df, key=target_key, shift=-n_horizon, target=True)
    df = df[df.index >= t0]
    return df


def classification_set(df, target_key, t0, horizon, deltat):
    n_horizon = _calc_horizon(df.index, horizon)
    n_deltat = _calc_horizon(df.index, deltat)
    for n in range(n_deltat):
        df = _shift_features(df, target_key, n+1)
    df = _shift_features(df, key=target_key, shift=-n_horizon, target=True)
    df = el_nino_class(df, target_key + '_tau', nominal=True)
    df = df[df.index >= t0]
    return df


#Todo Add later the moments of the keys
def _add_moments():
    pass

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
    'Hozizon too large, not enough data, exiting'
    exit(1)

def el_nino_class(df, key, width=0.417, threshold=0.5, nominal=False):
    """
    Creates a classification of events based on the duration of an event.

    df: Pandas Dataframe
        The dataframe of the data without classification index

    key: String
        A key of the dataframe according to which the classification is performed

    width: Float
        The width of the window, according to the index of the dataframe, for which
        an event is considered as such

    threshold: Float
        The thresholding value...

    :returns: Pandas Dataframe with a classification key added

    """
    classification_list = np.array([])
    fine = 0
    k = 0
    while fine == 0:
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
                if nominal:
                    classification_list = np.append(classification_list, 'no')
                else:
                    classification_list = np.append(classification_list, 0)
            for i in range(begin, end+1):
                if nominal:
                    classification_list = np.append(classification_list, 'yes')
                else:
                    classification_list = np.append(classification_list, 1)
        else:
            for i in range(k, end+1):
                if nominal:
                    classification_list = np.append(classification_list, 'no')
                else:
                    classification_list = np.append(classification_list, 0)
        k = end + 1
        if end == len(df) -1:
            fine = 1
        if begin > end:
            fine = 1
            for i in range(k, len(df)):
                if nominal:
                    classification_list = np.append(classification_list, 'no')
                else:
                    classification_list = np.append(classification_list, 0)
    df[key + '_class'] = pd.Series(classification_list, index=df.index)
    del df[key]
    return df

