import numpy as np


def RMSE(actual, prediction):
    """
    :param actual:
    :type actual: np.ndarray, pd.Series
    :param prediction:
    :type prediction: np.ndarray, pd.Series
    :returns: Root mean squared error.
    """
    return np.sqrt(np.mean((actual - prediction) ** 2))


def MAE(actual, prediction):
    """
    :param actual:
    :type actual: np.ndarray, pd.Series
    :param prediction:
    :type prediction: np.ndarray, pd.Series
    :returns: Mean absolute error.
    """
    return np.mean(np.abs(actual - prediction))


def MBE(actual, prediction):
    """
    :param actual:
    :type actual: np.ndarray, pd.Series
    :param prediction:
    :type prediction: np.ndarray, pd.Series
    :returns: Mean biased error.
    """
    return np.mean(actual - prediction)


def normalize_metric(error_func):
    """
    Decorator to normalize a given error metrix by range (max - min) of the data

    :param error_func: one error metric
    :type erroc_func: MBE, MAE, RMSE
    :returns: normalized error metric
    """
    def inner(actual, prediction):
        min_ = min(np.min(actual), np.min(prediction))
        max_ = max(np.max(actual), np.max(prediction))
        range_ = max_ - min_
        if range_:
            norm = 1. / range_
        else:
            norm = 0
        return norm * error_func(actual, prediction)
    return inner


NRMSE = normalize_metric(RMSE)
NMAE = normalize_metric(MAE)
MBE = normalize_metric(MBE)


def confusion_matrix(predicted, actual, key=None):
    if key is None:
        key = predicted.keys()[0]
    matrix = np.zeros((2,2))
    for i in range(len(predicted)):
        if predicted[key][predicted.index[i]] == 'no' and actual[key][actual.index[i]] == 'no':
            matrix[0][0] += 1
        elif predicted[key][predicted.index[i]] == 'no' and actual[key][actual.index[i]] == 'yes':
            matrix[0][1] += 1
        elif predicted[key][predicted.index[i]] == 'yes' and actual[key][actual.index[i]] == 'no':
            matrix[1][0] += 1
        else:
            matrix[1][1] += 1
    return matrix