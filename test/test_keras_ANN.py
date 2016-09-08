import unittest
from pickle import loads, dumps
import numpy as np
import logging
from climatelearn.learning.regression.keras_ANN import KerasRegressionModel


class TestRead(unittest.TestCase):
    def test_Keras_RegressionModel(self):
        x = np.linspace(0, 1, 100)
        y = 2 * x
        logger = logging.getLogger()
        logging.basicConfig(level=logging.INFO, format='%(message)s')
        logger.info("start")
        model = KerasRegressionModel(arity=1, network_structure=(5, 1)).fit(x, y)

        weights = model.weights
        m_restored = loads(dumps(model))

    # check if all weights are equal
        same_weights = all([np.array_equal(i, j) for i, j in zip(model.weights, m_restored.weights)])
        assert same_weights

