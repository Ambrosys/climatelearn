import unittest
import climatelearn.clean.features as features
import climatelearn.io.read as read


class TestFeatures(unittest.TestCase):
    def test(self):
        path = "../data/NINO3.txt"
        names = ['date', 'NINO3']
        data = read.read_csv(path, sep="\t", names=names)
        self.assertEqual(len(data.keys()), 1)

        shift = 10
        key = 'NINO3'
        data = features._shift_features(data, key, shift)
        for i in range(len(data)-shift):
            self.assertEqual(data[key][data.index[i]], data[key + '_' + str(shift)][data.index[i+shift]])

        #self.assertEqual(len(data.keys()), 3)
        #self.index = data.index


