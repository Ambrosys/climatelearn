import unittest
import climatelearn.io.read as read


class TestRead(unittest.TestCase):
    def test(self):
        path = "../data/NINO3.txt"
        names = ['date', 'NINO3']
        data = read.read_csv(path, sep="\t", names=names)
        self.assertEqual(len(data.keys()), 1)
        self.index = data.index
