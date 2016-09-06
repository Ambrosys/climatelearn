import climatelearn.io.read as read
import climatelearn.clean.features as features
from climatelearn.learning.classify import classification_train as train
from climatelearn.learning.errors import confusion_matrix
import pandas as pd


path = "../data/DATA_BIU.txt"
Net_BIU = read.read_csv(path, sep="\t", date_key='date_time')

path = "../data/NINO34_BIU.txt"
nino_data = read.read_csv(path=path, sep='\t', date_key='date_time')

nino_data = nino_data.set_index(Net_BIU.index)
raw_data = pd.concat([nino_data, Net_BIU], axis=1)


data = features.classification_set(raw_data, target_key='NINO34', t0=1950.0, horizon=1.0, deltat=0.0)

print data

exit()
train_set = data[data.index < 1960]
test_set = data[data.index >= 1960]

params_weka = {
    "epochs": 10,
    "structure": [6, 6, 6],
    "batch": 0,
    "momentum": 0.05,
    "learning_rate": 0.1,
    "seed": 2,
    "train_fraction": 0.9,
    "cross_validation": {
        "flag": 1,
        "folds": 3
    },
    "path_train" : "train.arff",
    "path_test": "test.arff",
    "path_model": "temp.model",
    "path_output": "output.csv"
}

model = train('MPWEKA', params_weka, train_set)

print confusion_matrix(model[0](train_set)[0], model[0](train_set)[1])
print confusion_matrix(model[0](test_set)[0], model[0](test_set)[1])

