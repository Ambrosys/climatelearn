import climatelearn.io.read as read
from climatelearn.preprocess.generating_targets import regression_set
from climatelearn.learning.regression import keras_ANN
import pandas as pd
import matplotlib.pyplot as plt

path = "../data/NINO3.txt"
NINO = read.read_csv(path, sep="\t", date_key='date_time')

path = "../data/ST_windburst.txt"
STwind = read.read_csv(path=path, sep='\t', date_key='date_time')

raw_data = pd.concat([NINO, STwind], axis=1).dropna(axis=0)

X, y = regression_set(raw_data, target_key='NINO3', initial_time=1969, horizon=1)

model = keras_ANN.KerasRegressionModel(arity=3, network_structure=(5, 1), batch_size=1, nb_epoch=1000)
model.fit(X, y)
yhat = model.predict(X)

plt.plot(range(len(yhat)), yhat, range(len(y)), y)
plt.show()

