import numpy as np
from copy import copy
from copy import deepcopy
import pandas as pd


from pybrain.structure import FeedForwardNetwork
from pybrain.structure import LinearLayer, SigmoidLayer
from pybrain.structure import FullConnection
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from climatelearn.learning.base_class import Regression
from .. import errors


class MP_Pybrain(Regression):
    """
    Fully connected multilayer perceptron using pybrain library.
    """
    def __init__(self, train_data, hyper,  n_targets=None, label_targets=None):
        """
    ------------

    train_data: pandas DataFrame
                Contains columns for features and for target variables. The names of the target variables ends
                with the suffix "_tau"
    hyper:      dictionary
                It contains the hyperparameters necessary to run all the functionalities of the model.
                 They are the following:
                "structure" is a list of integers determining the number of neurons in each hidden layer
                "epochs" an integer specifying the maximum number of epochs to run during every training session
                "learning_rate" a float giving the learning rate of the gradient descend
                "momentum" a float giving the value of the momentum for the algorithm
                "batch" a bool. If True the method performs full batch learning, i.e. updates of the weights is done
                using all the instances of the training set. Else, normal online method is performed
                Other parameters regarding cross validation are explained in the base class

        """
        Regression.__init__(self, train_data, hyper, n_targets=n_targets, label_targets=label_targets)

        self.N = FeedForwardNetwork()
        self.structure = [self.n_feature] + hyper['structure'] + [self.n_target]

        self._build_net(self.structure)
        self.res_params = [self.N.params[i] for i in range(len(self.N.params))]

        self.train_fraction = hyper['train_fraction']
        self.seed = hyper['seed']
        self.epochs = hyper['epochs']
        self.learning_rate = hyper['learning_rate']
        self.momentum = hyper['momentum']
        self.batch = bool(hyper['batch'])

    def learn(self, train_data = None, seed = None):
        """
    Performs single run training, and it is designed to be called after network instantiation.

    ----------

    train_data: pandas Dataframe
            It needs to contain datetime objects on index, and both features and target variables.
            The target variables need to end with the suffix "_tau". If None the self.train_set
            variable passed at the moment of instantiation will be used.

    Returns: tuple(MP_Pybrain object,float)
            It returns the model with the lowest training error, and the value of the training error.

        """
        if train_data is not None:
            self.train_set = train_data
            self.randomize()
        ds_train, ds_valid = self._build_dataset(self.train_set)
        trainer = BackpropTrainer(self.N, ds_train, learningrate=self.learning_rate,
                                  momentum=self.momentum,batchlearning=self.batch)
        trainer.train()
        e_train = [self._error(ds_train)]
        e_valid = [self._error(ds_valid)]
        final_model = copy(self)
        fin_error_train = e_train[0]
        fin_error_valid = e_valid[0]
        for i in range(1,self.epochs):
            if i%10 == 0:
                print "epoch: ", i
            trainer.train()
            e_train.append(self._error(ds_train))
            e_valid.append(self._error(ds_valid))
            if e_train[-1] < fin_error_train:
                final_model = deepcopy(self)
                fin_error_train = e_train[-1]
                fin_error_valid = e_valid[-1]
        return final_model, fin_error_train, fin_error_valid

    def xvalidate(self, train_data = None, folds = None):
        """
    Performs n-folds cross-validation on the a data set. The method is designed to reset the network
    to an initial configuration (decided at the moment of instantiation) every time a new training is
    started. The purpose is to make model comparison and returning an average error given a specific
    data set and collection of hyper-parameters. At the moment training and validation sets are chosen
    based on the input sequence of data, i.e. there is no random shuffling of the instances of the data set.

    ----------

    train_data: pandas Dataframe
            It needs to contain datetime objects on index, and both features and target variables.
            The target variables need to end with the suffix "_tau". If None the self.train_set
            variable passed at the moment of instantiation will be used.

    folds: integer
            The number of training/validation partition used in the method. If None it needs to be
            passed in the constructor when instantiating the object for the first time. If not passed
            ever, the method cannot work and an exception needs to be thrown.
    Returns: list, float, float
            A list of all the models trained for each fold, the mean train error and the cross-validation error,
            i.e. the average of NRMSE for all the training/validation partitions created.

        """
        if train_data is not None:
            self.train_set = train_data
        if folds is not None:
            self.cv_folds = folds
        train, validation = self._build_folds(random=False)
        models = []
        train_error = []
        cv_error = []
        for i in range(self.cv_folds):
            print "Cross-validation Fold: ", i+1
            self.randomize()
            model, error, _ = self.learn(train_data=train[i])
            models.append(deepcopy(model))
            train_error.append(error)
            predicted, actual = self.test(validation[i])
            e = 0
            for k in predicted.keys():
                e += errors.RMSE(np.array(actual[k]),np.array(predicted[k]))
            cv_error.append(e)
        return models, np.mean(train_error), np.mean(cv_error)

    def test(self, data):
        """
    Tests the trained model on data. The usage is two fold: 1) Internal usage to calculate errors on validation
    sets. 2) For external usage when a test set is provided. Both the validation and test set need to contain target
    columns. For prediction, where target variables are unknown, please refer to the function self.predict below.
    ----------

    data:       pandas Dataframe
                A pandas dataframe. A deepcopy of it will be made and only the feature columns will be considered.
                Due to the functionality of the pyBrain library we require (at the moment) that the order of the
                colums is the same as the one of the training set used for training.

    Returns:    pandas Dataframe
                A Dataframe with columns containing the predictions of the different target variables and same index as
                the input DataFrame

        """
        data_x = data[self.features]
        data_y = data[self.targets]
        predicted = np.array([])
        for i in range(len(data_x)):
            predicted = np.append(predicted, self.N.activate(data_x.values[i]))
        return pd.DataFrame(predicted, index=data.index, columns=self.targets), data_y

    def predict(self, data):
        """
    It returns target variables given a set of features, using the model trained and saved.
    ---------

    data: pandas Dataframe
         It must contain all the feature columns used for training of the model

    Returns: pandas Dataframe
         It contains the prediction on the target variables. The name of the variables is the same as the one
         provided at the moment of instantiation of object.

        """
        data_x = data[self.features]
        predicted = np.array([])
        for i in range(len(data_x)):
            predicted = np.append(predicted, self.N.activate(data_x.values[i]))
        return pd.DataFrame(predicted, index=data_x.index, columns=self.targets)

    def randomize(self):
        self.N.randomize()
        pass


    ### Private functions ###
    def _error(self, ds):
        """
    Calculates the RMSE over an input dataset, given the current state of the network.

    ds: Supervised dataset pybrain style

    Returns: float
        The total error between prediction and actual values.

        """
        predicted = np.array([list(self.N.activate(x)) for x in ds['input']]).transpose()
        actual = np.array([list(x) for x in ds['target']]).transpose()
        total_error = [errors.RMSE(np.array(actual[i]),np.array(predicted[i])) for i in range(len(actual))]
        return sum(total_error)

    def _build_net(self,s):
        layers = [LinearLayer(s[0])]
        self.N.addInputModule(layers[0])
        for i in range(1,len(s)-1):
            layers.append(SigmoidLayer(s[i]))
            self.N.addModule(layers[i])
        layers.append(SigmoidLayer(s[-1]))
        self.N.addOutputModule(layers[-1])
        self._build_connections(layers)

    def _build_connections(self, l):
        for i,j in zip(l,l[1:]):
            a = FullConnection(i,j)
            self.N.addConnection(a)
        self.N.sortModules()

    def _build_dataset(self, data):
        """
    Given a input training Dataframe with features and targets it returns the formatted training and validation
    datasets for pybrain usage, and randomly shuffled according to the self.seed given at instantiation.

    ----------

    data: pandas Dataframe
        It must contains both features and target columns

    Returns: (pybrain dataset, pybrain dataset)
        The first is the training dataset and the second is the validation dataset

        """
        np.random.seed(self.seed)
        permutation = np.random.permutation(np.arange(len(data)))
        sep = int(self.train_fraction * len(data))
        x = data[self.features]
        y = data[self.targets]
        ds_train = SupervisedDataSet(self.n_feature, self.n_target)
        ds_valid = SupervisedDataSet(self.n_feature, self.n_target)
        for i in permutation[:sep]:
            ds_train.addSample(x.values[i], y.values[i])
        for i in permutation[sep:]:
            ds_valid.addSample(x.values[i], y.values[i])
        return ds_train, ds_valid

