import pandas as pd


class Regression:
    def __init__(self, data, hyper,  n_targets=None, label_targets=None):
        """
        Initialize  the regression model.
        :param X: Training instance nxm matrix with n number of instances and m number of features.
        :param y: Training Target value
        :return:
        """
        self.train_set = data
        self.features = [x for x in self.train_set.keys() if "_tau" not in x]
        self.n_feature = len(self.features)
        if [x for x in self.train_set.keys() if "_tau" in x] != []:
            self.targets = [x for x in self.train_set.keys() if "_tau" in x]
            self.n_target = len(self.targets)
        else:
            self.targets = label_targets
            self.n_target = n_targets

        self.cv = bool(hyper['cross_validation']['flag'])
        if self.cv:
            self.cv_folds = hyper['cross_validation']['folds']

    def learn(self, data = None):
        """
        abstract method to implement in subclass to train  the specific model
        :param params: optional parameters for the specific regression model
        :return:
        """
        raise NotImplementedError('This is only an abstract class.')

    def xvalidate(self,data = None, folds = None):
        """
        abstract method to implement in subclass to train  the specific model
        :param params: optional parameters for the specific regression model
        :return:
        """
        raise NotImplementedError('This is only an abstract class.')

    def test(self, data):
        """
        abstract method to implement in subclass to train  the specific model
        :param params: optional parameters for the specific regression model
        :return:
        """
        raise NotImplementedError('This is only an abstract class.')

    def _build_folds(self, random = False):
        train = []
        validation = []
        if random:
            print 'To do soon...'
        else:
            n = len(self.train_set)/self.cv_folds
            for i in range(self.cv_folds):
                validation.append(self.train_set[i*n:(i+1)*n])
                if i==0:
                    train.append(self.train_set[(i+1)*n:])
                elif i == self.cv_folds - 1:
                    train.append(self.train_set[0:i*n])
                else:
                    x = pd.concat([self.train_set[0:i*n],self.train_set[(i+1)*n:]])
                    train.append(x)
        return train, validation


class Classification:
    def __init__(self, data, hyper, n_targets=None, label_targets=None):
        """
        Initialize  the regression model.
        :param X: Training instance nxm matrix with n number of instances and m number of features.
        :param y: Training Target value
        :return:
        """
        self.train_set = data
        self.features = [x for x in self.train_set.keys() if "_class" not in x]
        self.n_feature = len(self.features)
        if [x for x in self.train_set.keys() if "_class" in x] != []:
            self.targets = [x for x in self.train_set.keys() if "_class" in x]
            self.n_target = len(self.targets)
        else:
            self.targets = label_targets
            self.n_target = n_targets

        self.cv = bool(hyper['cross_validation']['flag'])
        if self.cv:
            self.cv_folds = hyper['cross_validation']['folds']

    def learn(self, data=None):
        """
        abstract method to implement in subclass to train  the specific model
        :param params: optional parameters for the specific regression model
        :return:
        """
        raise NotImplementedError('This is only an abstract class.')

    def xvalidate(self, data=None, folds=None):
        """
        abstract method to implement in subclass to train  the specific model
        :param params: optional parameters for the specific regression model
        :return:
        """
        raise NotImplementedError('This is only an abstract class.')

    def test(self, data):
        """
        abstract method to implement in subclass to train  the specific model
        :param params: optional parameters for the specific regression model
        :return:
        """
        raise NotImplementedError('This is only an abstract class.')


    def _build_folds(self, random=False):
        train = []
        validation = []
        if random:
            print 'To do soon...'
        else:
            n = len(self.train_set) / self.cv_folds
            for i in range(self.cv_folds):
                validation.append(self.train_set[i * n:(i + 1) * n])
                if i == 0:
                    train.append(self.train_set[(i + 1) * n:])
                elif i == self.cv_folds - 1:
                    train.append(self.train_set[0:i * n])
                else:
                    x = pd.concat([self.train_set[0:i * n], self.train_set[(i + 1) * n:]])
                    train.append(x)
        return train, validation
