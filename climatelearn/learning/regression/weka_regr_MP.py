import subprocess
import shlex
import pandas as pd
import numpy as np
from copy import deepcopy
from climatelearn.learning.base_class import Regression
from .. import arf
from .. import errors


class MP_Weka(Regression):
    def __init__(self,train_data, hyper):
        """
    Constructor

    ------------

    train_data: pandas DataFrame
                Contains columns for features and for target variables. The names of the target variables ends
                with the suffix "_tau"
    hyper:      dictionary
                It contains the hyper parameters necessary to run all the functionality of the model.
                They are the following:
                "structure" is a list of integers determining the number of neurons in each hidden layer.
                "epochs" an integer specifying the maximum number of epochs to run during every training session.
                "learning_rate" a float giving the learning rate of the gradient descend.
                "momentum" a float giving the value of the momentum for the algorithm.
                Other parameters regarding cross validation are explained in the base class.

        """
        Regression.__init__(self, train_data, hyper)

        self.structure = self._build_structure(hyper['structure'])

        self.epochs = hyper['epochs']
        self.learning_rate = hyper['learning_rate']
        self.momentum = hyper['momentum']
        self.path_train = hyper['path_train']
        self.path_test = hyper['path_test']
        self.path_model = hyper['path_model']
        self.path_output = hyper['path_output']

    def learn(self, train_data = None):
        """
    Performs single run training.

    ----------

    train_data: pandas Dataframe
            It needs to contain datetime objects on index, and both features and target variables.
            The target variables need to end with the suffix "_tau". If None the self.train_set
            variable passed at the moment of instantiation will be used.

    Returns: string
            It returns the path to the model created.

        """
        if train_data is not None:
            self.train_set = train_data
        self._write_arff(self.train_set, self.path_train)
        command = "java -classpath /usr/share/java/weka.jar weka.classifiers.functions.MultilayerPerceptron  -L " +\
                  str(self.learning_rate) + " -M " + str(self.momentum)
        command += " -N " + str(self.epochs) + " -V 0 -S 0 -E 20 -H " + self.structure + " -t " + self.path_train
        command += " -d " + self.path_model
        args = shlex.split(command)
        p = subprocess.Popen(args, stdout=subprocess.PIPE)
        p.wait()
        predicted, actual = self.test(self.train_set, train=True)
        train_error = 0
        for k in predicted.keys():
            train_error +=errors.RMSE(np.array(actual[k]),np.array(predicted[k]))
        return self.path_model, train_error, None

    def xvalidate(self, train_data = None, folds = None):
        return None

    def test(self, data, train=False):
        data_y = deepcopy(data)
        for k in [x for x in data_y.keys() if '_tau' not in x]:
            del data_y[k]
        if train:
            self._write_arff(data, self.path_train)
            command = "java -classpath /usr/share/java/weka.jar weka.classifiers.functions.MultilayerPerceptron  -l " \
                      + self.path_model + " -T " + self.path_train + " -p 0"
        else:
            self._write_arff(data, self.path_test)
            command = "java -classpath /usr/share/java/weka.jar weka.classifiers.functions.MultilayerPerceptron  -l "\
                  + self.path_model + " -T " + self.path_test + " -p 0"
        args = shlex.split(command)
        p = subprocess.Popen(args, stdout=subprocess.PIPE)
        p.wait()
        fil = open(self.path_output, "w")
        fil.write(p.communicate()[0])
        fil.close()
        predicted = self._read_output(self.path_output)
        predicted = pd.DataFrame(predicted,index=data.index,columns=self.targets)
        return predicted, data_y

    def predict(self, test_data):
        #Todo: Here we add a dummy energy_tau column needed for weka

        test_data["c"] = pd.Series(np.zeros(len(test_data.index)),index=test_data.index)
        self._write_arff(test_data, self.path_test)
        command = "java -classpath /usr/share/java/weka.jar weka.classifiers.functions.MultilayerPerceptron  -l "\
                  + self.path_model + " -T " + self.path_test + " -p 0"
        args = shlex.split(command)
        p = subprocess.Popen(args, stdout=subprocess.PIPE)
        p.wait()
        fil = open(self.path_output, "w")
        fil.write(p.communicate()[0])
        fil.close()
        return self._read_output(self.path_output)

    def _build_structure(self,structure):
        l = " \""
        for i in range(0,len(structure)-1):
            l += str(structure[i]) + ", "
        l += str(structure[-1]) + "\" "
        return l

    def _write_arff(self, data, path):


        attributes = []
        for k in data.keys():
            attributes.append([k,'REAL'])
        new_data = deepcopy(data)
        for k in [c for c in data.keys() if '_tau' in c]:
            new_data = self._exchange_attr(new_data, attributes, k)
        data_write = {'data': new_data[1:], 'attributes': [tuple(l) for l in attributes], 'relation': unicode("__"),
                      'description': unicode("__")}

        data_final = arf.dumps(data_write)
        with open(path, "w") as f:
            f.write(data_final)
        return None

    #Todo: better organize
    def _exchange_attr(self, data, attributes, y):
        new_list = []
        header = []

        for k in data.keys():
            header.append(k)
        header.remove(y)
        header.append(y)

        new_list.append(header)
        for i in range(0, len(data[data.keys()[0]])):
            lis_part = []
            for k in header:
                lis_part.append(data[k][data.index[i]])
            new_list.append(lis_part)

        attributes.remove([y, 'REAL'])
        attributes.append([y, 'REAL'])
        return new_list

    def _read_output(self,path):
        """
        Method for parsing weka regression results
        :param model_dir: directory of the output
        :param out_file: file name of the output
        :return: a dictionary with keys "actual" and "predicted"
        """
        res = []
        with open(path,'r') as fin:
            lines = fin.readlines()
        for i in range(5,len(lines) -1):
            linea = self._splitting(lines[i], ' ')
            res.append(float(linea[2]))
        return np.array(res)

    def _splitting(self, s,spacing = ' '):
        new_s = []
        for s in s.split(spacing):
            if not(s==''):
                new_s.append(s)
        return new_s
