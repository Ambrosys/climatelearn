
__author__ = 'ruggero'

import os
import subprocess
import shlex
import numpy as np
import el_nino_manip as manip
import ffx
import matplotlib.pyplot as plt
import el_nino_io as io


def J48(train_set, test_set , CV_folds = 10, C = 0.25, M = 5 ,out_file = "results" , model_file = "tmp.model", model_dir = '', print_feat = 0):
    command = "java -classpath /usr/share/java/weka.jar weka.classifiers.trees.J48  -C " + str(C) + " -M " + str(M) + " -x " + str(CV_folds) + " -t " + train_set + ".arff "
    command += " -d " + model_dir + "tmp.model"
    args = shlex.split(command)
    p1 = subprocess.Popen(args, stdout=subprocess.PIPE)
    output = p1.communicate()[0]
    command2 = "java -classpath /usr/share/java/weka.jar weka.classifiers.trees.J48  -l " + model_dir + "tmp.model -T " + test_set + '.arff -p ' + str(print_feat)
    args2 = shlex.split(command2)
    p2 = subprocess.Popen(args2, stdout=subprocess.PIPE)
    output2 = p2.communicate()[0]
    fil = open(model_dir + out_file + '.arff', "w")
    fil.write(output2)
    fil.close()
    return weka_class_result(model_dir,out_file)

def NN_regression(train_set, test_set , out_file = "results" , model_file = "tmp.model" ,model_dir = '', learn_rate = 0.3 , momentum = 0.2, train_time = 500 , layers = "a", print_feat = 1):
    command = "java -classpath /usr/share/java/weka.jar weka.classifiers.functions.MultilayerPerceptron  -L " + str(learn_rate) + " -M " + str(momentum)
    command += " -N " + str(train_time) + " -V 0 -S 0 -E 20 -H " + layers + " -t " + train_set + ".arff "
    command += " -d " + model_dir + model_file + ".model"
    args = shlex.split(command)
    p1 = subprocess.Popen(args, stdout=subprocess.PIPE)
    output = p1.communicate()[0]
    command2 = "java -classpath /usr/share/java/weka.jar weka.classifiers.functions.MultilayerPerceptron  -l " + model_dir + model_file + ".model -T " + test_set + '.arff -p '  + str(print_feat)
    args2 = shlex.split(command2)
    p2 = subprocess.Popen(args2, stdout=subprocess.PIPE)
    output2 = p2.communicate()[0]
    fil = open(model_dir + out_file + '.arff', "w")
    fil.write(output2)
    fil.close()
    return weka_regr_result(model_dir,out_file)

def NN_classification(train_set, test_set , out_file = "results" , model_file = "tmp_model", model_dir = '' , learn_rate = 0.3 , momentum = 0.2, train_time = 500, layers = "a", print_feat = 1):
    command = "java -classpath /usr/share/java/weka.jar weka.classifiers.functions.MultilayerPerceptron  -L " + str(learn_rate) + " -M " + str(momentum) 
    command += " -N " + str(train_time) + " -V 0 -S 0 -E 20 -H " + layers + " -t " + train_set + ".arff "
    command += " -d " + model_dir + model_file +".model"
    args = shlex.split(command)
    p1 = subprocess.Popen(args, stdout=subprocess.PIPE)
    output = p1.communicate()[0]
    command2 = "java -classpath /usr/share/java/weka.jar weka.classifiers.functions.MultilayerPerceptron  -l " + model_dir + model_file + ".model -T " + test_set + '.arff -p '  + str(print_feat)
    args2 = shlex.split(command2)
    p2 = subprocess.Popen(args2, stdout=subprocess.PIPE)
    output2 = p2.communicate()[0]
    fil = open(model_dir + out_file + '.arff', "w")
    fil.write(output2)
    fil.close()
    return weka_class_result(model_dir,out_file)

def FFX(dic, p_total = 100, p_train = 70 ,p_test = 30 , pop = []):
    """

    :param dic:
    :param p_total:
    :param p_train:
    :param p_test:
    :param pop:
    :return:
    """
    assert p_train + p_test <= 100

    # dividing the domain into train, void and test parts
    length = len(dic[dic.keys()[0]])
    init_train = 0
    fin_train = int(length*float(p_total)/100.0*float(p_train)/100.0)
    init_void = fin_train + 1
    fin_void = int(length*float(p_total)/100.0*float(100.0-p_test)/100.0)
    init_test = fin_void + 1
    fin_test = int(length*float(p_total)/100.0) - 1

    # eliminating some of the features
    new_dic = dic.copy()
    for k in pop:
        new_dic.pop(k, None)

    # Brings event as the last key element (both for regression and classification)
    if manip.is_in_list('ElNino_tau',new_dic.keys()):
        keys = new_dic.keys()
        keys.remove('ElNino_tau')
        keys.append('ElNino_tau')

    dic_train = {}
    dic_test = {}

    for k in new_dic.keys():
        dic_train[k] = np.array([])
        dic_test[k] = np.array([])
    for i in range(init_train,fin_train+1):
        for k in new_dic.keys():
            dic_train[k] = np.append(dic_train[k],new_dic[k][i])
    for i in range(init_test,fin_test+1):
        for k in new_dic.keys():
            dic_test[k] = np.append(dic_test[k],new_dic[k][i])

    keys = sorted(new_dic.keys())
    print keys
    keys.remove('ElNino_tau')
    keys.remove('t0')
    keys.append('ElNino_tau')
    keys.append('t0')


    y_train = dic_train['ElNino_tau']
    x_train = np.zeros(shape=(len(y_train),len(keys)-2))
    for i, t in enumerate(dic_train["t0"]):
            for k, key in enumerate(keys[:-2]):
                x_train[i,k] = dic_train[key][i]


    y_test = dic_test['ElNino_tau']
    x_test = np.zeros(shape=(len(y_test),len(keys)-2))
    for i, t in enumerate(dic_test["t0"]):
            for k, key in enumerate(keys[:-2]):
                x_test[i,k] = dic_test[key][i]


    keys.remove('t0')
    ffx.core.CONSIDER_THRESH = True
    models_ffx = ffx.run(x_train, y_train, x_test, y_test, keys)
    base_fxx = [model.numBases() for model in models_ffx]
    error_fxx = [model.test_nmse for model in models_ffx]
    model = models_ffx[-1]

    new_pred_FFX = np.array([])
    for i in model.simulate(x_test):
        if i >= 0:
            new_pred_FFX = np.append(new_pred_FFX,i)
        else:
            new_pred_FFX = np.append(new_pred_FFX,0.0)

    time = np.array([])
    for i in range(0,len(dic_test['t0'])):
        time = np.append(time,dic_test['t0'][i])

    return time,y_test,new_pred_FFX

def GP(dic, p_total = 100, p_train = 70 ,p_test = 30 , pop = [], working_dir = ""):
    assert p_train + p_test <= 100

    ecj_home = os.path.join(os.getcwd(),'../dist/')
    gp_executable = os.path.join(ecj_home,'gp.jar')
    gp_params     = os.path.join(ecj_home,'gp.params')

    # dividing the domain into train, void and test parts
    length = len(dic[dic.keys()[0]])
    init_train = 0
    fin_train = int(length*float(p_total)/100.0*float(p_train)/100.0)
    init_void = fin_train + 1
    fin_void = int(length*float(p_total)/100.0*float(100.0-p_test)/100.0)
    init_test = fin_void + 1
    fin_test = int(length*float(p_total)/100.0) - 1

    # eliminating some of the features
    new_dic = dic.copy()
    for k in pop:
        new_dic.pop(k, None)

    # Brings event as the last key element (both for regression and classification)
    if manip.is_in_list('ElNino_tau',new_dic.keys()):
        keys = new_dic.keys()
        keys.remove('ElNino_tau')
        keys.append('ElNino_tau')

    dic_train = {}
    dic_test = {}
    keys.remove('t0')

    for k in keys:
        dic_train[k] = np.array([])
        dic_test[k] = np.array([])
    for i in range(init_train,fin_train+1):
        for k in keys:
            dic_train[k] = np.append(dic_train[k],new_dic[k][i])
    for i in range(init_test,fin_test+1):
        for k in keys:
            dic_test[k] = np.append(dic_test[k],new_dic[k][i])

    # Check how to write better the files
    io.gp_file(dic_train,"train_set",order=keys,head=False)
    io.gp_file(dic_test,"test_set",order=keys,head = False)


    command = "java -jar " + gp_executable
    command += " -file " + gp_params
    command += " -Xmx500m -Xmx1024m -p eval.problem.training-file=" + working_dir + "train_set.csv -p"
    command += "eval.problem.testing-file=" + working_dir + "test_set.csv"
    args = shlex.split(command)
    p1 = subprocess.Popen(args, stdout=subprocess.PIPE)


    return None


# To complete
def LR_regression():
    return None

######################### Output options ######################

def weka_class_result(model_dir , out_file):
    results = {}
    results['actual'] = []
    results['predicted'] = []
    with open(model_dir + out_file + '.arff','r') as fin:
        lines = fin.readlines()
    for i in range(5,len(lines) -1):
        linea = splitting(lines[i], ' ')

        if linea[1][2:] == 'no':
            results['actual'].append(0)
        elif linea[1][2:] == 'yes':
            results['actual'].append(1)

        if linea[2][2:] == 'no':
            results['predicted'].append(0)
        elif linea[2][2:] == 'yes':
            results['predicted'].append(1)
    return results

def weka_regr_result(model_dir,out_file):
    results = {}
    results['actual'] = []
    results['predicted'] = []
    with open(model_dir + out_file + '.arff','r') as fin:
        lines = fin.readlines()
    for i in range(5,len(lines) -1):
        linea = splitting(lines[i], ' ')
        results['actual'].append(float(linea[1]))
        results['predicted'].append(float(linea[2]))
    return results


######################  Parsing options ######################

def splitting(stringa,spacing = ' '):
    new_s = []
    for s in stringa.split(spacing):
        if not(s==''):
            new_s.append(s)
    return new_s
