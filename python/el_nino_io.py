__author__ = 'ruggero'
__author__ = "markus"



import numpy as np
import os
import string
import csv
import scipy.io
import datetime
import time
import el_nino_manip as manip
import scipy.stats as stats

# we need to use version 2.0 of arff (for now the file is in our directory)
import arf


####################### READING ###########################


# Works for now only if all the files have the same data_time entries.
def read_Net_partial(inputDir,exten = '.dat'):
    if os.path.isdir(inputDir) == True:
        dic = {}
        dic['date_time'] = np.array([])
        file_num = 0
        for f in os.listdir(inputDir) :			
            extension = os.path.splitext(f)[1]
            inFile = inputDir+f
            if extension == exten:
                file_num += 1
                feat_name = f[0:len(f)-len(extension)]
                dic[feat_name] = np.array([])
                with open(inFile, 'r') as csvfile:
                    reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
                    ele = -1
                    for row in reader:
                        if len(row) != 0:
                            dt =(float)(row[0].split("\t")[0])
                            try:
							    value =(float)(row[0].split("\t")[1])
                            except: # should no happen
                                v = float('nan')
                            if file_num == 1:
                                dic['date_time'] = np.append(dic['date_time'],dt)
                                dic[feat_name] = np.append(dic[feat_name],value)
                            else:
                                if manip.is_in_list(dt,dic['date_time']):
                                    dic[feat_name] = np.append(dic[feat_name],value)
                                else: 
                                    dic['date_time'] = np.append(dic['date_time'],dt)
                                    dic[feat_name] = np.append(dic[feat_name],value)

        return dic
    else:
        print "Wrong input directory provided. Exiting!"
        exit(1)
        return 0

def read_ElNino(file_name):
    dic = {}
    dic['date_time'] = np.array([])
    dic['ElNino'] = np.array([])
    data = csv.reader(open(file_name),delimiter=' ', quotechar='|')
    for row in data:
        n = 0
        for j in row:
            if n == 0:
                dic['date_time'] = np.append(dic['date_time'],float(j))
                n = 1
            else:
                dic['ElNino'] = np.append(dic['ElNino'],float(j))
    return dic


def read_Qing_Alexis(file_net,file_nino):
    data = np.loadtxt(file_net)
    meanDegree = data.mean(1)
    varianceDegree = data.var(1)
    skewDegree = stats.skew(data,1)
    kurtDegree = stats.kurtosis(data,1)
    data_fin = {}
    data_fin['Mean'] = meanDegree
    data_fin['Var'] = varianceDegree
    data_fin['Skew'] = skewDegree
    data_fin['Kurtosis'] = kurtDegree
    data = np.loadtxt(file_nino)
    nino = data[:,1]
    time = data[:,0]
    data_fin["ElNino"] = nino
    data_fin["date_time"] = time
    return data_fin

########################### WRITING ###################################


def arff_file(data,attributes,relation,description,output_dir="./",filename="tmp"):

    x = []
    for k in attributes:
        x.append(k[0])
    data_write = {}
    data_write['data'] = manip.dic_to_list(data,order=x)[1:]
    data_write['attributes'] = [tuple(l) for l in attributes]
    data_write['relation'] = unicode(relation)
    data_write['description'] = unicode(description)
    data_final = arf.dumps(data_write)
    #print data_final
    fil = open(output_dir + filename + '.arff', "w")
    fil.write(data_final)
    fil.close()

    return None

def csv_file(data,output_dir,filename,order = [],head = True):
    with open(output_dir + filename + '.csv', 'w') as f:
            write = csv.writer(f)
            write.writerows(manip.dic_to_list(data,order,head),)
    return None


def gp_file(data,filename,output_dir='',order = [],head = True):

    with open(output_dir + filename + '.csv', 'w') as f:
            write = csv.writer(f)
            write.writerows(manip.dic_to_list(data,order,head),)
    return None

