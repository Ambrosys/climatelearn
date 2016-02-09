# Here is an example of simulation to apply el nino algorithms


__author__ = 'markus'
__change__ = "ruggero"

import os
import el_nino_manip as manip
import el_nino_weka as weka 
import el_nino_plot as plot
import el_nino_io as io
import el_nino_filter as filter
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    home=os.getenv("HOME")
    cwd=os.getcwd()
    #Setting home directory for reading and writing
    root_Dir_read  = os.path.join(cwd, '../../data/')
    root_Dir_write = os.path.join(cwd, '../../data/')
    print "reading data from ",root_Dir_read
    print "writing data to ",root_Dir_write

    #Reading network and el Nino data
    data_Pres = io.read_Net_partial(root_Dir_read+"/seaLevelPressure_Network/")
    data_Temp = io.read_Net_partial(root_Dir_read+"/seaLevelTemp_Network/")
    elNino = io.read_ElNino(root_Dir_read + 'NINO34.dat')

    exit(0)
    #Setting up which data we want to work with (one or both networks)
    data = manip.join_data_network(data_Temp,data_Pres)


    #join network and elnino data
    joined = manip.join_data_elnino(data,elNino)
    #define a classification on el nino data and save into a list
    WID = 0.4
    classified = manip.classify(joined,width = WID,threshold = 0.5,nominal=True)


    # create a weka instance-friendly file with given parameters
    t0 = 1946.0
    deltat = 0.0
    tau = 1.0
    nn = manip.el_nino_weka_class(joined,classified,t0,deltat,tau)

    # writing the full data file
    #keys = nn.keys()
    #keys.remove('Event')
    #keys.append('Event')
    #io.csv_file(nn,root_Dir_write,'full_data',order=keys)
    
    # Putting in the list the features we do not want to involve in the classification
    pop = np.array(['t0-deltat','t0'])


    name_train = 'train_temp'
    name_test = 'test_temp'
    train_set = root_Dir_write + name_train
    test_set = root_Dir_write + name_test
    p = manip.training_test_sets(nn, 100, 70 , 30 , name_train, name_test , root_Dir_write, pop = pop , typ = 'arff')

    #result = weka.NN_basic_classification(train_set,test_set,print_feat = p)
    result = weka.J48_basic(train_set,test_set, C = 0.5, M = 50, print_feat = p)

    t = nn['t0'][-len(result['predicted'])-1:-1]
    filtered = filter.filter(t,result['predicted'],width = WID, spacing = 0.05)
    plt.plot(t,filtered,'b',t,result['actual'],'r--')
    plt.ylim(-.1, 1.2)
    plt.show()


        
 

 


