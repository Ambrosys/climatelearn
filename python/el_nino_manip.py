
__author__ = 'Ruggero Vasile'
__date__ = '27/01/2015'


import string
import numpy as np
import el_nino_io as io

def join_data_network(d_net1,d_net2):
    """
    :param d_net1:
    :param d_net2:
    :return:
    """

    new_dic = {}
    keys = key_union(d_net1.keys(),d_net2.keys())
    for k in keys:
        if is_in_list(k,d_net1.keys()):
            new_dic[k] = d_net1[k]
        else:
            new_dic[k] = d_net2[k]
    return new_dic

def join_data_elnino(d_net,d_nino):
    """
    Joins network dictionaries and el nino dictionary. Works if elnino
    dictionary have data less sampled that network and to complete it
    performs linear interpolation between sampling

    :param d_net:
    :param d_nino:
    :return:
    """

    new_dic = {}
    for k in d_net.keys():
        new_dic[k] = np.array([])
    for k in new_dic.keys():
        new_dic[k] = d_net[k]
    new_dic['ElNino'] = np.array([])
    n = 0
    length = len(d_nino['date_time'])
    for d in new_dic['date_time']:
        if d < d_nino['date_time'][0]:
            new_dic['ElNino'] = np.append(new_dic['ElNino'],d_nino['ElNino'][0])
        elif d > d_nino['date_time'][length - 1]:
            new_dic['ElNino'] = np.append(new_dic['ElNino'],d_nino['ElNino'][length - 1])
        else:
            while not(d >= d_nino['date_time'][n])&(d < d_nino['date_time'][n+1]):
                n = n + 1
            a = d_nino['ElNino'][n]
            a += (d_nino['ElNino'][n+1]-d_nino['ElNino'][n])/(d_nino['date_time'][n+1]-d_nino['date_time'][n])*(d-d_nino['date_time'][n])
            new_dic['ElNino'] = np.append(new_dic['ElNino'],a)
    return new_dic

def dic_to_list(dic,order = [],head = True):
    """
    Brings a dictionary into a list form for csv printing. If head = True
    also writes the header
    :param dic:
    :param order: the list is ordered according to the vector...
    :param head:
    :return:
    """
    new_list = []
    header = []
    if order == []:
        for k in dic.keys():
            header.append(k)
    else:
        for k in order:
            header.append(k)
    if head:
        new_list.append(header)

    for i in range(0,len(dic[dic.keys()[0]])):
        lis_part = []
        for k in header:
            lis_part.append(dic[k][i])
        new_list.append(lis_part)
    return new_list

def el_nino_weka_class(dic,classify,t_0,delta_t,tau):
    """
    Prepares weka compatible sets starting from a dic. It adds a classification
    features according to a classify list previously prepared. t_0 determines
    the starting date, delta_t the width of the interval to use for prediction
    and tau the value of the delay of the prediction

    :param dic:
    :param classify:
    :param t_0:
    :param delta_t:
    :param tau:
    :return:
    """
    if t_0 < dic['date_time'][0]:
        t_0 = dic['date_time'][0]
        delta_t = 0
        #print 'delta_t = 0  imposed'
    if t_0-delta_t < dic['date_time'][0]:
        delta_t = t_0 - dic['date_time'][0]
    

    for i in range(0,len(dic['date_time'])):
        if dic['date_time'][i] >= t_0:
            n_init = i
            break
            
    n_tau = -1
    for i in range (n_init,len(dic['date_time'])):
        if dic['date_time'][i] >= dic['date_time'][n_init] + tau:
            n_tau = i
            break
    if n_tau == -1:
        print 'delay too large to build consistent training set. Exiting!'
        exit(1)

    n_delta = -1
    i = n_init 
    while i >= 0:
        if dic['date_time'][n_init] - dic['date_time'][i] >= delta_t:
            n_delta = i
            break
        i = i - 1 
    if n_delta == -1:
        print 'delta_t too large to build set. Exiting!'
        exit(1)


    keys = dic.keys()
    keys.remove('date_time')
    keys.remove('ElNino')

    dic_nn = {}
    dic_nn['t0'] = np.array([])
    dic_nn['t0-deltat'] = np.array([])
    for i in range(0,n_init - n_delta + 1):
        for j in range(0,len(keys)):
            feat = keys[j] + '_' + str(i)
            dic_nn[feat] = np.array([])

    for i in range(0,n_init - n_delta + 1):
        feat = 'ElNino_' + str(i)
        dic_nn[feat] = np.array([])
    dic_nn['Event'] = np.array([])

    
    header = np.array([])
    for k in keys:
        header = np.append(header,k)
    
    
    n_train = len(dic['date_time']) - 1 - n_tau
    assert n_delta >= 0
    for m in range(1,n_train+1):
        dic_nn['t0'] = np.append(dic_nn['t0'],dic['date_time'][n_init + m - 1])
        dic_nn['t0-deltat'] = np.append(dic_nn['t0-deltat'],dic['date_time'][n_delta + m - 1])
        for i in range(0,n_init - n_delta + 1):
            feat = 'ElNino_' + str(i)
            dic_nn[feat] = np.append(dic_nn[feat],dic['ElNino'][n_delta + i + m - 1])
        dic_nn['Event'] = np.append(dic_nn['Event'],classify[n_tau + m - 1])
        for i in range(0,n_init - n_delta + 1):
            for j in range(0,len(keys)):
                feat = keys[j] + '_' + str(i)
                dic_nn[feat] = np.append(dic_nn[feat],dic[keys[j]][n_delta + i + m - 1])

    return dic_nn

def el_nino_weka_regr(dic,t_0,delta_t,tau):
    """
    Prepares weka compatible sets starting from a dic. It adds a classification
    features according to a classify list previously prepared. t_0 determines
    the starting date, delta_t the width of the interval to use for prediction
    and tau the value of the delay of the prediction

    :param dic:
    :param classify:
    :param t_0:
    :param delta_t:
    :param tau:
    :return:
    """
    if t_0 < dic['date_time'][0]:
        t_0 = dic['date_time'][0]
        delta_t = 0
        #print 'delta_t = 0  imposed'
    if t_0-delta_t < dic['date_time'][0]:
        delta_t = t_0 - dic['date_time'][0]


    for i in range(0,len(dic['date_time'])):
        if dic['date_time'][i] >= t_0:
            n_init = i
            break

    n_tau = -1
    for i in range (n_init,len(dic['date_time'])):
        if dic['date_time'][i] >= dic['date_time'][n_init] + tau:
            n_tau = i
            break
    if n_tau == -1:
        print 'delay too large to build consistent training set. Exiting!'
        exit(1)

    n_delta = -1
    i = n_init
    while i >= 0:
        if dic['date_time'][n_init] - dic['date_time'][i] >= delta_t:
            n_delta = i
            break
        i = i - 1
    if n_delta == -1:
        print 'delta_t too large to build set. Exiting!'
        exit(1)


    keys = dic.keys()
    keys.remove('date_time')
    keys.remove('ElNino')

    dic_nn = {}
    dic_nn['t0'] = np.array([])
    dic_nn['t0-deltat'] = np.array([])
    for i in range(0,n_init - n_delta + 1):
        for j in range(0,len(keys)):
            feat = keys[j] + '_' + str(i)
            dic_nn[feat] = np.array([])

    for i in range(0,n_init - n_delta + 1):
        feat = 'ElNino_' + str(i)
        dic_nn[feat] = np.array([])
    dic_nn['ElNino_tau'] = np.array([])


    header = np.array([])
    for k in keys:
        header = np.append(header,k)


    n_train = len(dic['date_time']) - 1 - n_tau
    print n_train,n_tau,n_delta,n_init
    assert n_delta >= 0
    for m in range(1,n_train+1):
        dic_nn['t0'] = np.append(dic_nn['t0'],dic['date_time'][n_init + m - 1])
        dic_nn['t0-deltat'] = np.append(dic_nn['t0-deltat'],dic['date_time'][n_delta + m - 1])
        for i in range(0,n_init - n_delta + 1):
            feat = 'ElNino_' + str(i)
            dic_nn[feat] = np.append(dic_nn[feat],dic['ElNino'][n_delta + i + m - 1])
        dic_nn['ElNino_tau'] = np.append(dic_nn['ElNino_tau'],dic['ElNino'][n_tau + m - 1])
        for i in range(0,n_init - n_delta + 1):
            for j in range(0,len(keys)):
                feat = keys[j] + '_' + str(i)
                dic_nn[feat] = np.append(dic_nn[feat],dic[keys[j]][n_delta + i + m - 1])

    return dic_nn

def classify(dic,width = 0.417,threshold = 0.5, nominal = False):
    """
    Returns a list of classified el nino events from a dictionary. Width
    determines the minimum time length (in year) to catch an event, and
    threshold determines the value above which el nino events are considered as such

    :param dic:
    :param width:
    :param threshold:
    :param nominal:
    :return:
    """
    new_list = np.array([])
    end = 0
    k = 0
    while end == 0:
        for i in range(k,len(dic['date_time'])):
            if dic['ElNino'][i] >= threshold:
                nino_init = i
                break
            nino_init = i
        for i in range(nino_init+1,len(dic['date_time'])):
            if dic['ElNino'][i] < threshold:
                nino_fin = i - 1
                break
            nino_fin = i
        if dic['date_time'][nino_fin] - dic['date_time'][nino_init] > width:
            for i in range(k,nino_init):
                if nominal:
                    new_list = np.append(new_list,'no')
                else:
                    new_list = np.append(new_list,int(0.0))
            for i in range(nino_init,nino_fin+1):
                if nominal:
                    new_list = np.append(new_list,'yes')
                else:
                    new_list = np.append(new_list,int(1.0))
        else:
            for i in range(k,nino_fin + 1):
                if nominal:
                    new_list = np.append(new_list,'no')
                else:
                    new_list = np.append(new_list,int(0.0))
        k = nino_fin + 1
        if nino_fin == len(dic['date_time'])-1:
            end = 1
        if nino_init > nino_fin:
            end = 1
            for i in range(k,len(dic['date_time'])):
                if nominal:
                    new_list = np.append(new_list,'no')
                else:
                    new_list = np.append(new_list,int(0.0))
    return new_list

def training_test_sets(dic, p_total = 100, p_train = 70 ,p_test = 30 , name_train = 'train_set' , name_test = 'test_set', dir = '', pop = [] , typ = 'arff'):
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
    if is_in_list('Event',new_dic.keys()):
        keys = new_dic.keys()
        keys.remove('Event')
        keys.append('Event')
    if is_in_list('ElNino_tau',new_dic.keys()):
        keys = new_dic.keys()
        keys.remove('ElNino_tau')
        keys.append('ElNino_tau')


    p = 0
    # writing the attributes
    attr = []
    for k in keys:
        attr.append([k])
    for i in range(0,len(attr)):
        if attr[i][0] != 'Event':
            attr[i].append('REAL')
        else:
            attr[i].append(['yes','no'])
        if(attr[i][0] == 't0'):
            p = i+1
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

    if typ == 'csv':
        io.csv_file(dic_train,dir,name_train,order=keys)
        io.csv_file(dic_test,dir,name_test,order=keys)
    elif typ == 'arff':
        io.arff_file(dic_train,attr,'ElNino_training',u'',dir,name_train)
        io.arff_file(dic_test,attr,'ElNino_test',u'',dir,name_test)
    elif typ == 'all':
        io.csv_file(dic_train,dir,name_train,order=keys)
        io.csv_file(dic_test,dir,name_test,order=keys)
        io.arff_file(dic_train,attr,'ElNino_training',u'',dir,name_train)
        io.arff_file(dic_test,attr,'ElNino_test',u'',dir,name_test)
    else:
        print 'Not allowed file format. Exiting!'
        exit(1)
    return p

def El_nino_set(joined,classified,file_name = 'events.csv',directory = ''):
    """

    To write el_nino data, [date,NINO34,event(yes/no)] into a file in
    a given directory. Data is taken from a dic with network and el nino
    and from a list showing the particular classification (yes/no) for the
    same dates
    :param joined:
    :param classified:
    :param file_name:
    :param directory:
    :return:
    """

    nino = {}
    nino['date_time'] = joined['date_time']
    nino['Index'] = joined['ElNino']
    nino['Event'] = classified
    nino['keys'] = ['date_time','Index','Event']
    io.csv_file(nino,directory,file_name)
    return 0


#######
# To conclude
def El_nino_date_time(data):
    key_datetime = ""
    for k in data.keys():
        try:
            day = data[k][0].day()
            key_datetime = k
            break
        except:
            continue
    assert key_datetime != ""



    assert is_in_list("date_time",data.keys())
    return None

######## Useful functions

def is_in_list(k,lis):
	for kk in lis:
		if k == kk:
			return True
	return False
    
def key_union(key1,key2):
	union = key1[:]
	for k in key2:
		if not(is_in_list(k,key1)):
			union.append(k)
	return union

