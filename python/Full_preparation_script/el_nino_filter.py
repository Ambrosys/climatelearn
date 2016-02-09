__author__ = 'ruggero'

import matplotlib.pyplot as plt


def filter(t,results,width = 0.4,spacing = 0.05):
    # We find the
    init = []
    fin = []
    flag = 0
    for i in range(0,len(t)):
        if (flag == 0)and(results[i] == 1):
            flag = 1
            init.append(i)
        if (flag == 1)and(results[i] == 0):
            flag = 0
            fin.append(i)
    if len(init) != len(fin):
        fin.append(results[len(t)-1])
    assert len(init) == len(fin)


    res_filt = [0]*len(results)
    for i in range(0,len(init)):
        if t[fin[i]]-t[init[i]] >= width:
            for j in range(init[i],fin[i]+1):
                    res_filt[j] = 1
        elif (i < len(init)-1)and(t[init[i+1]]-t[fin[i]] < spacing):
            t[init[i+1]] = t[init[i]]

    return res_filt