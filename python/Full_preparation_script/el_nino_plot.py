__author__ = 'ruggero'

import matplotlib.pyplot as plt

def plot_J48(dic ,tau = 0):
    plt.ylim(-.1, 1.2)
    plt.plot(dic['date_time'],dic['actual'],'r--',dic['date_time'],dic['predicted'],'b')
    plt.show()
    return None
    
def plot_NN(dic ,tau = 0):
    plt.ylim(-.1, 1.2)
    plt.plot(dic['date_time'],dic['actual'],'r--',dic['date_time'],dic['predicted'],'b')
    plt.show()
    return None
