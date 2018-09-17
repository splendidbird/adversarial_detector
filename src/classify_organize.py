import os
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import numpy as np


def save_curve(imgfile, rankend, outputpath, savepath):
    # save classresult curve vs. Tucker ranks

    classresult = []

    imgfile = '1_attack.png'
    rankend = 299
    outputpath = '../output/'
    savepath = './curves/'

    for rank in range(1, rankend + 1):
        fname = imgfile.split('.')[0]+'_'+str(rank)+'.txt'
        with open(os.path.join(outputpath, fname), 'r') as f:
            classresult.append(int(f.read().split()[0]))
        os.remove(os.path.join(outputpath, fname))

    savename = imgfile.split('.')[0]
    np.save(os.path.join(savepath, savename), classresult)

    # plt.plot(classresult, "b*", markersize=7)
    # plt.show()
