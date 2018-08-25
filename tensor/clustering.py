import os
import math
import matplotlib.pyplot as plt
import tensorly as tl
import numpy as np
from numpy.linalg import pinv
from scipy.misc import face, imresize
from tensorly.decomposition import parafac
from tensorly.decomposition import tucker
from math import ceil
from PIL import Image

random_state = 12345


def last_trans_percent(classify_file, info_file):
    # locate the percentage of image info with up to last transisition rank
    # classify_file is .npy file storing classification vs. ranks
    # info_file is .npy file storing the info vs. ranks
    # rank is the rank up to which the info is accumulated
    category = np.load(classify_file)
    curve = np.load(info_file)
    if len(curve)!=len(category):
        raise Exception("category and info curves dimension mismatch")
        return -1
    i = len(category)-1
    while (category[i]==category[i-1] and i>1):
        i -= 1
    return curve[i], i


def tucker_core(imgarray, tucker_rank):
    """
    imgarray is 3-d numpy array of png image
    """
    imgtensor = tl.tensor(imgarray)
    # Tucker decomposition
    core, tucker_factors =tucker(imgtensor, ranks=tucker_rank, init='random', tol=10e-5, random_state=random_state)
    return core, tucker_factors


def info_curve_generation(imgfile, rankend, savepath):
    # generate percentage of info vs. rank
    imgobj = Image.open(imgfile)
    imgarray = np.array(imgobj)
    core_rank = [rankend, rankend, 1]

    core, tucker_factors = tucker_core(imgarray, core_rank)

    diag = []
    for i in range(rankend):
        diag.append(core[i,i,0]**2)

    savename = imgfile.split('/')[-1].split('.')[0] + '_infocurve.npy'
    info_list = [sum(diag[:i])/sum(diag) for i in range(1, len(diag)+1)]
    np.save(os.path.join(savepath, savename), info_list)
    return info_list


def main_info_curve_volume_generation():
    rankend = 299
    sample_path = './samples/'
    inputpath = '../input/'
    outputpath = '../output/'
    savepath = './curves/curves_info/'

    samples = os.listdir(sample_path)
    for fname in samples:
        print("Procesing %s" % fname)
        curve = info_curve_generation(os.path.join(sample_path, fname), rankend, savepath)
        plt.plot(curve)
        os.remove(os.path.join(sample_path, fname))
    plt.show()


def main_trans_percent_volume_generation():
    sample_path = './samples/'
    classify_path = './curves/curves_classify/'
    info_path = './curves/curves_info/'
    samples = os.listdir(sample_path)

    perc_list = []
    rank_list = []
    counter = 0
    for fname in samples:
        counter+=1
        #print(counter)
        classify_file = os.path.join(classify_path, fname.split('.')[0]+".npy")
        info_file = os.path.join(info_path, fname.split('.')[0]+"_infocurve.npy")
        perc, rank = last_trans_percent(classify_file, info_file)
        #print(perc)
        perc_list.append(perc)
        rank_list.append(rank)
    return perc_list, rank_list


if __name__=="__main__":
    # this section is for
    # main_info_curve_volume_generation()

    perc_list, rank_list = main_trans_percent_volume_generation()
    plt.plot(rank_list,'*')
    plt.show()

    plt.hist(rank_list, bins = 299, range=(0, 299))
    plt.show()

    plt.plot(np.log(np.ones(len(perc_list))-perc_list),'*')
    plt.show()

    plt.hist(np.log(np.ones(len(perc_list))-perc_list), bins = 100, range=(-10, 0))
    plt.show()





    # end of code
