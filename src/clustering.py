import os
import math
import matplotlib.pyplot as plt
#import tensorly as tl
import numpy as np
from numpy.linalg import pinv
from scipy.misc import face, imresize
#from tensorly.decomposition import parafac
#from tensorly.decomposition import tucker
from math import ceil
from PIL import Image

from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as qda

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


def last_trans_rank(classify_file):
    # find the rank at last transisition
    # classify_file is .npy file storing classification vs. ranks
    # info_file is .npy file storing the info vs. ranks
    # rank is the rank up to which the info is accumulated
    category = np.load(classify_file)
    i = len(category)-1
    while (category[i]==category[i-1] and i>1):
        i -= 1
    return i


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


def plothist(hist1, hist2, adv_title):
    plt.hist(hist1, bins = 60, alpha = 0.5, normed=1)
    plt.hist(hist2, bins = 60, color = 'r', alpha = 0.5, normed=1)
    plt.title('adversarial feature discrimination - ' + adv_title)
    plt.xlabel('number of singular values', fontsize = 12)
    plt.ylabel('training sample density', fontsize = 12)
    plt.show()


def nbtype(fname):
    if (not fname.startswith('.')):
        fnsection = fname.split('_')
        if 'orig' in fnsection:
            return 0
        else:
            return 1
    else:
        # wrong file name
        print('incorrect file type')


def qdaclustering():
    feature0 = './curves/curves_classify_svd2_500' # _svd2.npy feature
    feature1 = './curves/curves_classify' # _svd.npy feature
    X = np.empty([1,2]) # [sample, features], for now there are only two features available - svd2 and svd
    y = np.empty([1])
    testlist_x = []
    testlist_y = []
    testlist2_x = []
    testlist2_y = []
    for fname in os.listdir(feature0):
        if ((not fname.startswith('.')) and not ('_itertarget_' in fname)):# and (('orig' in fname) or ('_momentum_' in fname))):# or ('_steptarget_' in fname) or ('_momentum_' in fname) or ('_fgsm_' in fname))):
            fnsection = fname.split('_')
            sample_y = nbtype(fname)
            sample_x1 = last_trans_rank(os.path.join(feature0, fname))# first feature 'svd2'
            fname_x2 = fname.split('.')[0][:-1] + '.npy'

            if os.path.exists(os.path.join(feature1, fname_x2)):
                sample_x2 = last_trans_rank(os.path.join(feature1, fname_x2))
                # print(X.shape)
                # print(np.array([[sample_x1, sample_x2]]).shape)
                X = np.append(X, np.array([[sample_x1, sample_x2]]), 0)
                # X = np.append(X, np.array([[sample_x1]]), 0)
                y = np.append(y, np.array([sample_y]), 0)

                if sample_y == 0:
                    testlist_x.append(sample_x1)
                    testlist_y.append(sample_x2)
                else:
                    testlist2_x.append(sample_x1)
                    testlist2_y.append(sample_x2)

    X = np.delete(X, 0, 0)
    y = np.delete(y, 0, 0)
    print("dimensions of data:")
    print(X.shape)
    print(y.shape)

    clf = qda()
    clf.fit(X, y)

    pred = clf.predict(X)

    print("Classifier Score:")
    print(clf.score(X,y))
    print("Error:")
    print(sum(abs(pred - y)))
    print("Ordinary images trained:")
    print(len(testlist_x))
    print("")
    print(len(testlist2_x))

    plt.scatter(testlist_x, testlist_y, alpha=0.5)
    plt.scatter(testlist2_x, testlist2_y, color='r', alpha=0.5)
    plt.title('Example - 2D distribution of original and adversarial images')
    plt.xlabel('SVD2')
    plt.ylabel('SVD')
    plt.legend(['orig','adv'], prop={'size': 12})
    plt.show()


if __name__=="__main__":
    qdaclustering()


    # end of code
