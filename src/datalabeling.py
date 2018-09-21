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


def main_trans_rank_volume_generation(feature_str='.npy'):
    # classify_path = './curves/curves_classify_svd2_500/'
    # classify_path = './curves/curves_classify/'
    # classify_path = './curves/curves_classify_svd3/'
    classify_path = './curves/curves_classify_svd132/'
    samples = os.listdir(classify_path)

    rank_list = []
    counter = 0
    for fname in samples:
        if (not fname.startswith('.')) and (feature_str in fname):
            counter += 1
            #print(counter)
            classify_file = os.path.join(classify_path, fname)
            rank = last_trans_rank(classify_file)
            #print(perc)
            rank_list.append(rank)
    return rank_list


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
        if ((not fname.startswith('.')) and (('orig' in fname) or ('_momentum_' in fname))):# or ('_steptarget_' in fname) or ('_momentum_' in fname) or ('_fgsm_' in fname))):
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
    # test_nb()

    #
    # # tensor decomposition
    # orig_rank_list = main_trans_rank_volume_generation('_orig.')
    # fgsm_rank_list = main_trans_rank_volume_generation('_fgsm.')
    # momentum_rank_list = main_trans_rank_volume_generation('_momentum.')
    # jpeg_rank_list = main_trans_rank_volume_generation('_jpeg.')
    # toshi_rank_list = main_trans_rank_volume_generation('_toshi.')
    # toshiSangxia_rank_list = main_trans_rank_volume_generation('_toshiSangxia.')
    #
    # # tensor decomposition over all
    # attacked_rank_list = fgsm_rank_list\
    #  + momentum_rank_list\
    #  + jpeg_rank_list\
    #  + toshi_rank_list\
    #  + toshiSangxia_rank_list
    #
    # # svd
    # orig_rank_svd_list = main_trans_rank_volume_generation('_orig_svd')
    # fgsm_rank_svd_list = main_trans_rank_volume_generation('_fgsm_svd')
    # jpeg_rank_svd_list = main_trans_rank_volume_generation('_jpeg_svd')
    # gd_rank_svd_list = main_trans_rank_volume_generation('_gd_svd')
    # itertarget_rank_svd_list = main_trans_rank_volume_generation('_itertarget_svd')
    # steptarget_rank_svd_list = main_trans_rank_volume_generation('_steptarget_svd')
    # toshi_rank_svd_list = main_trans_rank_volume_generation('_toshi_svd')
    # toshiSangxia_rank_svd_list = main_trans_rank_volume_generation('_toshiSangxia_svd')
    # momentum_rank_svd_list = main_trans_rank_volume_generation('_momentum_svd')
    #
    # # svd over all
    # attacked_rank_svd_list = fgsm_rank_svd_list + jpeg_rank_svd_list + gd_rank_svd_list\
    #  + steptarget_rank_svd_list\
    #  + toshiSangxia_rank_svd_list\
    #  + momentum_rank_svd_list\
    #  + toshi_rank_svd_list
    #  # itertarget_rank_svd_list
    #

    # # svd2
    # orig_rank_svd2_list = main_trans_rank_volume_generation('_orig_svd2')
    # fgsm_rank_svd2_list = main_trans_rank_volume_generation('_fgsm_svd2')
    # jpeg_rank_svd2_list = main_trans_rank_volume_generation('_jpeg_svd2')
    # gd_rank_svd2_list = main_trans_rank_volume_generation('_gd_svd2')
    # itertarget_rank_svd2_list = main_trans_rank_volume_generation('_itertarget_svd2')
    # steptarget_rank_svd2_list = main_trans_rank_volume_generation('_steptarget_svd2')
    # toshi_rank_svd2_list = main_trans_rank_volume_generation('_toshi_svd2')
    # toshiSangxia_rank_svd2_list = main_trans_rank_volume_generation('_toshiSangxia_svd2')
    # momentum_rank_svd2_list = main_trans_rank_volume_generation('_momentum_svd2')
    #
    # # svd2 over all
    # attacked_rank_svd2_list = fgsm_rank_svd2_list + jpeg_rank_svd2_list + gd_rank_svd2_list\
    #  + steptarget_rank_svd2_list\
    #  + toshiSangxia_rank_svd2_list\
    #  + momentum_rank_svd2_list\
    #  + toshi_rank_svd2_list
    #  # itertarget_rank_svd_list
    #
    # # svd3
    # orig_rank_svd3_list = main_trans_rank_volume_generation('_orig_svd3')
    # fgsm_rank_svd3_list = main_trans_rank_volume_generation('_fgsm_svd3')
    # jpeg_rank_svd3_list = main_trans_rank_volume_generation('_jpeg_svd3')
    # gd_rank_svd3_list = main_trans_rank_volume_generation('_gd_svd3')
    # itertarget_rank_svd3_list = main_trans_rank_volume_generation('_itertarget_svd3')
    # steptarget_rank_svd3_list = main_trans_rank_volume_generation('_steptarget_svd3')
    # toshi_rank_svd3_list = main_trans_rank_volume_generation('_toshi_svd3')
    # toshiSangxia_rank_svd3_list = main_trans_rank_volume_generation('_toshiSangxia_svd3')
    # momentum_rank_svd3_list = main_trans_rank_volume_generation('_momentum_svd3')
    #
    # # svd3 over all
    # attacked_rank_svd3_list = fgsm_rank_svd3_list + jpeg_rank_svd3_list + gd_rank_svd3_list\
    #  + steptarget_rank_svd3_list\
    #  + toshiSangxia_rank_svd3_list\
    #  + momentum_rank_svd3_list\
    #  + toshi_rank_svd3_list
    #  # itertarget_rank_svd_list


    # # svd132
    # orig_rank_svd132_list = main_trans_rank_volume_generation('_orig_svd132')
    # fgsm_rank_svd132_list = main_trans_rank_volume_generation('_fgsm_svd132')
    # jpeg_rank_svd132_list = main_trans_rank_volume_generation('_jpeg_svd132')
    # gd_rank_svd132_list = main_trans_rank_volume_generation('_gd_svd132')
    # itertarget_rank_svd132_list = main_trans_rank_volume_generation('_itertarget_svd132')
    # steptarget_rank_svd132_list = main_trans_rank_volume_generation('_steptarget_svd132')
    # toshi_rank_svd132_list = main_trans_rank_volume_generation('_toshi_svd132')
    # toshiSangxia_rank_svd132_list = main_trans_rank_volume_generation('_toshiSangxia_svd132')
    # momentum_rank_svd132_list = main_trans_rank_volume_generation('_momentum_svd132')
    #
    # # svd3 over all
    # attacked_rank_svd132_list = fgsm_rank_svd132_list + jpeg_rank_svd132_list + gd_rank_svd132_list\
    #  + steptarget_rank_svd132_list\
    #  + toshiSangxia_rank_svd132_list\
    #  + momentum_rank_svd132_list\
    #  + toshi_rank_svd132_list
    #  # itertarget_rank_svd132_list




    #
    # #tensor decomposition
    # plothist(orig_rank_list, fgsm_rank_list, 'fgsm')
    # plothist(orig_rank_list, jpeg_rank_list, 'jpeg')
    # plothist(orig_rank_list, toshi_rank_list, 'toshi')
    # plothist(orig_rank_list, momentum_rank_list, 'momentum')
    # plothist(orig_rank_list, toshiSangxia_rank_list, 'toshiSangxia')
    # plothist(orig_rank_list, attacked_rank_list, 'attacked vs orig (tensor)')
    #
    #
    # #svd full depth
    # plothist(orig_rank_svd_list, toshi_rank_svd_list, 'toshi svd')
    # plothist(orig_rank_svd_list, toshiSangxia_rank_svd_list, 'toshiSangxia svd')
    # plothist(orig_rank_svd_list, itertarget_rank_svd_list, 'itertarget svd')
    # plothist(orig_rank_svd_list, steptarget_rank_svd_list, 'steptarget svd')
    # plothist(orig_rank_svd_list, fgsm_rank_svd_list, 'fgsm svd')
    # plothist(orig_rank_svd_list, jpeg_rank_svd_list, 'jpeg svd')
    # plothist(orig_rank_svd_list, gd_rank_svd_list, 'gd svd')
    # plothist(orig_rank_svd_list, momentum_rank_svd_list, 'momentum svd')
    # plothist(orig_rank_svd_list, attacked_rank_svd_list, 'attacked vs orig (svd)')


    # #svd2
    # plothist(orig_rank_svd2_list, toshi_rank_svd2_list, 'toshi svd2')
    # plothist(orig_rank_svd2_list, toshiSangxia_rank_svd2_list, 'toshiSangxia svd2')
    # plothist(orig_rank_svd2_list, itertarget_rank_svd2_list, 'itertarget svd2')
    # plothist(orig_rank_svd2_list, steptarget_rank_svd2_list, 'steptarget svd2')
    # plothist(orig_rank_svd2_list, fgsm_rank_svd2_list, 'fgsm svd2')
    # plothist(orig_rank_svd2_list, jpeg_rank_svd2_list, 'jpeg svd2')
    # plothist(orig_rank_svd2_list, gd_rank_svd2_list, 'gd svd2')
    # plothist(orig_rank_svd2_list, momentum_rank_svd2_list, 'momentum svd2')
    # plothist(orig_rank_svd2_list, attacked_rank_svd2_list, 'attacked vs orig (svd2)')

    # #svd3
    # plothist(orig_rank_svd3_list, toshi_rank_svd3_list, 'toshi svd3')
    # plothist(orig_rank_svd3_list, toshiSangxia_rank_svd3_list, 'toshiSangxia svd3')
    # plothist(orig_rank_svd3_list, itertarget_rank_svd3_list, 'itertarget svd3')
    # plothist(orig_rank_svd3_list, steptarget_rank_svd3_list, 'steptarget svd3')
    # plothist(orig_rank_svd3_list, fgsm_rank_svd3_list, 'fgsm svd3')
    # plothist(orig_rank_svd3_list, jpeg_rank_svd3_list, 'jpeg svd3')
    # plothist(orig_rank_svd3_list, gd_rank_svd3_list, 'gd svd3')
    # plothist(orig_rank_svd3_list, momentum_rank_svd3_list, 'momentum svd3')
    # plothist(orig_rank_svd3_list, attacked_rank_svd3_list, 'attacked vs orig (svd3)')


    # #svd132
    # plothist(orig_rank_svd132_list, toshi_rank_svd132_list, 'toshi svd132')
    # plothist(orig_rank_svd132_list, toshiSangxia_rank_svd132_list, 'toshiSangxia svd132')
    # plothist(orig_rank_svd132_list, itertarget_rank_svd132_list, 'itertarget svd132')
    # plothist(orig_rank_svd132_list, steptarget_rank_svd132_list, 'steptarget svd132')
    # plothist(orig_rank_svd132_list, fgsm_rank_svd132_list, 'fgsm svd132')
    # plothist(orig_rank_svd132_list, jpeg_rank_svd132_list, 'jpeg svd132')
    # plothist(orig_rank_svd132_list, gd_rank_svd132_list, 'gd svd132')
    # plothist(orig_rank_svd132_list, momentum_rank_svd132_list, 'momentum svd132')
    # plothist(orig_rank_svd132_list, attacked_rank_svd132_list, 'attacked vs orig (svd132)')

    # end of code
