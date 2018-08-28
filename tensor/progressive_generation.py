"""
Generate progressive images based on an input standard image
==========================================
"""
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


def to_image(tensor):
    """A convenience function to convert from a float dtype back to uint8"""
    im = tl.to_numpy(tensor)
    im -= im.min()
    im /= im.max()
    im *= 255
    return im.astype(np.uint8)


def progressive_generation(imgfile, rankend, result_path):
    # max effective rankend is min(imgarray.shape[0],imgarray.shape[1])
    target = imgfile
    imgobj = Image.open(target)
    imgarray = np.array(imgobj)
    imgtensor = tl.tensor(imgarray)

    for rank in range(1, rankend+1):
        fullfname = os.path.join(result_path, imgfile.split('/')[-1].split('.')[0]+'_'+str(rank)+'.png')
        core_rank = [rank, rank, 1]
        core, tucker_factors = tucker(imgtensor, ranks=core_rank, init='random', tol=10e-5, random_state=random_state)
        tucker_reconstruction = tl.tucker_to_tensor(core, tucker_factors)
        Image.fromarray(to_image(tucker_reconstruction)).save(fullfname)


def svdRGBimg(imgarray, n_svd):
    # return an imageobj of the
    # imgarray - 299 x 299 x 3 RBG image array
    # n_svd - number of svd reserved, value between 0 and total number of svd
    (la,lb,lc) = imgarray.shape
    stopper = n_svd
    reducedimgarray = np.empty((la, lb, 3))
    for rgb in range(lc):
        U, S, Vt = np.linalg.svd(imgarray[:,:,rgb])
        reducedimgarray[:,:,rgb] = np.array(np.matrix(U[:, :stopper]) * np.diag(S[:stopper])
            * np.matrix(Vt[:stopper, :]))
    return reducedimgarray


def progressive_generation_svd(imgfile, rankend, result_path):
    # max effective rankend is min(imgarray.shape[0],imgarray.shape[1])
    target = imgfile
    imgobj = Image.open(target)
    imgarray = np.array(imgobj)
    for rank in range(1, rankend+1):
        fullfname = os.path.join(result_path, imgfile.split('/')[-1].split('.')[0]+'_svd_'+str(rank)+'.png')
        reducedimgarray = svdRGBimg(imgarray, rank)
        Image.fromarray(to_image(reducedimgarray)).save(fullfname)


def save_curve(imgfile, rankend, outputpath, savepath):
    # save classresult curve vs. Tucker ranks

    classresult = []

    for rank in range(1, rankend + 1):
        fname = imgfile.split('.')[0]+'_'+str(rank)+'.txt'
        # fname = imgfile.split('.')[0]+'_svd_'+str(rank)+'.txt'
        with open(os.path.join(outputpath, fname), 'r') as f:
            classresult.append(int(f.read().split()[0]))
        os.remove(os.path.join(outputpath, fname))

    savename = imgfile.split('.')[0]#+'_svd'
    np.save(os.path.join(savepath, savename), classresult)


    # plt.plot(classresult, "b*", markersize=7)
    # plt.show()

if __name__=="__main__":

    rankend = 299
    sample_path = './samples/'
    inputpath = '../input/'
    outputpath = '../output/'
    savepath = './curves/'

    samples = os.listdir(sample_path)
    for fname in samples:
        print("Procesing %s" % fname)
        # progressive_generation_svd(os.path.join(sample_path, fname), rankend, inputpath)
        progressive_generation(os.path.join(sample_path, fname), rankend, inputpath)
        while(len(os.listdir(outputpath)) < rankend):
            pass
        save_curve(fname, rankend, outputpath, savepath )

        os.remove(os.path.join(sample_path, fname))
    # end of cod
