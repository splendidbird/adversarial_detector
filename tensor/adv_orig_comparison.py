"""
Compare Tucker core values distributions between adversarial and original images
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

def tucker_filter(imgarray, tucker_rank, compfilename):
    """
    imgarray is 3-d numpy array of png image
    """
    imgtensor = tl.tensor(imgarray)

    # Tucker decomposition
    core, tucker_factors = tucker(imgtensor, ranks=tucker_rank, init='random', tol=10e-5, random_state=random_state)
    tucker_reconstruction = tl.tucker_to_tensor(core, tucker_factors)
    Image.fromarray(to_image(tucker_reconstruction)).save('../input/'+compfilename)
    return 0

def tucker_core(imgarray, tucker_rank):
    """
    imgarray is 3-d numpy array of png image
    """
    imgtensor = tl.tensor(imgarray)
    # Tucker decomposition
    core, tucker_factors =tucker(imgtensor, ranks=tucker_rank, init='random', tol=10e-5, random_state=random_state)
    # print(tucker_factors[0].shape)
    # print(tucker_factors[1].shape)
    # print(tucker_factors[2].shape)
    return core, tucker_factors


if __name__=="__main__":

    target ="3_attack.png"
    imgobj = Image.open(target)
    imgarray = np.array(imgobj)
    imgarray_adv = imgarray
    #imgarray = np.array([[[1,2,3,4],[4,5,6,4],[7,8,9,4]],[[11,12,13,4],[14,15,16,4],[17,18,19,4]]])
    path = "../input/"

    core_rank = [299,299,1]

    core_adv, tucker_factors = tucker_core(imgarray, core_rank)
    [facA, facB, facC] = tucker_factors
    print(imgarray.shape)
    print(facA.shape)
    print(facB.shape)
    print(facC.shape)

    # plt.imshow(np.log(core_orig[:,:,0]**2))
    # plt.colorbar()
    # plt.show()

    target ="3_orig.png"
    imgobj = Image.open(target)
    imgarray = np.array(imgobj)
    imgarray_orig = imgarray

    mid1 = np.matmul(imgarray, facC)
    print(mid1.shape)
    mid1 = np.swapaxes(mid1,1,2)
    mid2 = np.matmul(mid1, facB)
    mid2 = np.swapaxes(mid2,1,2)
    print(mid2.shape)
    mid2 = np.swapaxes(mid2,0,2)
    mid3 = np.matmul(mid2, facA)
    core_orig = np.swapaxes(mid3,0,2)

    print(core_orig.shape)

    # for rank in range(1,300,1):
    #     print ("Attack rank: %s" % str(rank))
    #     tucker_rank = [rank, rank, 2]
    #     compfilename = "1_attack_"+str(tucker_rank[0])+"_"+str(tucker_rank[1])+"_"+str(tucker_rank[2])+".png"
    #     tucker_filter(imgarray, tucker_rank, os.path.join(path,compfilename))

    img_diff = imgarray_orig - imgarray_adv
    plt.imshow(img_diff[:,:,0])
    plt.title("image difference")
    plt.show()

    core_diff = core_adv - core_orig
    print("core_diff")
    print(core_diff.shape)
    # plt.imshow(np.log(core_diff[:,:,0]))

    plt.imshow(np.log(core_orig[:,:,0]**2))
    plt.title("original core")
    plt.colorbar()
    plt.show()

    plt.imshow(np.log(core_adv[:,:,0]**2))
    plt.title("adversarial core")
    plt.colorbar()
    plt.show()

    plt.imshow(np.log(core_diff[:,:,0]**2))
    plt.title("core difference")
    plt.colorbar()
    plt.show()

    # fig = plt.figure()
    # ax = fig.add_subplot(1, 3, 1)
    # plt.imshow(np.log(core_orig[:,:,0]))
    #
    # ax = fig.add_subplot(1, 3, 2)
    # plt.imshow(np.log(core_orig[:,:,1]))
    #
    # ax = fig.add_subplot(1, 3, 3)
    # plt.imshow(np.log(core_orig[:,:,2]))
    #
    # plt.show()

    diag = []
    for i in range(299):
        diag.append(abs((core_adv[i,i,0] - core_orig[i,i,0])))
    plt.plot(diag,"r*")
    plt.show()

    diag = []
    for i in range(299):
        diag.append(np.log(abs(core_adv[i,i,0])))
    plt.plot(diag,"r*")
    # plt.show()
    diag = []
    for i in range(299):
        diag.append(np.log(abs(core_orig[i,i,0])))
    plt.plot(diag,"b*")
    plt.show()


    # end of cod
