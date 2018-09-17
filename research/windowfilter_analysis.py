import os
import math
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

class adv_filter(object):
    #operate at the level of tensor-format image_array


    def __self__(self, inputarray):
        # inputarray - input image array in tensor format
        self._inputarray = inputarray


    def svd_stack_crop(self, imgarray, n_start, n_end):
        pass


    def svd_layer_crop(self, layer, imgarray, n_start, n_end):)
        pass


    def svdRGBimg(self, imgarray, n_svd):
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

if __name__=='__main__':
    for fnn in range(14, 31):
        for ftype in ['']:
            target = str(fnn)+".png"
            imgobj = Image.open(target)
            imgarray = np.array(imgobj)

            n_svd = 290
            for n_svd in range(101, n_svd+1):
                reduced_img = Image.fromarray((svdRGBimg(imgarray, n_svd)).astype(np.uint8))
                path = "./reduced1/"
                path2 = "../input/"
                reduced_img.save(os.path.join(path2, str(fnn) + "_" + str(n_svd) + ".png"))
