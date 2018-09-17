"""
Image compression via tensor decomposition
==========================================

Example on how to use :func:`tensorly.decomposition.parafac`and :func:`tensorly.decomposition.tucker` on images.
"""
import math
import matplotlib.pyplot as plt
import tensorly as tl
import numpy as np
from scipy.misc import face, imresize
from tensorly.decomposition import parafac
from tensorly.decomposition import tucker
from math import ceil

from PIL import Image

random_state = 12345

image = tl.tensor(imresize(face(), 0.3), dtype='float64')

target ="1.png"
imgobj = Image.open(target)
imgarray = np.array(imgobj)
imgarray_21 = imgarray
for i in range(6):
    imgarray_21 = np.append(imgarray_21, imgarray, axis=2)

image = tl.tensor(imgarray_21)

print("##image shape##")
print(image.shape)

def to_image(tensor):
    """A convenience function to convert from a float dtype back to uint8"""
    im = tl.to_numpy(tensor)
    im -= im.min()
    im /= im.max()
    im *= 255
    return im.astype(np.uint8)

def cp_filter(imgarray, cp_rank):
    """
    imgarray is 3-d numpy array of png image
    """
    imgtensor = tl.tensor(imgarray)
    factors = parafac(imgtensor, rank=cp_rank, init='random', tol=10e-6)
    pass

# Rank of the CP decomposition
# cp_rank = 10
# Rank of the Tucker decomposition
tucker_rank = [20, 20, 30]

# Perform the CP decomposition
# factors = parafac(image, rank=cp_rank, init='random', tol=10e-6)
# print("###### factor size ########")
# print(len(factors[0]))
# print(len(factors[1]))
# print(len(factors[2]))
# print("###### factors[2] ########")
# print(len(factors[0][1]))
# print(len(factors[1][1]))
# print(len(factors[2][1]))
# print(factors[2])
# print(factors.shape)

# Reconstruct the image from the factors
# cp_reconstruction = tl.kruskal_to_tensor(factors)

# Tucker decomposition
core, tucker_factors = tucker(image, ranks=tucker_rank, init='random', tol=10e-5, random_state=random_state)
tucker_reconstruction = tl.tucker_to_tensor(core, tucker_factors)

# Plotting the original and reconstruction from the decompositions
fig = plt.figure()
ax = fig.add_subplot(2, 3, 1)

"""
ax.imshow(to_image(image))
ax.set_title('original')

# Image.fromarray(to_image(cp_reconstruction)).save("1_cp.png")
# ax = fig.add_subplot(2, 3, 2)
# ax.set_axis_off()
# ax.imshow(to_image(cp_reconstruction))
# ax.set_title('CP')

ax = fig.add_subplot(2, 3, 3)
ax.set_axis_off()
ax.imshow(to_image(tucker_reconstruction))
ax.set_title('Tucker')

# ax = fig.add_subplot(2, 3, 4)
# ax.imshow(core[0][1])
# ax.set_title('factors[0]')
#
# ax = fig.add_subplot(2, 3, 5)
# # ax.set_axis_off()
# ax.plot(factors[1][1])
# ax.set_title('factors[1]')
#
# ax = fig.add_subplot(2, 3, 6)
# # ax.set_axis_off()
# # ax.plot(factors[2][1])
# ax.set_title('factors[2]')


plt.tight_layout()
plt.show()
"""

print("core")
print(core.shape)
# print(core)
plt.imshow(np.log(core[0,:,:]**2))
plt.show()

pv = []
pvsum = []
for i in range(core.shape[0]):
    # print(core[i,i,0])
    pv.append(abs((core**2).sum(axis=2)[i,i]))
    pvsum.append(sum(pv))
plt.plot(pv)
plt.plot(pvsum)
plt.show()
print("@@@@@@@@@@@@@@@@")
print(tucker_factors[0].shape)
print(tucker_factors[1].shape)
print(tucker_factors[2].shape)










# end of code
