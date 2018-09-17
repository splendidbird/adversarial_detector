import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy import fftpack
from PIL import Image
from scipy.misc import imread
from scipy.ndimage import gaussian_filter


def plot_spectrum(im_fft):
    plt.imshow(np.abs(np.fft.fftshift(im_fft)), norm=LogNorm(vmin=5))
    plt.colorbar()


def FFTfilterRGB(im_rgb, startx, endx, starty, endy):
    # im_rgb - RBG 3-D array image data acquired from plt.imread
    la, lb, lc = im_rgb.shape
    im_fixed = np.zeros((la,lb,lc), 'uint8')
    for color in range(lc):
        im_fft2 = np.fft.fft2(im_rgb[:,:,color])
        im_fft2[startx:endx,:] = 0
        im_fft2[:,starty:endy] = 0
        im_fixed[:,:,color] = np.abs(np.fft.ifft2(im_fft2)) * 256
    return im_fixed


# for edge in range(1, 40):
#     for img in ['1','2','3']:
#         im_attack = plt.imread(img+'_attack.png').astype(float)
#
#         im_fixed = FFTfilterRGB(im_attack, edge*5, 300-edge*5, edge*5, 300-edge*5)
#         png_fixed = Image.fromarray(im_fixed)
#         png_fixed.save(img+'_fixed_'+str(edge)+'.png')

    #print im_fixed.shape
    # plt.figure()
    # #plt.imshow(im_fixed[:,:,1])
    # plt.imshow(png_fixed)
    # plt.show()

# below for debug only
#
im_orig = gaussian_filter(imread('1_orig.png', mode='RGB').astype(np.float), sigma=1)/255.0*2.0-1.0
print im_orig.shape

im_attack = imread('1_attack.png', mode='RGB').astype(np.float)/255.0*2.0-1.0
print im_attack.shape

im_orig_fft = np.fft.fft2(im_orig[:,:,2])
im_attack_fft = np.fft.fft2(im_attack[:,:,2])

plt.figure()
plt.imshow(im_orig)
plt.show()

plt.figure()
plt.imshow(im_attack)
plt.show()

plt.figure()
plot_spectrum(im_orig_fft)
plt.show()

plt.figure()
plot_spectrum(im_attack_fft)
plt.show()

plt.figure()
plot_spectrum(im_attack_fft - im_orig_fft)
plt.show()

im_attack_fft2 = im_attack_fft.copy()

im_attack_fft2[100:200,:] = 0
# im_fft2[-1:,:1] = 0
# im_fft2[:,:1] = 0
im_attack_fft2[:,100:200] = 0

plt.figure()
plt.imshow(np.abs(np.fft.fftshift(im_attack_fft2)), norm=LogNorm(vmin=5))
plt.show()


im_fixed = np.abs(np.fft.ifft2(im_attack_fft2))
plt.figure()
# plt.imshow(im_new)
plt.imshow(im_fixed, plt.cm.gray)
plt.show()
