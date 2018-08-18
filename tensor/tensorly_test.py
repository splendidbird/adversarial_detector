import matplotlib.pyplot as plt
import tensorly as tl
import numpy as np
from scipy.misc import face, imresize
from tensorly.decomposition import parafac
from tensorly.decomposition import tucker
from math import ceil

random_state = 12345
image = tl.tensor(imresize(face(), 0.3), dtype='float64')

image = image[:,:,0]

print (image.shape)
# Rank of the Tucker decomposition
tucker_rank = [10, 10, 3]

def to_image(tensor):
    """A convenience function to convert from a float dtype back to uint8"""
    im = tl.to_numpy(tensor)
    im -= im.min()
    im /= im.max()
    im *= 255
    return im.astype(np.uint8)

# Tucker decomposition
core, tucker_factors = tucker(image, ranks=tucker_rank, init='random', tol=10e-5, random_state=random_state)
tucker_reconstruction = tl.tucker_to_tensor(core, tucker_factors)
print("core shape:")
print(core.shape)
print("factors shape")
print(len(tucker_factors[0]))
print (tucker_factors)

# Plotting the original and reconstruction from the decompositions
fig = plt.figure()
ax = fig.add_subplot(1, 3, 1)
ax.set_axis_off()
ax.imshow(to_image(image))
ax.set_title('original')

ax = fig.add_subplot(1, 3, 3)
ax.set_axis_off()
ax.imshow(to_image(tucker_reconstruction))
ax.set_title('Tucker')

plt.tight_layout()
plt.show()
