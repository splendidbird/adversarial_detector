
import os
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import numpy as np


classresult_orig = []
classresult_attack = []

for fnn in range(14, 31):
    for fn in range(101, 291):
        with open(str(fnn) + "_" + str(fn) + ".txt", 'r') as f:
            classresult_orig.append(int(f.read().split()[0]))
            if os.path.isfile(str(fnn) + "_" + str(fn) + ".png"):
                os.remove(str(fnn) + "_" + str(fn) + ".png")
        # with open("1_attack_"+str(fn)+".txt", 'r') as f:
        #     classresult_attack.append(int(f.read().split()[0]))
    print(np.std(np.array(classresult_orig)))
    classresult_orig = []

# colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
# plt.figure()
# plt.plot(classresult_orig, "b*", markersize=7)
# # plt.plot(classresult_attack,"r*", markersize=7)
#
# # plt.legend(["Original", "Attacked"])
# plt.xlabel("Image Info Component")
# plt.ylabel("Classified Category")
# plt.show()
