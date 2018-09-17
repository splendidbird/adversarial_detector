
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors


classresult_fixed = []
classresult_orig = []
classresult_attack = []

for fn in range(1, 40):
    with open("3_fixed_"+str(fn)+".txt", 'r') as f:
        classresult_orig.append(int(f.read().split()[0]))
    with open("2_fixed_"+str(fn)+".txt", 'r') as f:
        classresult_attack.append(int(f.read().split()[0]))
    with open("1_fixed_"+str(fn)+".txt", 'r') as f:
        classresult_fixed.append(int(f.read().split()[0]))

colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
plt.figure()
plt.plot(classresult_orig, "b*", markersize=7)
plt.plot(classresult_attack,"r*", markersize=7)
plt.plot(classresult_fixed, "g*", markersize=7)
plt.legend(["Original", "Attacked", "Fixed"])
plt.xlabel("Image Info Component")
plt.ylabel("Classified Category")

plt.show()
