import numpy as np
import matplotlib.pyplot as plt
import os


def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]


fileNames = listdir_fullpath("/Users/tillspaeth/Google Drive/V19DiffMap")
ff = []
for f in fileNames:
    if ".npy" in f:
        ff.append(f)
fileNames = ff
ff = []
for f in fileNames:
    if not "5000" in f:
        ff.append(f)
fileNames = ff

fileContent = np.load(fileNames[1], allow_pickle=True)
diffs = fileContent[4]

plt.rcParams.update({'font.size': 14})
plt.plot(diffs)
plt.xlabel("iteration")
plt.ylabel("| Δ |")
# plt.title("n=3")

plt.savefig("/Users/tillspaeth/Desktop/Masterarbeit/Ausarbeitung/behavTrend2.png", dpi=300)
plt.close()
