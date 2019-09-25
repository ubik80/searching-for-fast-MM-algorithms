import numpy as np
import matplotlib.pyplot as plt
import os
import checkSolution_V2 as cs


# def listdir_fullpath(d):
#     return [os.path.join(d, f) for f in os.listdir(d)]
#
#
# fileNames = listdir_fullpath("/Users/tillspaeth/Google Drive/V19DiffMap")
# ff = []
# for f in fileNames:
#     if ".npy" in f:
#         ff.append(f)
# fileNames = ff
# ff = []
# for f in fileNames:
#     if not "5000" in f:
#         ff.append(f)
# fileNames = ff

fileName = "/Users/tillspaeth/Google Drive/n5/solution_5_118_6491_1568085317.398804_V25.npy"

fc = np.load(fileName, allow_pickle=True)
diffs = fc[4]

plt.rcParams.update({'font.size': 14})
plt.plot(diffs)
plt.xlabel("iteration")
plt.ylabel("| Δ |")
# plt.title("n=3")

# plt.savefig("/Users/tillspaeth/Desktop/Masterarbeit/Ausarbeitung/behavTrend2.png", dpi=300)
# plt.close()
cs.checkResult(fc[0], fc[1], fc[2], prints=True)
fc[2][3][3] = 1.0
