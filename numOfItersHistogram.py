import numpy as np
import matplotlib.pyplot as plt
from numba import jit
import os
import checkSolution_V2 as cs
from scipy.optimize import curve_fit

os.chdir("/Users/tillspaeth/Google Drive/V21DiffMap")
fileNames = os.listdir()
TfileNames = []
for f in fileNames:
    if "_V21.npy" in f:
        TfileNames.append(f)
fileNames = TfileNames

itersDict = dict()
iters = []

for f in fileNames:
    fc = np.load(f, allow_pickle=True)

    if "array" in str(type(fc[0])):
        if not cs.checkSolutionInt([fc[0], fc[1], fc[2]], 0.0000001):
            print("Lösung nicht korrekt !!!!")
            exit()

        numOfIters = len(fc[4])

        iters.append(numOfIters)

        if numOfIters in itersDict:
            itersDict[numOfIters] += 1
        else:
            itersDict[numOfIters] = 1

iters = np.array(iters, dtype=int)

plt.rcParams.update({'font.size': 14})
plt.hist(iters, bins=50)
plt.xlabel("iterations")
plt.ylabel("frequency")

plt.savefig("/Users/tillspaeth/Desktop/Masterarbeit/Ausarbeitung/itersHisto.png",
            dpi=300,  bbox_inches='tight')
plt.close()
