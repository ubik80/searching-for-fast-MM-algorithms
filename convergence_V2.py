import numpy as np
import matplotlib.pyplot as plt
from numba import jit
import os
import checkSolution as cs
np.set_printoptions(precision=2, suppress=True)

os.chdir("/Users/tillspaeth/Google Drive/V19DiffMap")
fileNames = os.listdir()
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

i = 0
for f in fileNames:
    i += 1
    sol = np.load(f, allow_pickle=True)
    if len(sol) >= 10:
        Wa = sol[0]
        Wb = sol[1]
        Wc = sol[2]
        if not cs.checkSolutionInt([Wa, Wb, Wc]):
            print("Lösung nicht korrekt !!!!")
            exit()
        jumpFaktor = sol[3]
        diffs = sol[4]
        jumps = sol[5]
        heights = sol[6]

        print("filename: ", f)
        maxY = np.max(diffs)-0.2
        minY = np.min(diffs)-0.2
        plt.rcParams.update({'font.size': 10})
        plt.plot(diffs)
        plt.xlabel("iteration")
        plt.ylabel("| Δ |")
        y = [minY, maxY]
        for j in range(len(jumps)):
            x = [jumps[j], jumps[j]]
            plt.plot(x, y, '-r', alpha=0.5)
            #plt.text(jumps[j]-20, maxY-0.2, heights[j])
        ttl = 'jumpFactor:'+str(jumpFaktor)
        plt.title(ttl)
        picName = 'example_'+str(i)+'.png'
        plt.savefig(picName, dpi=300)
        plt.close()
