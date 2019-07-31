import numpy as np
import matplotlib.pyplot as plt
import os
import checkSolution as cs
import reduceOps as ro

maxLength = 2000


def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]


fileNames = listdir_fullpath("/Users/tillspaeth/Google Drive/V14DiffMap")
fileNames += listdir_fullpath("/Users/tillspaeth/Google Drive/V15DiffMap")
fileNames += listdir_fullpath("/Users/tillspaeth/Google Drive/V16DiffMap")
fileNames += listdir_fullpath("/Users/tillspaeth/Google Drive/alteVersionenDiffMap")
fileNames += listdir_fullpath("/Users/tillspaeth/Google Drive/heuristik")

for f in fileNames:
    if not ".npy" in f:
        fileNames.remove(f)
for f in fileNames:
    if "5000" in f:
        fileNames.remove(f)

minNumOfOps = 999
bestFile = "not set"

for f in fileNames:
    fileContent = np.load(f, allow_pickle=True)

    if 'numpy' in str(type(fileContent[0])):
        correct = cs.checkSolutionInt([fileContent[0], fileContent[1], fileContent[2]])
    else:
        correct = False

    if correct:
        Wa = np.matrix(fileContent[0], dtype=int)
        Wb = np.matrix(fileContent[1], dtype=int)
        Wc = np.matrix(fileContent[2], dtype=int)
        Wa, Wb, Wc = np.round(Wa), np.round(Wb), np.round(Wc)
        if np.linalg.matrix_rank(Wa) < 9 or np.linalg.matrix_rank(Wb) < 9:
            print("Rang < N !!!")
            quit()

        print(f)
        numOfOps = 0
        print("Wa", end=" ", flush=True)
        M = Wa
        A, L = ro.optimizeAddsSubs(M, maxLength)
        R = ro.calcResiduals(A, L, M)
        optOps = ro.countOptimizedOps(A, L, R)
        if optOps < 0:
            optOps = countRawOps(M)
        numOfOps += optOps
        print("Wb", end=" ", flush=True)
        M = Wb
        A, L = ro.optimizeAddsSubs(M, maxLength)
        R = ro.calcResiduals(A, L, M)
        optOps = ro.countOptimizedOps(A, L, R)
        if optOps < 0:
            optOps = countRawOps(M)
        numOfOps += optOps
        print("Wc", end=" ", flush=True)
        M = Wc
        A, L = ro.optimizeAddsSubs(M, maxLength)
        R = ro.calcResiduals(A, L, M)
        optOps = ro.countOptimizedOps(A, L, R)
        if optOps < 0:
            optOps = ro.countRawOps(M)
        numOfOps += optOps

        print("     Anzahl Adds/Subs ≤", numOfOps, " (", minNumOfOps, ")")
        if numOfOps < minNumOfOps:
            minNumOfOps = numOfOps
            bestFile = f

        print("---------------------------------------------------------------")
print(" ")
print("         best file: ", bestFile)
quit()

##############################################

f = "solution_3_1117_1562626845.291288_V15.npy"
fileContent = np.load(f, allow_pickle=True)
cs.checkSolutionInt([fileContent[0], fileContent[1], fileContent[2]])
Wa = np.matrix(fileContent[0], dtype=int)
Wb = np.matrix(fileContent[1], dtype=int)
Wc = np.matrix(fileContent[2], dtype=int)

# so.reducedOps(Wc)

A, L = ro.optimizeAddsSubs(Wa, 2000)
R = ro.calcResiduals(A, L, Wa)
ro.countOptimizedOps(A, L, R)
ro.countRawOps(Wa)

A, L = ro.optimizeAddsSubs(Wb, 2000)
R = ro.calcResiduals(A, L, Wb)
ro.countOptimizedOps(A, L, R)
ro.countRawOps(Wb)

A, L = ro.optimizeAddsSubs(Wc, 3000)
R = ro.calcResiduals(A, L, Wc)
ro.countOptimizedOps(A, L, R)
ro.countRawOps(Wc)

# 2000: 28 / 43
