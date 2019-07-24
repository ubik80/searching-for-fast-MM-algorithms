import numpy as np
import matplotlib.pyplot as plt
import os
import checkSolution as cs
import summationOps as so


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
        if np.linalg.matrix_rank(Wa) < 9 or np.linalg.matrix_rank(Wb) < 9:
            print("Rang < N !!!")
            quit()

        print(f)
        numOfOps = so.reducedOps(Wa)  # +so.reducedOps(Wb)+so.reducedOps(Wc)
        # numOfOps = so.numOfOps(Wc)
        print("Anzahl Adds/Subs (Wc) ≤", numOfOps, " (", minNumOfOps, ")")
        if numOfOps < minNumOfOps:
            minNumOfOps = numOfOps
            bestFile = f

print("------------------------")
print("best file: ", bestFile)
quit()

##############################################

f = "/Users/tillspaeth/Google Drive/V14DiffMap/solution_3_25_1562364026.7989964_V14.npy"
fileContent = np.load(f, allow_pickle=True)
cs.checkSolutionInt([fileContent[0], fileContent[1], fileContent[2]])
Wa = np.matrix(fileContent[0], dtype=int)
Wb = np.matrix(fileContent[1], dtype=int)
Wc = np.matrix(fileContent[2], dtype=int)

so.reducedOps(Wc)