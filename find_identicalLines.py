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

        for i in range(Wa.shape[0]):
            for ii in range(i+1, Wa.shape[0]):
                if np.array_equal(Wa[i, :], Wa[ii, :]):
                    print("FOUND in Wa!!!")
                    print(f)
                    print("Wa:")
                    print(Wa[i, :])
                    print(Wa[ii, :])
                    print("Wb:")
                    print(Wb[i, :])
                    print(Wb[ii, :])
                    print("--------------------------------")
                    # quit()
        for i in range(Wb.shape[0]):
            for ii in range(i+1, Wb.shape[0]):
                if np.array_equal(Wb[i, :], Wb[ii, :]):
                    print("FOUND in Wb!!!")
                    print(f)
                    print("Wa:")
                    print(Wa[i, :])
                    print(Wa[ii, :])
                    print("Wb:")
                    print(Wb[i, :])
                    print(Wb[ii, :])
                    print("--------------------------------")
                    # quit()
