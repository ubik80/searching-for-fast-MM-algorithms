import numpy as np
import os
import checkSolution as cs
import reduceOps as ro


def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]


fileNames = []
fileNames += listdir_fullpath("/Users/tillspaeth/Google Drive/alteVersionenDiffMap")
fileNames += listdir_fullpath("/Users/tillspaeth/Google Drive/heuristik")
fileNames += listdir_fullpath("/Users/tillspaeth/Google Drive/V14DiffMap")
fileNames += listdir_fullpath("/Users/tillspaeth/Google Drive/V15DiffMap")
fileNames += listdir_fullpath("/Users/tillspaeth/Google Drive/V16DiffMap")
fileNames += listdir_fullpath("/Users/tillspaeth/Google Drive/V17DiffMap")
fileNames += listdir_fullpath("/Users/tillspaeth/Google Drive/V18DiffMap")
fileNames += listdir_fullpath("/Users/tillspaeth/Google Drive/V19DiffMap")
fileNames += listdir_fullpath("/Users/tillspaeth/Google Drive/V20DiffMap")
fileNames += listdir_fullpath("/Users/tillspaeth/Google Drive/V21DiffMap")

ff = []
for f in fileNames:
    if ".npy" in f and not "5000" in f:
        ff.append(f)
fileNames = ff
# ff = []
# for f in fileNames:
#     if not "5000" in f:
#         ff.append(f)
# fileNames = ff

numOfSolutions = 0
solutions = []

for f in fileNames:
    fileContent = np.load(f, allow_pickle=True)

    if 'numpy.ndarray' in str(type(fileContent[0])):
        if len(fileContent) >= 10:
            if fileContent[9] < 0:
                correct = False
            else:  # <0...
                correct = cs.checkSolutionInt([fileContent[0], fileContent[1], fileContent[2]])
        else:  # len...
            correct = cs.checkSolutionInt([fileContent[0], fileContent[1], fileContent[2]])
    else:  # numpy...
        correct = False

    if correct:
        print(f)
        numOfSolutions += 1
        Wa = np.matrix(fileContent[0], dtype=int)
        Wb = np.matrix(fileContent[1], dtype=int)
        Wc = np.matrix(fileContent[2], dtype=int)
        Aa, La = ro.optimizeAddsSubs(Wa, 2000)
        Ra = ro.calcResiduals(Aa, La, Wa)
        Ab, Lb = ro.optimizeAddsSubs(Wb, 2000)
        Rb = ro.calcResiduals(Ab, Lb, Wb)
        Ac, Lc = ro.optimizeAddsSubs(Wc, 2000)
        Rc = ro.calcResiduals(Ac, Lc, Wc)

        solutions.append([Wa, Wb, Wc, Aa, La, Ra, Ab, Lb, Rb, Ac, Lc, Rc])

print("total numb. of solutions: ", numOfSolutions)
print(" ")

np.save("solutions_n3", solutions)

##############################################

# S = np.load("solutions_n3.npy", allow_pickle=True)
# s = S[13]
# s[12]
