import numpy as np
import matplotlib.pyplot as plt
import os
import checkSolution as cs
import reduceOps as ro
import bloomFilter as bf

maxCEListLength = 2000


def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]


fileNames = []
# fileNames += listdir_fullpath("/Users/tillspaeth/Google Drive/V14DiffMap")
# fileNames += listdir_fullpath("/Users/tillspaeth/Google Drive/V15DiffMap")
fileNames += listdir_fullpath("/Users/tillspaeth/Google Drive/V16DiffMap")
# fileNames += listdir_fullpath("/Users/tillspaeth/Google Drive/alteVersionenDiffMap")
# fileNames += listdir_fullpath("/Users/tillspaeth/Google Drive/heuristik")

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

minNumOfOps = 999
bestFile = "not set"

n = 3
nn = int(n**2)
p = 23
# bloomFilter = bf.bloomFilter(2*nn*p, 0.00000001)
# identicals = []
bloomFilterWa = bf.bloomFilterMatrix(2000, 0.00001)
identicalsWa = []
bloomFilterWb = bf.bloomFilterMatrix(2000, 0.00001)
identicalsWb = []
bloomFilterWc = bf.bloomFilterMatrix(2000, 0.00001)
identicalsWc = []

opsWaWbWc = []

numOfSolutions = 0
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
        numOfSolutions += 1

        found = bloomFilterWa.store(Wa)
        if found:
            identicalsWa.append(f)
            print("******* Wa mehrfach vorhanden ********")
        found = bloomFilterWb.store(Wb)
        if found:
            identicalsWb.append(f)
            print("******* Wb mehrfach vorhanden ********")
        found = bloomFilterWc.store(Wc)
        if found:
            identicalsWc.append(f)
            print("******* Wc mehrfach vorhanden ********")

        numOfOps = 0
        print("Wa", end=" ", flush=True)
        A, L = ro.optimizeAddsSubs(Wa, maxCEListLength)
        R = ro.calcResiduals(A, L, Wa)
        optOps = ro.countOptimizedOps(A, L, R)
        if optOps < 0:
            optOps = countRawOps(Wa)
        opsWa = optOps
        numOfOps += optOps
        print("Wb", end=" ", flush=True)
        A, L = ro.optimizeAddsSubs(Wb, maxCEListLength)
        R = ro.calcResiduals(A, L, Wb)
        optOps = ro.countOptimizedOps(A, L, R)
        if optOps < 0:
            optOps = countRawOps(Wb)
        opsWb = optOps
        numOfOps += optOps
        print("Wc", end=" ", flush=True)
        A, L = ro.optimizeAddsSubs(Wc, maxCEListLength)
        R = ro.calcResiduals(A, L, Wc)
        optOps = ro.countOptimizedOps(A, L, R)
        if optOps < 0:
            optOps = ro.countRawOps(Wc)
        opsWc = optOps
        numOfOps += optOps

        opsWaWbWc.append((opsWa, opsWb, opsWc))

        print("     Anzahl Adds/Subs ≤", numOfOps, " (", minNumOfOps, ")")
        if numOfOps < minNumOfOps:
            minNumOfOps = numOfOps
            bestFile = f

        print("---------------------------------------------------------------")
print(" ")
print("best file: ", bestFile)
print(" ")
print("total numb. of solutions: ", numOfSolutions)
print(" ")
# print("identicals Wa:")
# print(identicalsWa)
# print(" ")
# print("identicals Wb:")
# print(identicalsWb)
# print(" ")
# print("identicals Wc:")
# print(identicalsWc)
# print("---------------------------------------------------------------")
# print(" ")
#
# identicalsWa = []
# identicalsWa.append(
#     "/Users/tillspaeth/Google Drive/V16DiffMap/solution_3_1277_1564202673.5679944_V16.npy")

WaGroups = []
for i in range(len(identicalsWa)):
    newGroup = []
    newGroup.append(identicalsWa[i])
    print("Wa-identicals for ", identicalsWa[i], ":")
    fileContent1 = np.load(identicalsWa[i], allow_pickle=True)
    Wa1 = np.array(fileContent1[0], dtype=int)
    Wa1 = bf.sortMatrix(Wa1)
    for f in fileNames:
        if f != identicalsWa[i]:
            fileContent2 = np.load(f, allow_pickle=True)
            Wa2 = np.array(fileContent2[0], dtype=int)
            Wa2Pos = bf.sortMatrix(Wa2)
            Wa2Neg = bf.sortMatrix(-Wa2)
            if np.array_equal(Wa1, Wa2Pos) or np.array_equal(Wa1, Wa2Neg):
                print("    ", f)
                newGroup.append(f)
    WaGroups.append(newGroup)
print(" ")

WbGroups = []
for i in range(len(identicalsWb)):
    newGroup = []
    newGroup.append(identicalsWb[i])
    print("Wb-identicals for ", identicalsWb[i], ":")
    fileContent1 = np.load(identicalsWb[i], allow_pickle=True)
    Wb1 = np.array(fileContent1[0], dtype=int)
    Wb1 = bf.sortMatrix(Wb1)
    for f in fileNames:
        if f != identicalsWb[i]:
            fileContent2 = np.load(f, allow_pickle=True)
            Wb2 = np.array(fileContent2[0], dtype=int)
            Wb2Pos = bf.sortMatrix(Wb2)
            Wb2Neg = bf.sortMatrix(-Wb2)
            if np.array_equal(Wb1, Wb2Pos) or np.array_equal(Wb1, Wb2Neg):
                print("    ", f)
                newGroup.append(f)
    WbGroups.append(newGroup)
print(" ")

WcGroups = []
for i in range(len(identicalsWc)):
    newGroup = []
    newGroup.append(identicalsWc[i])
    print("Wc-identicals for ", identicalsWc[i], ":")
    fileContent1 = np.load(identicalsWc[i], allow_pickle=True)
    Wc1 = np.array(fileContent1[0], dtype=int)
    Wc1 = bf.sortMatrix(Wc1)
    for f in fileNames:
        if f != identicalsWc[i]:
            fileContent2 = np.load(f, allow_pickle=True)
            Wc2 = np.array(fileContent2[0], dtype=int)
            Wc2Pos = bf.sortMatrix(Wc2)
            Wc2Neg = bf.sortMatrix(-Wc2)
            if np.array_equal(Wc1, Wc2Pos) or np.array_equal(Wc1, Wc2Neg):
                print("    ", f)
                newGroup.append(f)
    WcGroups.append(newGroup)
print(" ")

np.save("identicalsWaWbWc", [WaGroups, WbGroups, WcGroups])
# quit()

##############################################

# f = "solution_3_1117_1562626845.291288_V15.npy"
# fileContent = np.load(f, allow_pickle=True)
# cs.checkSolutionInt([fileContent[0], fileContent[1], fileContent[2]])
# Wa = np.matrix(fileContent[0], dtype=int)
# Wb = np.matrix(fileContent[1], dtype=int)
# Wc = np.matrix(fileContent[2], dtype=int)
#
# # so.reducedOps(Wc)
#
# A, L = ro.optimizeAddsSubs(Wa, 2000)
# R = ro.calcResiduals(A, L, Wa)
# ro.countOptimizedOps(A, L, R)
# ro.countRawOps(Wa)
#
# A, L = ro.optimizeAddsSubs(Wb, 2000)
# R = ro.calcResiduals(A, L, Wb)
# ro.countOptimizedOps(A, L, R)
# ro.countRawOps(Wb)
#
# A, L = ro.optimizeAddsSubs(Wc, 3000)
# R = ro.calcResiduals(A, L, Wc)
# ro.countOptimizedOps(A, L, R)
# ro.countRawOps(Wc)
