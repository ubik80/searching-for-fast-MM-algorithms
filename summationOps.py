import numpy as np
import matplotlib.pyplot as plt
import os
import checkSolution as cs


def numOfOps(M):
    ops = 0
    for i in range(M.shape[0]):
        if np.sum(np.abs(M[i, :])) > 0:
            opsPerLine = -1
            for j in range(M.shape[1]):
                if np.abs(M[i, j]) > 0:
                    opsPerLine += 1
            ops += opsPerLine
    return ops


def reducedOps(W):
    M = W.copy()
    MOrg = M.copy()

    # Aufstellen aller möglichen Teilsummen:
    A = []
    for i in range(M.shape[0]):
        A.append(np.array(M[i, :], dtype=int).flatten())
    AA = A.copy()
    while not len(AA) == 0:
        B = AA.copy()
        AA = []
        for i in range(len(B)):
            row = B[i].flatten()
            pos = np.argwhere(row != 0).flatten()
            if len(pos) > 1:
                for j in range(len(pos)):
                    newRow = row.copy()
                    newRow[pos[j]] = 0
                    found = False
                    for a in AA:
                        if np.array_equal(a, newRow):
                            found = True
                            break
                    if not found:
                        for a in AA:
                            if np.array_equal(a, -newRow):
                                found = True
                                break
                    if not found:
                        for a in A:
                            if np.array_equal(a, newRow):
                                found = True
                                break
                    if not found:
                        for a in A:
                            if np.array_equal(a, -newRow):
                                found = True
                                break
                    if not found:
                        AA.append(newRow.flatten())
        A += AA
        if len(A) > 1000:
            break

    # nur verschiedene Zeilen herauskopieren:
    AA = []
    for a in A:
        found = False
        for aa in AA:
            if np.array_equal(a, aa) or np.array_equal(a, -aa):
                found = True
        if not found:
            AA.append(a)
    A = np.matrix(AA)

    # repeats = wie oft werden die Teilsummen verwendet:
    repeats = np.zeros(A.shape[0], dtype=int)
    for i in range(A.shape[0]):
        for ii in range(M.shape[0]):
            contained = True
            for j in range(A.shape[1]):
                if A[i, j] != 0 and A[i, j] != M[ii, j]:
                    contained = False
                    break
            if contained:
                repeats[i] += 1
    for i in range(A.shape[0]):
        for ii in range(M.shape[0]):
            contained = True
            for j in range(A.shape[1]):
                if A[i, j] != 0 and A[i, j] != -M[ii, j]:
                    contained = False
                    break
            if contained:
                repeats[i] += 1

    # nur die Teilsummen behalten welche mindestens zwei mal verwendet werden:
    A = A[np.argwhere(repeats > 1).flatten(), :]
    repeats = repeats[np.argwhere(repeats > 1).flatten()]

    # profit = möglicher Profit bei aussschliesslicher Verwendung jeweils einer Teilsumme:
    profit = np.zeros(A.shape[0], dtype=int)
    for i in range(A.shape[0]):
        row = A[i, :]
        profit[i] = (np.sum(np.abs(row))-1)*(repeats[i]-1)

    # nur Teilsummen mit möglichem Profit > 0 weiter verwenden:
    A = A[np.argwhere(profit > 0).flatten(), :]
    profit = profit[np.argwhere(profit > 0).flatten()]

    # nach Profit absteigend sortieren:
    sorted = np.array(np.argsort(profit)).flatten()
    sorted = sorted[range(len(sorted)-1, -1, -1)]
    A = A[sorted]

    # Teilsummen in M durch Zeilen in A ersetzen -> MM:
    MM = np.zeros([M.shape[0], A.shape[0]], dtype=int)
    for i in range(A.shape[0]):
        used = False
        for ii in range(M.shape[0]):
            contained = True
            for j in range(A.shape[1]):
                if A[i, j] != 0 and A[i, j] != M[ii, j]:
                    contained = False
            if contained:
                M[ii, :] -= A[i, :]
                MM[ii, i] = 1
                used = True
            contained = True
            for j in range(A.shape[1]):
                if A[i, j] != 0 and A[i, j] != -M[ii, j]:
                    contained = False
            if contained:
                M[ii, :] += A[i, :]
                MM[ii, i] = -1
                used = True
        if not used:
            A[i, :] *= 0

    # Nullzeilen aus A und entsprechende Spalten aus MM löschen:
    AA = []
    MMM = []
    for i in range(A.shape[0]):
        if np.sum(np.abs(A[i, :])) != 0:
            AA.append(np.array(A[i, :]).flatten())
            MMM.append(np.array(MM[:, i]).flatten())
    A = np.matrix(AA, dtype=int)
    MM = np.matrix(MMM, dtype=int).T

    # Spalten mit Summe=1 aus MM und entsprechende Zeilen aus A löschen:
    AA = []
    MMM = []
    for j in range(MM.shape[1]):
        if np.sum(np.abs(MM[:, j])) > 1:
            MMM.append(np.array(MM[:, j]).flatten())
            AA.append(np.array(A[j, :]).flatten())
    A = np.matrix(AA, dtype=int)
    MM = np.matrix(MMM, dtype=int).T

    if A.shape[0] == 0 or A.shape[1] == 0:
        print("keine Teilsummen identifiziert")
        return -999

    # Bestimmung von finalem M:
    M = MOrg.copy()
    for i in range(A.shape[0]):
        for ii in range(M.shape[0]):
            contained = True
            for j in range(A.shape[1]):
                if A[i, j] != 0 and A[i, j] != M[ii, j]:
                    contained = False
            if contained:
                M[ii, :] -= A[i, :]
            contained = True
            for j in range(A.shape[1]):
                if A[i, j] != 0 and A[i, j] != -M[ii, j]:
                    contained = False
            if contained:
                M[ii, :] += A[i, :]

    # Anzahl der nötigen Summationen zwischen MM*A und M:
    adds = 0
    for i in range(M.shape[0]):
        if np.sum(np.abs(M[i, :])) > 0 and np.sum(np.abs(MM[i, :])) > 0:
            adds += 1

    # Anzahl der Wiederholungen von Teilsummen + Negation
    repeats = np.zeros(MOrg.shape[0])
    for i in range(A.shape[0]):
        for ii in range(MOrg.shape[0]):
            contained = True
            for j in range(A.shape[1]):
                if A[i, j] != 0 and A[i, j] != MOrg[ii, j]:
                    contained = False
            if contained:
                repeats[ii] = i+1
            contained = True
            for j in range(A.shape[1]):
                if A[i, j] != 0 and A[i, j] != -MOrg[ii, j]:
                    contained = False
            if contained:
                repeats[ii] = -(i+1)

    # Anzahl der nötigen Negationen von Teilsummen:
    numOfNegs = 0
    for i in range(A.shape[0]):
        neg = False
        pos = False
        for r in repeats:
            if r == i+1:
                pos = True
            if r == -(i+1):
                neg = True
        if neg and pos:
            numOfNegs += 1

    ret = numOfOps(A)+numOfOps(MM)+numOfOps(M)+adds+numOfNegs
    return ret


# os.chdir("/Users/tillspaeth/Google Drive/V16DiffMap")
# fileNames = os.listdir()
# f = fileNames[13]
# fileContent = np.load(f, allow_pickle=True)
# Wa = np.matrix(fileContent[0], dtype=int)
# Wb = np.matrix(fileContent[1], dtype=int)
# Wc = np.matrix(fileContent[2], dtype=int)
#
# print(numOfOps(Wa)+numOfOps(Wb)+numOfOps(Wc))
# print(reducedOps(Wa)+reducedOps(Wb)+reducedOps(Wc))


# R = np.linalg.lstsq(A.T, Wa.T, rcond=None)
# R = np.matrix(R[0])
# WWa = R.T.dot(A)
# if np.sum(np.abs(WWa-Wa)) > 0.00000001:
#     failed = True
#     print("failed")
#     quit()
