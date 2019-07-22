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

# R = np.linalg.lstsq(A.T, Wa.T, rcond=None)
# R = np.matrix(R[0])
# WWa = R.T.dot(A)
# if np.sum(np.abs(WWa-Wa)) > 0.00000001:
#     failed = True
#     print("failed")
#     quit()


def reduceAddSub(Wa):
    WaOrig = Wa.copy()
    # Aufstellen aller möglichen Teilsummen:
    A = []
    for i in range(Wa.shape[0]):
        A.append(np.array(Wa[i, :], dtype=int).flatten())

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
                    AA.append(newRow)
        A = A.copy()+AA.copy()

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
        for ii in range(Wa.shape[0]):
            contained = True
            for j in range(A.shape[1]):
                if A[i, j] != 0 and A[i, j] != Wa[ii, j]:
                    contained = False
                    break
            if contained:
                repeats[i] += 1
    for i in range(A.shape[0]):
        for ii in range(Wa.shape[0]):
            contained = True
            for j in range(A.shape[1]):
                if A[i, j] != 0 and A[i, j] != -Wa[ii, j]:
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
    #profit = profit[sorted]

    # Teilsummen in Wa durch Zeilen in WWa ersetzen:
    WWa = np.zeros([Wa.shape[0], A.shape[0]], dtype=int)
    for i in range(A.shape[0]):
        used = False
        for ii in range(Wa.shape[0]):
            contained = True
            for j in range(A.shape[1]):
                if A[i, j] != 0 and A[i, j] != Wa[ii, j]:
                    contained = False
            if contained:
                Wa[ii, :] -= A[i, :]
                WWa[ii, i] = 1
                used = True
            contained = True
            for j in range(A.shape[1]):
                if A[i, j] != 0 and A[i, j] != -Wa[ii, j]:
                    contained = False
            if contained:
                Wa[ii, :] += A[i, :]
                WWa[ii, i] = -1
                used = True
        if not used:
            A[i, :] *= 0

    # Nullzeilen aus A und entsprechende Spalten aus WWa löschen:
    AA = []
    WWWa = []
    for i in range(A.shape[0]):
        if np.sum(np.abs(A[i, :])) != 0:
            AA.append(np.array(A[i, :]).flatten())
            WWWa.append(np.array(WWa[:, i]).flatten())
    A = np.matrix(AA, dtype=int)
    WWa = np.matrix(WWWa, dtype=int).T

    # Spalten mit Summe=1 aus WWa und entsprechende Zeilen aus A löschen:
    AA = []
    WWWa = []
    for j in range(WWa.shape[1]):
        if np.sum(np.abs(WWa[:, j])) > 1:
            WWWa.append(np.array(WWa[:, j]).flatten())
            AA.append(np.array(A[j, :]).flatten())
    A = np.matrix(AA, dtype=int)
    WWa = np.matrix(WWWa, dtype=int).T

    # Bestimmung von finalem Wa:
    Wa = WaOrig.copy()
    for i in range(A.shape[0]):
        for ii in range(Wa.shape[0]):
            contained = True
            for j in range(A.shape[1]):
                if A[i, j] != 0 and A[i, j] != Wa[ii, j]:
                    contained = False
            if contained:
                Wa[ii, :] -= A[i, :]
            contained = True
            for j in range(A.shape[1]):
                if A[i, j] != 0 and A[i, j] != -Wa[ii, j]:
                    contained = False
            if contained:
                Wa[ii, :] += A[i, :]

    # Anzahl der nötigen Summationen zwischen WWa*A und Wa:
    adds = 0
    for i in range(Wa.shape[0]):
        if np.sum(np.abs(Wa[i, :])) > 0 and np.sum(np.abs(WWa[i, :])) > 0:
            adds += 1

    # Anzahl der Wiederholungen von Teilsummen + Negation
    repeats = np.zeros(WaOrig.shape[0])
    for i in range(A.shape[0]):
        for ii in range(WaOrig.shape[0]):
            contained = True
            for j in range(A.shape[1]):
                if A[i, j] != 0 and A[i, j] != WaOrig[ii, j]:
                    contained = False
            if contained:
                repeats[ii] = i+1
            contained = True
            for j in range(A.shape[1]):
                if A[i, j] != 0 and A[i, j] != -WaOrig[ii, j]:
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

    return numOfOps(A)+numOfOps(WWa)+numOfOps(Wa)+adds+numOfNegs

################################################################################


os.chdir("/Users/tillspaeth/Google Drive/V16DiffMap")
fileNames = os.listdir()

for f in fileNames:
    #f = fileNames[0]
    fileContent = np.load(f, allow_pickle=True)
    notEmpty = ('numpy' in str(type(fileContent[0])))
    if notEmpty:
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
    print("Anzahl Adds/Subs ≤ ",
          reduceAddSub(Wa.copy())+reduceAddSub(Wb.copy())+reduceAddSub(Wc.copy()))
