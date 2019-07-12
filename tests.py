import numpy as np
import matplotlib.pyplot as plt
from numba import jit
import os
np.set_printoptions(precision=2, suppress=True)


def checkSolution(W):
    p = W[0].shape[0]
    nn = W[0].shape[1]
    n = int(np.sqrt(nn))
    BIdx = np.array([k*n for k in range(n)])
    c = np.zeros(nn, dtype=float)
    Wa = np.round(W[0])
    Wb = np.round(W[1])
    Wc = np.round(W[2])
    Wa = np.maximum(np.minimum(Wa, 1.0), -1.0)
    Wb = np.maximum(np.minimum(Wb, 1.0), -1.0)
    Wc = np.maximum(np.minimum(Wc, 1.0), -1.0)

    @jit(nopython=True, nogil=True, cache=True)
    def fastLoop(n, nn, p, BIdx, c, Wa, Wb, Wc):
        for i in range(100):
            a = np.random.rand(nn)*2.0-1.0
            b = np.random.rand(nn)*2.0-1.0

            for ii in range(n):  # Matrixmultiplikation für abgerollte Matrizen
                AA = a[ii*n:ii*n+n]
                for jj in range(n):
                    BB = b[BIdx+jj]
                    c[ii*n+jj] = AA.dot(BB)

            aWaveStar = Wa.dot(a)
            bWaveStar = Wb.dot(b)
            cWaveStar = aWaveStar*bWaveStar
            cWave = Wc.dot(cWaveStar)
            errC = cWave-c
            err2Norm = np.linalg.norm(errC, 2)
            if err2Norm > 0.001:
                return False
        return True  # fastLoop

    ret = fastLoop(n, nn, p, BIdx, c, Wa, Wb, Wc)
    return ret  # checkSolution


nn = 9
n = 3
p = 23

os.chdir("/Users/tillspaeth/Google Drive/V15DiffMap")
fileNames = os.listdir()
TfileNames = []

for f in fileNames:
    if ".npy" in f:
        TfileNames.append(f)
fileNames = TfileNames

#f = fileNames[0]

a = np.ones(nn)
b = np.ones(nn)

for f in fileNames:
    sol1 = np.load(f, allow_pickle=True)
    Wa1 = sol1[0]
    Wb1 = sol1[1]
    Wc1 = sol1[2]

    for ff in fileNames:
        if f != ff:
            sol2 = np.load(ff, allow_pickle=True)
            Wa2 = sol2[0]
            Wb2 = sol2[1]
            Wc2 = sol2[2]

            c = Wc2.dot(Wa2.dot(a)*Wb1.dot(b))

            if np.array_equal(np.ones(nn)*3, c):
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!")
                print(f, ff)
                print(c)
                break


# sol1 = np.load(f, allow_pickle=True)
# Wa1 = sol1[0]
# Wb1 = sol1[1]
# Wc1 = sol1[2]
#
# sol2 = np.load(ff, allow_pickle=True)
# Wa2 = sol2[0]
# Wb2 = sol2[1]
# Wc2 = sol2[2]
#
# Wa = Wa2
# Wb = Wb2
# Wc = Wc1
#
# a = np.ones(nn)
# b = np.ones(nn)
# c = Wc.dot(Wa.dot(a)*Wb.dot(b))
#
# c
