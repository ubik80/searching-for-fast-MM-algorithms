#import imp
import numpy as np
import matplotlib.pyplot as plt
from numba import jit
import os
import backprop as biM
np.set_printoptions(precision=2, suppress=True)

# imp.find_module('murmurhash')


def checkSolution(W):
    p = W[0].shape[0]
    nn = W[0].shape[1]
    n = int(np.sqrt(nn))
    BIdx = np.array([k*n for k in range(n)])
    c = np.zeros(nn, dtype=float)
    # Wa = np.maximum(np.minimum(np.round(W[0]), 1.0), -1.0)
    # Wb = np.maximum(np.minimum(np.round(W[1]), 1.0), -1.0)
    # Wc = np.maximum(np.minimum(np.round(W[2]), 1.0), -1.0)

    @jit(nopython=True, nogil=True, cache=True)
    def fastLoop(n, nn, p, BIdx, c, Wa, Wb, Wc):
        for i in range(1000):
            a = np.random.rand(nn)*2.0-1.0
            b = np.random.rand(nn)*2.0-1.0
            nA = np.linalg.norm(a, 2)
            nB = np.linalg.norm(b, 2)

            if np.abs(nA) > 0.1 and np.abs(nB) > 0.1:
                a /= nA
                b /= nB

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
                if err2Norm > 0.0001:
                    print(err2Norm)
                    return False
            else:
                i -= 1
        return True  # fastLoop
    ret = fastLoop(n, nn, p, BIdx, c, Wa, Wb, Wc)
    return ret  # checkSolution

# sol = np.load(
#     "roundedStartValues_5_105_temp.npy", allow_pickle=True)
# Wa = sol[0]
# Wb = sol[1]
# Wc = sol[2]
#
# np.max(Wc)
# np.min(Wc)
#
# min = -1.0
# max = -min
#
# Wa = np.minimum(np.maximum(Wa, min), max)
# Wb = np.minimum(np.maximum(Wb, min), max)
# Wc = np.minimum(np.maximum(Wc, min), max)
#
# for i in range(10):
#     success = biM.backprop(Wa, Wb, Wc, 3000000, 0.01)
# print(success)
#
# MAB = np.ones(Wa.shape)
# MC = np.ones(Wc.shape)
# np.save("roundedStartValues_5_105_temp", [Wa, Wb, Wc, MAB, MAB, MC, MAB, MAB, MC])
#


# sol = np.load(
#     "solution_n5_temp_7500.npy", allow_pickle=True)
# Wa, Wb, Wc = sol[0], sol[1], sol[2]
#
# for i in range(20):
#     Wa = np.vstack((Wa, np.zeros(25, dtype=float)))
#     Wb = np.vstack((Wb, np.zeros(25, dtype=float)))
#     Wc = np.vstack((Wc.T, np.zeros(25, dtype=float)))
#     Wc = Wc.T

n = 5
nn = 25
p = 105

# Wa = np.random.rand(p*nn).reshape([p, nn])*2.0-1.0
# Wb = np.random.rand(p*nn).reshape([p, nn])*2.0-1.0
# Wc = np.random.rand(nn*p).reshape([nn, p])*2.0-1.0

success = False
while not success:
    Wa = np.random.rand(p*nn).reshape([p, nn])*2.0-1.0
    Wb = np.random.rand(p*nn).reshape([p, nn])*2.0-1.0
    Wc = np.random.rand(nn*p).reshape([nn, p])*2.0-1.0
    iters = biM.backprop(Wa, Wb, Wc, 10000000, 0.1, 0.00001)  # 0.07
    success = checkSolution([Wa, Wb, Wc])
    if np.isnan(np.min(Wa)):
        success = False

np.save("roundedStartValues_5_105_V2", [Wa, Wb, Wc])
#
