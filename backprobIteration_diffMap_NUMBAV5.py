import checkSolution as cs
import sys
import time
from numba import jit
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(precision=4, suppress=True)


def backprop(n, p, numIters, Wa, Wb, Wc, limit=0.01, nue=0.1):
    nn = n**2
    nueC = nue
    nueAB = nue
    Wa = np.maximum(np.minimum(np.asarray(Wa), 99.9), -99.9)
    Wb = np.maximum(np.minimum(np.asarray(Wb), 99.9), -99.9)
    Wc = np.maximum(np.minimum(np.asarray(Wc), 99.9), -99.9)
    errHist = np.zeros(numIters, dtype=float)
    BIdx = np.array([k*n for k in range(n)])
    c = np.zeros(nn, dtype=float)
    bestA = np.zeros(nn, dtype=float)
    bestB = np.zeros(nn, dtype=float)
    bestErrC = np.zeros(nn, dtype=float)
    bestErrCStar = np.zeros(p, dtype=float)
    bestCWaveStar = np.zeros(p, dtype=float)

    @jit(nopython=True, nogil=True, cache=True)
    def iteration(n, nn, p, numIters, Wa, Wb, Wc, limit, nueAB, nueC, BIdx, errHist, c, bestA, bestB, bestErrC, bestErrCStar, bestCWaveStar):
        for i in range(numIters):
            maxErr = -999
            for ii in range(5):
                a = np.random.rand(nn)*2.0-1.0
                b = np.random.rand(nn)*2.0-1.0
                nA = np.linalg.norm(a, 2)
                nB = np.linalg.norm(b, 2)
                if np.abs(nA) > 0.1 and np.abs(nB) > 0.1:
                    a /= nA
                    b /= nB
                    for ii in range(n):  # Matrixmultiplikation für aufgerollte Matrizen
                        AA = a[ii*n:ii*n+n]
                        for jj in range(n):
                            BB = b[BIdx+jj]
                            c[ii*n+jj] = AA.dot(BB)
                    aWaveStar = Wa.dot(a)
                    bWaveStar = Wb.dot(b)
                    cWaveStar = aWaveStar*bWaveStar
                    cWave = Wc.dot(cWaveStar)
                    errC = cWave-c
                    errCStar = Wc.T.dot(errC)
                    errCStarNorm = np.linalg.norm(errCStar, 2)
                    if errCStarNorm > maxErr:
                        maxErr = errCStarNorm
                        bestA = a
                        bestB = b
                        bestErrC = errC
                        bestErrCStar = errCStar
                        bestCWaveStar = cWaveStar
                else:
                    ii -= 1
            errHist[i] = np.linalg.norm(bestErrC, 2)
            deltaWc = -nueC*np.outer(bestErrC, bestCWaveStar)
            deltaWa = -nueAB*np.outer(bestErrCStar*bWaveStar, bestA)
            deltaWb = -nueAB*np.outer(bestErrCStar*aWaveStar, bestB)
            Wa += deltaWa
            Wb += deltaWb
            Wc += deltaWc
            if i > 500 and np.max(errHist[i-500:i]) < limit:
                return True

            if i % 100000 == 0:
                print(errHist[i])

        return False  # iteration

    success = iteration(n, nn, p, numIters, Wa, Wb, Wc, limit,
                        nueAB, nueC, BIdx, errHist, c, bestA, bestB, bestErrC, bestErrCStar, bestCWaveStar)
    hasNANs = np.sum(np.isnan(Wa))+np.sum(np.isnan(Wb))+np.sum(np.isnan(Wc)) > 0
    if hasNANs:
        success = False

    return Wa, Wb, Wc, errHist, success  # findCalcRule


if __name__ == '__main__':
    n = 3
    nn = 9
    p = 23

    success = False
    while not success:
        Wa = np.random.rand(p, nn)
        Wb = np.random.rand(p, nn)
        Wc = np.random.rand(nn, p)
        M = np.ones(Wc.shape)

        start = time. time()
        Wa, Wb, Wc, errHist, success = backprop(n, p, 3000000, Wa, Wb, Wc, 0.01, 0.02)
        cs.checkSolutionReal([Wa, Wb, Wc])
        end = time. time()
        print("time: ", end - start)
        print(success)

    plt.rcParams.update({'font.size': 10})
    plt.plot(errHist)
    plt.xlabel('iteration')
    plt.ylabel('| deviation |')
    plt.axis([0, 1500000, 0, 1])

    if success:
        plt.savefig('backpropTrendXXX.png', dpi=300)
