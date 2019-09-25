import os
from numba import jit
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(precision=4, suppress=True)


def checkResult(_Wa, _Wb, _Wc, prints=False):
    Wa, Wb, Wc = np.matrix(_Wa, dtype=int), np.matrix(_Wb, dtype=int), np.matrix(_Wc, dtype=int)
    ret = 0
    p = Wa.shape[0]
    nn = Wa.shape[1]
    n = int(np.sqrt(nn))

    cStar = np.zeros([p, nn**2], dtype=int)
    cWave = np.zeros([nn, nn**2], dtype=int)

    for i in range(p):
        cStar[i, :] = [Wa[i, kk]*Wb[i, k] for kk in range(nn) for k in range(nn)]

    cWave = Wc.dot(cStar)

    cStarTxt = ["a("+str(int((kk-kk % n)/n))+","+str(int(kk % n))+")"
                + "*b("+str(int((k-k % n)/n))+","+str(int(k % n))+")"
                for kk in range(nn) for k in range(nn)]
    cWaveTxt = np.chararray((n, n), unicode=True, itemsize=256)
    cTxt = np.chararray((n, n), unicode=True, itemsize=256)

    for i in range(n):
        for j in range(n):
            cWaveTxt[i, j] = ''
            for k in range(nn**2):
                if cWave[i*n+j, k] == 1:
                    cWaveTxt[i, j] += cStarTxt[k]+"+"
                elif not cWave[i*n+j, k] == 0:
                    cWaveTxt[i, j] += str(cWave[i*n+j, k])+"*"+cStarTxt[k]+"+"
            cWaveTxt[i, j] = cWaveTxt[i, j][0:max(len(cWaveTxt[i, j])-1, 0)]

    for i in range(n):
        for j in range(n):
            cTxt[i, j] = ''
            for k in range(n):
                cTxt[i, j] += "a("+str(i)+","+str(k)+")*b("+str(k)+","+str(j)+")+"
            cTxt[i, j] = cTxt[i, j][0:max(len(cTxt[i, j])-1, 0)]

    for i in range(n):
        for j in range(n):
            if not cTxt[i, j] == cWaveTxt[i, j]:
                ret += 1
                if prints:
                    print("Fehler in C("+str(i)+","+str(j)+"):")
                    print("C_correct = "+cTxt[i, j])
                    print("C_learned = "+cWaveTxt[i, j])
    return ret


def checkSolutionReal(W, limit):
    p = W[0].shape[0]
    nn = W[0].shape[1]
    n = int(np.sqrt(nn))
    BIdx = np.array([k*n for k in range(n)])
    c = np.zeros(nn, dtype=float)
    Wa = W[0]
    Wb = W[1]
    Wc = W[2]

    @jit(nopython=True, nogil=True, cache=True)
    def fastLoop(n, nn, p, BIdx, c, Wa, Wb, Wc, limit):
        for i in range(100):
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
                if err2Norm > limit:
                    return False
            else:
                i -= 1
        return True  # fastLoop
    ret = fastLoop(n, nn, p, BIdx, c, Wa, Wb, Wc, limit)
    return ret  # checkSolution


def checkSolutionInt(W, limit):
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
    def fastLoop(n, nn, p, BIdx, c, Wa, Wb, Wc, limit):
        for i in range(100):
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
                if err2Norm > limit:
                    return False
            else:
                i -= 1
        return True  # fastLoop
    ret = fastLoop(n, nn, p, BIdx, c, Wa, Wb, Wc, limit)
    return ret  # checkSolution


fc = np.load("solution_5_117_14794_1569170014.8737447_V25.npy", allow_pickle=True)
Wa, Wb, Wc = fc[0], fc[1], fc[2]
# Wc[4,4]+=0.000000001
# Wc[4,4]=-1

limit = 0.0000000000001
checkSolutionReal([Wa, Wb, Wc], limit)
checkSolutionInt([Wa, Wb, Wc], limit)
checkResult(Wa, Wb, Wc, True)