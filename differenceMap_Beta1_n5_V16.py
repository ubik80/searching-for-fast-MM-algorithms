# coding: utf8
import numpy as np
import backprop as biM
import bloomFilter as bf
import multiprocessing as mp
import time
import uuid
from numba import jit
np.set_printoptions(precision=2, suppress=True)


def PA(W):  # no copy / overwriting !
    W[0] = np.minimum(np.maximum(np.round(W[0]), -1.0), 1.0)
    W[1] = np.minimum(np.maximum(np.round(W[1]), -1.0), 1.0)
    W[2] = np.minimum(np.maximum(np.round(W[2]), -1.0), 1.0)
    return W  # PA


def PB(W):  # copy / not overwriting
    p = W[0].shape[0]
    nn = W[0].shape[1]
    n = int(np.sqrt(nn))
    minDist = 99999.9
    solFound = False
    WaRet = []
    WbRet = []
    WcRet = []
    for tries in range(5):
        Wa = W[0].copy()
        Wb = W[1].copy()
        Wc = W[2].copy()
        success = biM.backprop(Wa, Wb, Wc, 30000000, 0.1)
        if success > 0:
            dist = np.linalg.norm(Wa-W[0], 2)**2+np.linalg.norm(Wb-W[1],
                                                                2)**2+np.linalg.norm(Wc-W[2], 2)**2
            if dist < minDist:
                solFound = True
                minDist = dist
                WaRet = Wa
                WbRet = Wb
                WcRet = Wc
    if solFound:
        return [WaRet, WbRet, WcRet], True
    return W, False  # PB


def checkSolution(W):
    p = W[0].shape[0]
    nn = W[0].shape[1]
    n = int(np.sqrt(nn))
    BIdx = np.array([k*n for k in range(n)])
    c = np.zeros(nn, dtype=float)
    Wa = np.maximum(np.minimum(np.round(W[0]), 1.0), -1.0)
    Wb = np.maximum(np.minimum(np.round(W[1]), 1.0), -1.0)
    Wc = np.maximum(np.minimum(np.round(W[2]), 1.0), -1.0)

    @jit(nopython=True, nogil=True, cache=True)
    def fastLoop(n, nn, p, BIdx, c, Wa, Wb, Wc):
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
                if err2Norm > 0.001:
                    return False
            else:
                i -= 1
        return True  # fastLoop
    ret = fastLoop(n, nn, p, BIdx, c, Wa, Wb, Wc)
    return ret  # checkSolution


@jit(nopython=True, nogil=True, cache=True)
def rankWeight(i, j, Wa, Wb, Wc, baseDev, matSel, WaiWci, WbiWci, WajWbj, ei):
    nn = Wa.shape[1]
    p = Wa.shape[0]
    n = int(np.sqrt(nn))
    if matSel == 0:
        deltaVec = (np.round(Wa[i, j])-Wa[i, j])*WbiWci[i]
    elif matSel == 1:
        deltaVec = (np.round(Wb[i, j])-Wb[i, j])*WaiWci[i]
    else:
        ei *= 0.0
        ei[i] = 1.0
        deltaVec = (np.round(Wc[i, j])-Wc[i, j])*WajWbj[j]*ei
    ret = np.linalg.norm(deltaVec+baseDev, 2)
    return ret  # rankWeight


@jit(nopython=True, nogil=True, cache=True)
def findWeight(Wa, Wb, Wc, MA, MB, MC, ei):
    nn = Wa.shape[1]
    p = Wa.shape[0]
    n = int(np.sqrt(nn))
    bestI = -999
    bestJ = -999
    bestErr = 999999
    matSel = -9
    a = np.ones(nn)
    b = np.ones(nn)
    baseDev = Wc.dot(Wa.dot(a)*Wb.dot(b))-np.ones(nn)*n
    WaiWci = [np.sum(Wa[i, :])*Wc[:, i] for i in range(p)]
    WbiWci = [np.sum(Wb[i, :])*Wc[:, i] for i in range(p)]
    WajWbj = [np.sum(Wa[j, :])*np.sum(Wb[j, :]) for j in range(p)]
    for i in range(p):
        for j in range(nn):
            if MA[i, j] == 1:
                err = rankWeight(i, j, Wa, Wb, Wc, baseDev, 0, WaiWci, WbiWci, WajWbj, ei)
                if err < bestErr:
                    bestErr = err
                    bestI = i
                    bestJ = j
                    matSel = 0
    for i in range(p):
        for j in range(nn):
            if MB[i, j] == 1:
                err = rankWeight(i, j, Wa, Wb, Wc, baseDev, 1, WaiWci, WbiWci, WajWbj, ei)
                if err < bestErr:
                    bestErr = err
                    bestI = i
                    bestJ = j
                    matSel = 1
    for i in range(nn):
        for j in range(p):
            if MC[i, j] == 1:
                err = rankWeight(i, j, Wa, Wb, Wc, baseDev, 2, WaiWci, WbiWci, WajWbj, ei)
                if err < bestErr:
                    bestErr = err
                    bestI = i
                    bestJ = j
                    matSel = 2
    return bestI, bestJ, bestErr, matSel  # findWeight


def roundInit(n, p):
    Wa, Wb, Wc, MA, MB, MC, TA, TB, TC = np.load(
        "roundedStartValues_5_105_temp.npy", allow_pickle=True)
    print("start values loaded")
    return [Wa, Wb, Wc]  # roundInit


def diffMap(id, mutex):
    p = 105
    n = 5
    nn = int(n**2)

    seed = int(time.time())+int(uuid.uuid4())+id
    np.random.seed(seed % 135790)
    W = roundInit(n, p)
    BFs = [bf.bloomFilter(2*nn*p, 0.00001) for b in range(20)]
    i = 0  # iteration
    numOfCycles = 0
    numOfTries = 0
    diffs = []
    jumps = []  # indices of jumps
    heights = []  # multpliers (cyclCnt) of jumps

    jumpFactor = 0.0125
    bloomOn = True

    while True:
        s = False
        while not s:
            PBx, s = PB(W)  # not overwriting
            if not s:
                print("   Prz: ", id, " BP failed -> reset")
                seed = int(time.time())+int(uuid.uuid4())+id
                np.random.seed(seed % 135745)
                W = roundInit(n, p)
                BFs = [bf.bloomFilter(2*nn*p, 0.00001) for b in range(20)]
                i = 0
                numOfCycles = 0
                numOfTries += 1
                diffs = []
                jumps = []
                heights = []

        PAy = PA([2.0*PBx[0]-W[0], 2.0*PBx[1]-W[1], 2.0*PBx[2]-W[2]])
        delta = [PAy[0]-PBx[0], PAy[1]-PBx[1], PAy[2]-PBx[2]]
        W = [W[0]+delta[0], W[1]+delta[1], W[2]+delta[2]]
        norm2Delta = np.linalg.norm(
            delta[0], 2)**2+np.linalg.norm(delta[1], 2)**2+np.linalg.norm(delta[2], 2)**2
        norm2Delta = np.sqrt(norm2Delta)
        diffs.append(norm2Delta)

        cyclCnt = 0
        for b in range(len(BFs)):
            if not BFs[b].store(PAy):
                cyclCnt = b
                break

        if norm2Delta < 0.5:
            print("Lösung gefunden?")
            WW = PA(PB(W)[0])  # PA is overwriting, but PB is not
            c2 = checkSolution(WW)
            if c2:
                np.save("solution_"+str(n)+"_"+str(i)+"_"+str(time.time())+"_"+"V16",
                        [WW[0], WW[1], WW[2], jumpFactor, diffs, jumps, heights, i, numOfCycles, numOfTries, bloomOn])
                print(".... Lösung korrekt")
                W = roundInit(n, p)
                BFs = [bf.bloomFilter(2*nn*p, 0.00001) for b in range(20)]
                numOfCycles = 0
                numOfTries = 0
                diffs = []
                jumps = []
                heights = []
                i = 0
            else:
                print(".... keine gültige Lösung")

        mutex.acquire()
        if i % 1 == 0:
            print("---------------------------")
            print("Prozess:", id)
            print("Iter.:  ", i)
            print("Delta:  ", norm2Delta)
        if cyclCnt > 0:
            #print("**** Zyklus entdeckt! *****")
            print("**** cyclCnt: ", cyclCnt)
        if i > 20000 and norm2Delta > 3.0:
            print(i, " cycles -> Reset")
            print("tries:", numOfTries)
        mutex.release()

        if cyclCnt > 0 and bloomOn:
            W[0] += (np.random.rand(p*nn).reshape([p, nn])*2.0-1.0)*cyclCnt*jumpFactor
            W[1] += (np.random.rand(p*nn).reshape([p, nn])*2.0-1.0)*cyclCnt*jumpFactor
            W[2] += (np.random.rand(p*nn).reshape([nn, p])*2.0-1.0)*cyclCnt*jumpFactor
        if cyclCnt > 0:
            jumps.append(i)
            heights.append(cyclCnt)
            numOfCycles += 1
        if i > 20000 and norm2Delta > 3.0:
            seed = int(time.time())+int(uuid.uuid4())+id
            np.random.seed(seed % 135790)
            W = roundInit(n, p)
            BFs = [bf.bloomFilter(2*nn*p, 0.00001) for b in range(20)]
            numOfCycles = 0
            numOfTries += 1
            diffs = []
            jumps = []
            heights = []
            i = 0
        i += 1
    return  # diffMap


if __name__ == '__main__':
    numOfProc = int(mp.cpu_count())*0+4
    print("Anzahl Prozessoren: ", numOfProc)

    mutex = mp.Lock()
    procs = [mp.Process(target=diffMap, args=(i, mutex)) for i in range(numOfProc)]

    for pp in procs:
        pp.start()
    for pp in procs:
        pp.join()


#
