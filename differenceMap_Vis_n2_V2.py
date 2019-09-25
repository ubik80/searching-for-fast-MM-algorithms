# coding: utf8
import numpy as np
import backprop as biM
import multiprocessing as mp
import time
import uuid
from numba import jit
import checkSolution as cs
np.set_printoptions(precision=2, suppress=True)


def PA(W):  # no copy / overwriting !
    W[0] = np.minimum(np.maximum(np.round(W[0]), -1.0), 1.0)
    W[1] = np.minimum(np.maximum(np.round(W[1]), -1.0), 1.0)
    W[2] = np.minimum(np.maximum(np.round(W[2]), -1.0), 1.0)
    return W  # PA


def PB(W):  # copy / not overwriting
    minDist = 99999.9
    solFound = False
    WaRet = []
    WbRet = []
    WcRet = []
    for tries in range(3):
        Wa = W[0].copy()
        Wb = W[1].copy()
        Wc = W[2].copy()
        success = biM.backprop(Wa, Wb, Wc, 3000000, 0.1, 0.01)
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


@jit(nopython=True, nogil=True, cache=True)
def rankWeight(i, j, Wa, Wb, Wc, baseDev, matSel, WaiWci, WbiWci, WajWbj, ei):
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
    Wa, Wb, Wc = np.load("roundedStartVals_n2.npy", allow_pickle=True)
    return [Wa, Wb, Wc]
    nn = int(n**2)
    success = -1
    while success < 0:
        Wa = np.random.rand(p*nn).reshape([p, nn])*2.0-1.0
        Wb = np.random.rand(p*nn).reshape([p, nn])*2.0-1.0
        Wc = np.random.rand(nn*p).reshape([nn, p])*2.0-1.0
        success = biM.backprop(Wa, Wb, Wc, 3000000, 0.1, 0.01)
    MA = np.ones(Wa.shape)
    MB = np.ones(Wb.shape)
    MC = np.ones(Wc.shape)
    TA = np.ones(Wa.shape)
    TB = np.ones(Wb.shape)
    TC = np.ones(Wc.shape)
    ei = np.zeros(nn, dtype=float)
    rounds = 0
    while True:
        i, j, err, matSel = findWeight(Wa, Wb, Wc, TA, TB, TC, ei)
        if i < 0:
            break
        WaT = Wa.copy()
        WbT = Wb.copy()
        WcT = Wc.copy()
        if matSel == 0:
            TA[i, j] = 0
            MA[i, j] = 0
            WaT[i, j] = np.minimum(np.maximum(np.round(WaT[i, j]), -1), 1)
        if matSel == 1:
            TB[i, j] = 0
            MB[i, j] = 0
            WbT[i, j] = np.minimum(np.maximum(np.round(WbT[i, j]), -1), 1)
        if matSel == 2:
            TC[i, j] = 0
            MC[i, j] = 0
            WcT[i, j] = np.minimum(np.maximum(np.round(WcT[i, j]), -1), 1)
        success = biM.backpropM(WaT, WbT, WcT, MA, MB, MC, 100000, 0.1, 0.01)
        if success > 0:
            Wa = WaT
            Wb = WbT
            Wc = WcT
            rounds += 1
        else:
            if matSel == 0:
                MA[i, j] = 1
            if matSel == 1:
                MB[i, j] = 1
            if matSel == 2:
                MC[i, j] = 1
    #np.save("roundedStartVals_n2", [Wa, Wb, Wc])
    return [Wa, Wb, Wc]  # roundInit


def diffMap(id, mutex):
    p = 7
    n = 2
    nn = int(n**2)
    seed = int(time.time())+int(uuid.uuid4())+id
    np.random.seed(seed % 135790)
    W = roundInit(n, p)
    i = 0  # iteration
    numOfTries = 0
    diffs = []
    jumps = []  # indices of jumps
    heights = []
    numOfJumps = 0
    maxNumIters = 1000
    minNumIters = 50
    jumpFactor = 0.3
    minDiff = 99999
    maxDiff = -99999
    inBand = 0
    bandWith = 10

    WHist = [[W[0], W[1], W[2], False, -1]]

    while True:
        s = False
        while not s:
            PBx, s = PB(W)  # not overwriting
            if not s:
                print("   Prz: ", id, " BP failed -> reset")
                seed = int(time.time())+int(uuid.uuid4())+id
                np.random.seed(seed % 135745)
                W = roundInit(n, p)
                numOfTries += 1
                diffs = []
                WHist = [[W[0], W[1], W[2], False, -1]]
                jumps = []
                heights = []
                numOfJumps = 0
                minDiff = 99999
                maxDiff = -99999
                inBand = 0
                i = 0
        PAy = PA([2.0*PBx[0]-W[0], 2.0*PBx[1]-W[1], 2.0*PBx[2]-W[2]])
        delta = [PAy[0]-PBx[0], PAy[1]-PBx[1], PAy[2]-PBx[2]]
        W = [W[0]+delta[0], W[1]+delta[1], W[2]+delta[2]]
        norm2Delta = np.linalg.norm(
            delta[0], 2)**2+np.linalg.norm(delta[1], 2)**2+np.linalg.norm(delta[2], 2)**2
        norm2Delta = np.sqrt(norm2Delta)
        diffs.append(norm2Delta)

        WHist.append([W[0], W[1], W[2], False, norm2Delta])

        if norm2Delta < 0.5 and i > minNumIters:  # and numOfJumps > 5:
            mutex.acquire()
            print(id, ", Lösung gefunden?")
            WW = PA(PB(W)[0])  # PA is overwriting, but PB is not
            c2 = cs.checkSolutionInt(WW)
            if c2:
                print(id, ".... Lösung korrekt")
                mutex.release()
                WHist.append([W[0], W[1], W[2], False, -2])
                np.save("n2_vis_V2_4",
                        [WW[0], WW[1], WW[2], diffs, WHist])
                return
            else:
                print(id, ".... keine gültige Lösung")
                mutex.release()
        # elif norm2Delta < 0.5 and i <= minNumIters:
        #     print("Trend zu kurz")
        #     W = roundInit(n, p)
        #     numOfTries = 0
        #     diffs = []
        #     WHist = []
        #     jumps = []
        #     heights = []
        #     numOfJumps = 0
        #     minDiff = 99999
        #     maxDiff = -99999
        #     inBand = 0
        #     i = 0

        mutex.acquire()
        if i % 100 == 0 and i > 0:
            print("---------------------------")
            print("Prozess:", id)
            print("Iter.:  ", i)
            print("Delta:  ", norm2Delta)
            print("Jumps:  ", numOfJumps)
        if i > maxNumIters:  # and norm2Delta > 3.0:
            print(i, " cycles -> Reset")
            print("tries:", numOfTries)
        mutex.release()

        if len(diffs) > bandWith:
            minDiff = min(diffs[max(len(diffs)-bandWith, 0): len(diffs)])
            maxDiff = max(diffs[max(len(diffs)-bandWith, 0): len(diffs)])
        if norm2Delta > minDiff and norm2Delta < maxDiff:
            inBand += 1
        else:
            inBand = 0
        if inBand > 6:
            W[0] += (np.random.rand(p*nn).reshape([p, nn])*2.0-1.0)*jumpFactor
            W[1] += (np.random.rand(p*nn).reshape([p, nn])*2.0-1.0)*jumpFactor
            W[2] += (np.random.rand(p*nn).reshape([nn, p])*2.0-1.0)*jumpFactor
            jumps.append(i)
            heights.append(1)
            numOfJumps += 1
            WHist.append([W[0], W[1], W[2], True, 0])
        if i > maxNumIters:  # and norm2Delta > 3.0:
            seed = int(time.time())+int(uuid.uuid4())+id
            np.random.seed(seed % 135790)
            W = roundInit(n, p)
            numOfTries += 1
            diffs = []
            WHist = [[W[0], W[1], W[2], False, -1]]
            jumps = []
            heights = []
            numOfJumps = 0
            minDiff = 99999
            maxDiff = -99999
            inBand = 0
            i = 0
        i += 1

    return  # diffMap


if __name__ == '__main__':
    numOfProc = int(mp.cpu_count())*0+1
    print("Anzahl Prozessoren: ", numOfProc)

    mutex = mp.Lock()
    procs = [mp.Process(target=diffMap, args=(i, mutex)) for i in range(numOfProc)]

    for pp in procs:
        pp.start()
    for pp in procs:
        pp.join()


#