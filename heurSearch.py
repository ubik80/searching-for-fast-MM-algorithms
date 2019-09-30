# coding: utf8
import os
import numpy as np
from numba import jit
import smallStuff as sm
import backprop as bp
import multiprocessing as mp
import time
import uuid
np.set_printoptions(precision=2, suppress=True)


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


@jit(nopython=True, nogil=True, cache=True)
def resets(MA, MB, MC):
    nn = MA.shape[1]
    p = MA.shape[0]
    resets = np.random.randint(np.floor((3*p*nn-np.sum(MA)-np.sum(MB)-np.sum(MC))-1))+1
    for r in range(resets):
        ss = False
        while not ss:
            ii = np.random.randint(p)
            jj = np.random.randint(3*nn)
            if jj < nn:
                if MA[ii, jj] == 0.0:
                    MA[ii, jj] = 1.0
                    ss = True
            elif jj < 2*nn:
                jj -= nn
                if MB[ii, jj] == 0.0:
                    MB[ii, jj] = 1.0
                    ss = True
            else:
                jj -= 2*nn
                if MC[jj, ii] == 0.0:
                    MC[jj, ii] = 1.0
                    ss = True
    return


def intSolutionSearch(n, p, maxTries, maxNumIters,
                      bestWa, bestWb, bestWc, bestMA, bestMB, bestMC, mutex, finished, id):
    seed = int(time.time())+int(uuid.uuid4())+id
    seed = seed % 135790
    np.random.seed(seed)
    nn = int(n**2)
    ei = np.zeros(nn, dtype=float)
    success = -1
    while success < 0:
        Wa = np.random.rand(p*nn).reshape([p, nn])*2.0-1.0
        Wb = np.random.rand(p*nn).reshape([p, nn])*2.0-1.0
        Wc = np.random.rand(nn*p).reshape([nn, p])*2.0-1.0
        success = bp.backpropNueRND(Wa, Wb, Wc, 90000000, 0.01, 0.05, 0.1, 0)
    MA = np.ones(Wa.shape)
    MB = np.ones(Wb.shape)
    MC = np.ones(Wc.shape)
    iterFact = 1

    mutex.acquire()
    bestWa_np = np.frombuffer(bestWa, dtype='d').reshape([p, nn])
    bestWb_np = np.frombuffer(bestWb, dtype='d').reshape([p, nn])
    bestWc_np = np.frombuffer(bestWc, dtype='d').reshape([nn, p])
    bestMA_np = np.frombuffer(bestMA, dtype='d').reshape([p, nn])
    bestMB_np = np.frombuffer(bestMB, dtype='d').reshape([p, nn])
    bestMC_np = np.frombuffer(bestMC, dtype='d').reshape([nn, p])
    oldBest = np.sum(bestMA_np)+np.sum(bestMB_np)+np.sum(bestMC_np)
    print("Prozess ", id, " initialisiert")
    mutex.release()

    for tries in range(maxTries):
        i, j, err, matSel = findWeight(Wa, Wb, Wc, MA, MB, MC, ei)
        if matSel == 0:
            MA[i, j] = 0.0
            Wa[i, j] = float(min(max(round(Wa[i, j]), -1.0), 1.0))
        elif matSel == 1:
            MB[i, j] = 0.0
            Wb[i, j] = float(min(max(round(Wb[i, j]), -1.0), 1.0))
        else:
            MC[i, j] = 0.0
            Wc[i, j] = float(min(max(round(Wc[i, j]), -1.0), 1.0))

        success = bp.backpropM(Wa, Wb, Wc, MA, MB, MC, maxNumIters*iterFact, 0.1, 0.1, 0.01)
        iterFact = 1

        if not success:
            if matSel == 0:
                MA[i, j] = 1.0
            elif matSel == 1:
                MB[i, j] = 1.0
            else:
                MC[i, j] = 1.0
            resets(MA, MB, MC)
            iterFact = 4

        mutex.acquire()
        if finished.value == 1:
            print("Prozess: ", str(id), " terminate ...")
            mutex.release()
            return
        print("-------------------------------------")
        print("Prozess: ", str(id))
        print("Durchgang: ", str(tries))
        print("Position: ", str([i, j]), " Wa/Wb/Wc: ", str(matSel))
        numOfMltpls = np.sum(bestMA_np)+np.sum(bestMB_np)+np.sum(bestMC_np)
        if numOfMltpls < oldBest:
            oldBest = numOfMltpls
            print("Innovation - Daten tauschen")
            Wa = bestWa_np.copy()
            Wb = bestWb_np.copy()
            Wc = bestWc_np.copy()
            MA = bestMA_np.copy()
            MB = bestMB_np.copy()
            MC = bestMC_np.copy()
        elif not success:
            print("keine Lösung gefunden - resets")
        else:  # success
            print("Lösung gefunden")
            if numOfMltpls > np.sum(MA)+np.sum(MB)+np.sum(MC):
                print("beste Lösung bisher")
                np.copyto(bestWa_np, Wa)
                np.copyto(bestWb_np, Wb)
                np.copyto(bestWc_np, Wc)
                np.copyto(bestMA_np, MA)
                np.copyto(bestMB_np, MB)
                np.copyto(bestMC_np, MC)
                numOfMltpls = np.sum(bestMA_np)+np.sum(bestMB_np)+np.sum(bestMC_np)
                oldBest = numOfMltpls
        if np.sum(np.isnan(Wa)) > 0 or np.sum(np.isnan(Wb)) > 0 or np.sum(np.isnan(Wc)) > 0:
            print("NAN")
            if id == 0:
                print("Daten tauschen")
            Wa = bestWa_np.copy()
            Wb = bestWb_np.copy()
            Wc = bestWc_np.copy()
            MA = bestMA_np.copy()
            MB = bestMB_np.copy()
            MC = bestMC_np.copy()

        sm.printMM(MA, MB, -999, -999, -999, -999)
        print('')
        sm.printM(MC, i, j)
        print(str(int(np.sum(MA)+np.sum(MB)+np.sum(MC)))+" ("+str(int(numOfMltpls))+")")

        if (numOfMltpls == 0):
            np.save("solution", [Wa, Wb, Wc])
            print("in Datei geschrieben")
            finished.value = 1
            mutex.release()
            return
        mutex.release()
    return


if __name__ == '__main__':
    numOfProc = int(mp.cpu_count())
    print("Anzahl Prozessoren: ", numOfProc)

    n = 2
    p = 7

    nn = int(n**2)

    mutex = mp.Lock()
    finished = mp.Value('i', 0)
    bestWa = mp.RawArray('d', np.zeros(p*nn, dtype=float))
    bestWb = mp.RawArray('d', np.zeros(p*nn, dtype=float))
    bestWc = mp.RawArray('d', np.zeros(p*nn, dtype=float))
    bestMA = mp.RawArray('d', np.ones(p*nn, dtype=float))
    bestMB = mp.RawArray('d', np.ones(p*nn, dtype=float))
    bestMC = mp.RawArray('d', np.ones(p*nn, dtype=float))

    procs = [mp.Process(target=intSolutionSearch,
                        args=(n, p, 50000, 1000000, bestWa, bestWb, bestWc, bestMA, bestMB, bestMC, mutex, finished, i))
             for i in range(numOfProc)]

    for pp in procs:
        pp.start()
    for pp in procs:
        pp.join()
