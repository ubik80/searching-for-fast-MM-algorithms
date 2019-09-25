# coding: utf8
import numpy as np
import backprop as biM
import multiprocessing as mp
import time
import uuid
from numba import jit
import checkSolution as cs
import os
np.set_printoptions(precision=2, suppress=True)


def PA(W):  # überschreibend
    W[0] = np.minimum(np.maximum(np.round(W[0]), -1.0), 1.0)
    W[1] = np.minimum(np.maximum(np.round(W[1]), -1.0), 1.0)
    W[2] = np.minimum(np.maximum(np.round(W[2]), -1.0), 1.0)
    return W  # PA


def PB(W):  # überschreibend
    nn = W[0].shape[1]
    p = W[0].shape[0]
    n = int(np.sqrt(nn))
    numOfProc = int(mp.cpu_count())*0+4
    WAs = [mp.RawArray('d', W[0].reshape(nn*p)) for i in range(numOfProc)]
    WBs = [mp.RawArray('d', W[1].reshape(nn*p)) for i in range(numOfProc)]
    WCs = [mp.RawArray('d', W[2].reshape(nn*p)) for i in range(numOfProc)]

    def backprop(WaMP, WbMP, WcMP, nn, p, i):
        Wa = np.frombuffer(WaMP, dtype='d').reshape([p, nn])
        Wb = np.frombuffer(WbMP, dtype='d').reshape([p, nn])
        Wc = np.frombuffer(WcMP, dtype='d').reshape([nn, p])
        biM.backpropRND(Wa, Wb, Wc, 30000000, 0.05, 0.01, i)
        return  # backprop

    procs = [mp.Process(target=backprop, args=(WAs[i], WBs[i], WCs[i], nn, p, i))
             for i in range(numOfProc)]
    for pp in procs:
        pp.start()
    for pp in procs:
        pp.join()

    WAs = [np.frombuffer(WAs[i], dtype='d').reshape([p, nn]) for i in range(numOfProc)]
    WBs = [np.frombuffer(WBs[i], dtype='d').reshape([p, nn]) for i in range(numOfProc)]
    WCs = [np.frombuffer(WCs[i], dtype='d').reshape([nn, p]) for i in range(numOfProc)]

    minDist = 999.9
    success = False
    for i in range(numOfProc):
        if np.isnan(WAs[i].any()) or np.isnan(WBs[i].any()) or np.isnan(WCs[i].any()):
            dist = 999.9
        else:
            dist = np.linalg.norm(WAs[i]-W[0], 2)**2+np.linalg.norm(WBs[i]-W[1],
                                                                    2)**2+np.linalg.norm(WCs[i]-W[2], 2)**2
        if dist < minDist:
            minDist = dist
            W = [WAs[i], WBs[i], WCs[i]]
            success = True
    return W  # PB


def initW(n, p):
    nn = int(n**2)
    success = -1
    while success < 0:
        print("initW - Initialisierung ...")
        Wa = np.random.rand(p*nn).reshape([p, nn])*2.0-1.0
        Wb = np.random.rand(p*nn).reshape([p, nn])*2.0-1.0
        Wc = np.random.rand(nn*p).reshape([nn, p])*2.0-1.0
        success = biM.backpropNue(Wa, Wb, Wc, 90000000, 0.01, 0.05, 0.1)
        print("initW - success=", success)
    print("initW - Initialisierung erfolgreich")
    return [Wa, Wb, Wc]  # initW


def diffMap(n, p, id):
    nn = int(n**2)
    print("n: ", n, "     p: ", p)
    seed = int(time.time())+int(uuid.uuid4())+id
    np.random.seed(seed % 135790)
    W = initW(n, p)
    pk = [W[0].copy()*0.0, W[1].copy()*0.0, W[2].copy()*0.0]
    qk = [W[0].copy()*0.0, W[1].copy()*0.0, W[2].copy()*0.0]
    iter = 0  # iteration
    maxNumIters = 30000
    diffs = []

    while True:
        WOld = W.copy()

        ################
        y = PA([W[0]+pk[0], W[1]+pk[1], W[2]+pk[2]])
        pk = [W[0]+pk[0]-y[0], W[1]+pk[1]-y[1], W[2]+pk[2]-y[2]]
        W = PB([y[0]+qk[0], y[1]+qk[1], y[2]+qk[2]])
        qk = [y[0]+qk[0]-W[0], y[1]+qk[1]-W[1], y[2]+qk[2]-W[2]]
        ################

        WDelta = [W[0]-WOld[0], W[1]-WOld[1], W[2]-WOld[2]]
        norm2Delta = np.linalg.norm(
            WDelta[0], 2)**2+np.linalg.norm(WDelta[1], 2)**2+np.linalg.norm(WDelta[2], 2)**2
        norm2Delta = np.sqrt(norm2Delta)
        diffs.append(norm2Delta)

        if norm2Delta < 0.0000000000000001:
            print(id, ", Lösung gefunden?")
            WW = PA(W.copy())
            check = cs.checkSolutionInt(WW)
            if check:
                print(id, ".... Lösung korrekt")
                np.save("solution_"+str(n)+"_"+str(p)+"_"+str(iter)+"_"+str(time.time())+"_proj_"+"V1",
                        [WW[0], WW[1], WW[2], diffs, iter])
                break
            else:
                print(id, ".... keine gültige Lösung")

        if iter % 1 == 0:
            print("---------------------------")
            print("Prozess:", id)
            print("Iter.:  ", iter)
            print("Delta:  ", norm2Delta)
        iter += 1
    return  # diffMap


if __name__ == '__main__':
    n = 2
    p = 7
    diffMap(n, p, id=0)


#
