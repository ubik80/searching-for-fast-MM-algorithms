# coding: utf8
import numpy as np
import backprop as biM
import bloomFilter as bf
#import multiprocessing as mp
import time
import uuid
#import line_profiler
from numba import jit
np.set_printoptions(precision=2, suppress=True)

@profile
def PA(W):  # no copy / overwriting !
    W[0] = np.minimum(np.maximum(np.round(W[0]), -1.0), 1.0)
    W[1] = np.minimum(np.maximum(np.round(W[1]), -1.0), 1.0)
    W[2] = np.minimum(np.maximum(np.round(W[2]), -1.0), 1.0)
    return W  # PA

@profile
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

@profile
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

@profile
def roundInit(n, p):
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
            # print("o",end='',flush=True)
        else:
            if matSel == 0:
                MA[i, j] = 1
            if matSel == 1:
                MB[i, j] = 1
            if matSel == 2:
                MC[i, j] = 1
            # print("x",end='',flush=True)
    print("roundInit-Rundungen: ", str(rounds))
    return [Wa, Wb, Wc]  # roundInit

@profile
def diffMap(id):
    p = 23
    n = 3
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
                np.save("solution_"+str(n)+"_"+str(i)+"_"+str(time.time())+"_"+"V15",
                        [WW[0], WW[1], WW[2], jumpFactor, diffs, jumps, heights, i, numOfCycles, numOfTries, bloomOn])
                print(".... Lösung korrekt")
                return
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

        #mutex.acquire()
        if i % 100 == 0:
            print("---------------------------")
            print("Prozess:", id)
            print("Iter.:  ", i)
            print("Delta:  ", norm2Delta)
        if cyclCnt > 0:
            #print("**** Zyklus entdeckt! *****")
            print("**** cyclCnt: ", cyclCnt)
        if i > 2000 and norm2Delta > 3.0:
            print(i, " cycles -> Reset")
            print("tries:", numOfTries)
        #mutex.release()

        if cyclCnt > 0 and bloomOn:
            W[0] += (np.random.rand(p*nn).reshape([p, nn])*2.0-1.0)*cyclCnt*jumpFactor
            W[1] += (np.random.rand(p*nn).reshape([p, nn])*2.0-1.0)*cyclCnt*jumpFactor
            W[2] += (np.random.rand(p*nn).reshape([nn, p])*2.0-1.0)*cyclCnt*jumpFactor
        if cyclCnt > 0:
            jumps.append(i)
            heights.append(cyclCnt)
            numOfCycles += 1
        if i > 2000 and norm2Delta > 3.0:
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
#    numOfProc = int(mp.cpu_count())
#    print("Anzahl Prozessoren: ", numOfProc)
#
    #mutex = mp.Lock()
#    procs = [mp.Process(target=diffMap, args=(i, mutex)) for i in range(numOfProc)]
#
#    for pp in procs:
#        pp.start()
#    for pp in procs:
#        pp.join()
        
        
    diffMap(0)
    
    print("finished")
    
#
#Function: PA at line 12
#
#Line #      Hits         Time  Per Hit   % Time  Line Contents
#==============================================================
#    12                                           @profile
#    13                                           def PA(W):  # no copy / overwriting !
#    14     19829     258524.0     13.0     49.3      W[0] = np.minimum(np.maximum(np.round(W[0]), -1.0), 1.0)
#    15     19829     139044.0      7.0     26.5      W[1] = np.minimum(np.maximum(np.round(W[1]), -1.0), 1.0)
#    16     19829     120511.0      6.1     23.0      W[2] = np.minimum(np.maximum(np.round(W[2]), -1.0), 1.0)
#    17     19829       6803.0      0.3      1.3      return W  # PA
#
#Total time: 2646.89 s
#File: /Users/tillspaeth/Desktop/Masterarbeit/searching-for-fast-MM-algorithms/differenceMap_Beta1_n3_V15_Profiling.py
#Function: PB at line 19
#
#Line #      Hits         Time  Per Hit   % Time  Line Contents
#==============================================================
#    19                                           @profile
#    20                                           def PB(W):  # copy / not overwriting
#    21     19829      11962.0      0.6      0.0      minDist = 99999.9
#    22     19829       9176.0      0.5      0.0      solFound = False
#    23     19829       9370.0      0.5      0.0      WaRet = []
#    24     19829       8422.0      0.4      0.0      WbRet = []
#    25     19829       8444.0      0.4      0.0      WcRet = []
#    26     79316      50899.0      0.6      0.0      for tries in range(3):
#    27     59487     147248.0      2.5      0.0          Wa = W[0].copy()
#    28     59487      81199.0      1.4      0.0          Wb = W[1].copy()
#    29     59487      73404.0      1.2      0.0          Wc = W[2].copy()
#    30     59487 2630030923.0  44211.9     99.4          success = biM.backprop(Wa, Wb, Wc, 3000000, 0.1, 0.01)
#    31     59487     109937.0      1.8      0.0          if success > 0:
#    32                                                       dist = np.linalg.norm(Wa-W[0], 2)**2+np.linalg.norm(Wb-W[1],
#    33     59486   16091523.0    270.5      0.6                                                                  2)**2+np.linalg.norm(Wc-W[2], 2)**2
#    34     59486      78792.0      1.3      0.0              if dist < minDist:
#    35     36424      17771.0      0.5      0.0                  solFound = True
#    36     36424      18911.0      0.5      0.0                  minDist = dist
#    37     36424      48861.0      1.3      0.0                  WaRet = Wa
#    38     36424      34143.0      0.9      0.0                  WbRet = Wb
#    39     36424      32888.0      0.9      0.0                  WcRet = Wc
#    40     19829       9579.0      0.5      0.0      if solFound:
#    41     19829      13587.0      0.7      0.0          return [WaRet, WbRet, WcRet], True
#    42                                               return W, False  # PB
#
#Total time: 1.01216 s
#File: /Users/tillspaeth/Desktop/Masterarbeit/searching-for-fast-MM-algorithms/differenceMap_Beta1_n3_V15_Profiling.py
#Function: checkSolution at line 44
#
#Line #      Hits         Time  Per Hit   % Time  Line Contents
#==============================================================
#    44                                           @profile
#    45                                           def checkSolution(W):
#    46         1          1.0      1.0      0.0      p = W[0].shape[0]
#    47         1          1.0      1.0      0.0      nn = W[0].shape[1]
#    48         1          6.0      6.0      0.0      n = int(np.sqrt(nn))
#    49         1          9.0      9.0      0.0      BIdx = np.array([k*n for k in range(n)])
#    50         1          4.0      4.0      0.0      c = np.zeros(nn, dtype=float)
#    51         1          7.0      7.0      0.0      Wa = np.maximum(np.minimum(np.round(W[0]), 1.0), -1.0)
#    52         1          6.0      6.0      0.0      Wb = np.maximum(np.minimum(np.round(W[1]), 1.0), -1.0)
#    53         1          6.0      6.0      0.0      Wc = np.maximum(np.minimum(np.round(W[2]), 1.0), -1.0)
#    54                                           
#    55         1        756.0    756.0      0.1      @jit(nopython=True, nogil=True, cache=True)
#    56                                               def fastLoop(n, nn, p, BIdx, c, Wa, Wb, Wc):
#    57                                                   for i in range(100):
#    58                                                       a = np.random.rand(nn)*2.0-1.0
#    59                                                       b = np.random.rand(nn)*2.0-1.0
#    60                                                       nA = np.linalg.norm(a, 2)
#    61                                                       nB = np.linalg.norm(b, 2)
#    62                                           
#    63                                                       if np.abs(nA) > 0.1 and np.abs(nB) > 0.1:
#    64                                                           a /= nA
#    65                                                           b /= nB
#    66                                           
#    67                                                           for ii in range(n):  # Matrixmultiplikation für abgerollte Matrizen
#    68                                                               AA = a[ii*n:ii*n+n]
#    69                                                               for jj in range(n):
#    70                                                                   BB = b[BIdx+jj]
#    71                                                                   c[ii*n+jj] = AA.dot(BB)
#    72                                           
#    73                                                           aWaveStar = Wa.dot(a)
#    74                                                           bWaveStar = Wb.dot(b)
#    75                                                           cWaveStar = aWaveStar*bWaveStar
#    76                                                           cWave = Wc.dot(cWaveStar)
#    77                                                           errC = cWave-c
#    78                                                           err2Norm = np.linalg.norm(errC, 2)
#    79                                                           if err2Norm > 0.001:
#    80                                                               return False
#    81                                                       else:
#    82                                                           i -= 1
#    83                                                   return True  # fastLoop
#    84         1    1011366.0 1011366.0     99.9      ret = fastLoop(n, nn, p, BIdx, c, Wa, Wb, Wc)
#    85         1          2.0      2.0      0.0      return ret  # checkSolution
#
#Total time: 1068.95 s
#File: /Users/tillspaeth/Desktop/Masterarbeit/searching-for-fast-MM-algorithms/differenceMap_Beta1_n3_V15_Profiling.py
#Function: roundInit at line 146
#
#Line #      Hits         Time  Per Hit   % Time  Line Contents
#==============================================================
#   146                                           @profile
#   147                                           def roundInit(n, p):
#   148        10         15.0      1.5      0.0      nn = int(n**2)
#   149        10          8.0      0.8      0.0      success = -1
#   150        56        182.0      3.2      0.0      while success < 0:
#   151        46       1744.0     37.9      0.0          Wa = np.random.rand(p*nn).reshape([p, nn])*2.0-1.0
#   152        46        434.0      9.4      0.0          Wb = np.random.rand(p*nn).reshape([p, nn])*2.0-1.0
#   153        46        322.0      7.0      0.0          Wc = np.random.rand(nn*p).reshape([nn, p])*2.0-1.0
#   154        46  178422005.0 3878739.2     16.7          success = biM.backprop(Wa, Wb, Wc, 3000000, 0.1, 0.01)
#   155        10        264.0     26.4      0.0      MA = np.ones(Wa.shape)
#   156        10         43.0      4.3      0.0      MB = np.ones(Wb.shape)
#   157        10         30.0      3.0      0.0      MC = np.ones(Wc.shape)
#   158        10         30.0      3.0      0.0      TA = np.ones(Wa.shape)
#   159        10         29.0      2.9      0.0      TB = np.ones(Wb.shape)
#   160        10         27.0      2.7      0.0      TC = np.ones(Wc.shape)
#   161        10         56.0      5.6      0.0      ei = np.zeros(nn, dtype=float)
#   162        10          3.0      0.3      0.0      rounds = 0
#   163        10          6.0      0.6      0.0      while True:
#   164      6220    6396727.0   1028.4      0.6          i, j, err, matSel = findWeight(Wa, Wb, Wc, TA, TB, TC, ei)
#   165      6220       5552.0      0.9      0.0          if i < 0:
#   166        10          7.0      0.7      0.0              break
#   167      6210      58809.0      9.5      0.0          WaT = Wa.copy()
#   168      6210      14302.0      2.3      0.0          WbT = Wb.copy()
#   169      6210      12999.0      2.1      0.0          WcT = Wc.copy()
#   170      6210       4215.0      0.7      0.0          if matSel == 0:
#   171      2070       2620.0      1.3      0.0              TA[i, j] = 0
#   172      2070       1678.0      0.8      0.0              MA[i, j] = 0
#   173      2070      75971.0     36.7      0.0              WaT[i, j] = np.minimum(np.maximum(np.round(WaT[i, j]), -1), 1)
#   174      6210       4405.0      0.7      0.0          if matSel == 1:
#   175      2070       2579.0      1.2      0.0              TB[i, j] = 0
#   176      2070       1637.0      0.8      0.0              MB[i, j] = 0
#   177      2070      76512.0     37.0      0.0              WbT[i, j] = np.minimum(np.maximum(np.round(WbT[i, j]), -1), 1)
#   178      6210       4400.0      0.7      0.0          if matSel == 2:
#   179      2070       2669.0      1.3      0.0              TC[i, j] = 0
#   180      2070       1690.0      0.8      0.0              MC[i, j] = 0
#   181      2070      74023.0     35.8      0.0              WcT[i, j] = np.minimum(np.maximum(np.round(WcT[i, j]), -1), 1)
#   182      6210  883723022.0 142306.4     82.7          success = biM.backpropM(WaT, WbT, WcT, MA, MB, MC, 100000, 0.1, 0.01)
#   183      6210      19597.0      3.2      0.0          if success > 0:
#   184      1742       7027.0      4.0      0.0              Wa = WaT
#   185      1742       3085.0      1.8      0.0              Wb = WbT
#   186      1742       2803.0      1.6      0.0              Wc = WcT
#   187      1742       1812.0      1.0      0.0              rounds += 1
#   188                                                       # print("o",end='',flush=True)
#   189                                                   else:
#   190      4468       3388.0      0.8      0.0              if matSel == 0:
#   191      1477       5844.0      4.0      0.0                  MA[i, j] = 1
#   192      4468       2946.0      0.7      0.0              if matSel == 1:
#   193      1530       6056.0      4.0      0.0                  MB[i, j] = 1
#   194      4468       2846.0      0.6      0.0              if matSel == 2:
#   195      1461       5871.0      4.0      0.0                  MC[i, j] = 1
#   196                                                       # print("x",end='',flush=True)
#   197        10        103.0     10.3      0.0      print("roundInit-Rundungen: ", str(rounds))
#   198        10         13.0      1.3      0.0      return [Wa, Wb, Wc]  # roundInit
#
#Total time: 3759.97 s
#File: /Users/tillspaeth/Desktop/Masterarbeit/searching-for-fast-MM-algorithms/differenceMap_Beta1_n3_V15_Profiling.py
#Function: diffMap at line 200
#
#Line #      Hits         Time  Per Hit   % Time  Line Contents
#==============================================================
#   200                                           @profile
#   201                                           def diffMap(id):
#   202         1          4.0      4.0      0.0      p = 23
#   203         1          1.0      1.0      0.0      n = 3
#   204         1          2.0      2.0      0.0      nn = int(n**2)
#   205                                           
#   206         1         30.0     30.0      0.0      seed = int(time.time())+int(uuid.uuid4())+id
#   207         1          9.0      9.0      0.0      np.random.seed(seed % 135790)
#   208         1   82490412.0 82490412.0      2.2      W = roundInit(n, p)
#   209         1        111.0    111.0      0.0      BFs = [bf.bloomFilter(2*nn*p, 0.00001) for b in range(20)]
#   210         1          1.0      1.0      0.0      i = 0  # iteration
#   211         1          1.0      1.0      0.0      numOfCycles = 0
#   212         1          1.0      1.0      0.0      numOfTries = 0
#   213         1          2.0      2.0      0.0      diffs = []
#   214         1          2.0      2.0      0.0      jumps = []  # indices of jumps
#   215         1          1.0      1.0      0.0      heights = []  # multpliers (cyclCnt) of jumps
#   216                                           
#   217         1          1.0      1.0      0.0      jumpFactor = 0.0125
#   218         1          1.0      1.0      0.0      bloomOn = True
#   219                                           
#   220         1          1.0      1.0      0.0      while True:
#   221     19828      20782.0      1.0      0.0          s = False
#   222     39656      44706.0      1.1      0.0          while not s:
#   223     19828 2647589596.0 133527.8     70.4              PBx, s = PB(W)  # not overwriting
#   224     19828      23291.0      1.2      0.0              if not s:
#   225                                                           print("   Prz: ", id, " BP failed -> reset")
#   226                                                           seed = int(time.time())+int(uuid.uuid4())+id
#   227                                                           np.random.seed(seed % 135745)
#   228                                                           W = roundInit(n, p)
#   229                                                           i = 0
#   230                                                           numOfCycles = 0
#   231                                                           numOfTries += 1
#   232                                                           diffs = []
#   233                                                           jumps = []
#   234                                                           heights = []
#   235                                           
#   236     19828     891385.0     45.0      0.0          PAy = PA([2.0*PBx[0]-W[0], 2.0*PBx[1]-W[1], 2.0*PBx[2]-W[2]])
#   237     19828      97004.0      4.9      0.0          delta = [PAy[0]-PBx[0], PAy[1]-PBx[1], PAy[2]-PBx[2]]
#   238     19828     103949.0      5.2      0.0          W = [W[0]+delta[0], W[1]+delta[1], W[2]+delta[2]]
#   239                                                   norm2Delta = np.linalg.norm(
#   240     19828    4009885.0    202.2      0.1              delta[0], 2)**2+np.linalg.norm(delta[1], 2)**2+np.linalg.norm(delta[2], 2)**2
#   241     19828     160128.0      8.1      0.0          norm2Delta = np.sqrt(norm2Delta)
#   242     19828      31791.0      1.6      0.0          diffs.append(norm2Delta)
#   243                                           
#   244     19828      21165.0      1.1      0.0          cyclCnt = 0
#   245     27482      48700.0      1.8      0.0          for b in range(len(BFs)):
#   246     27482   36101573.0   1313.6      1.0              if not BFs[b].store(PAy):
#   247     19828      27670.0      1.4      0.0                  cyclCnt = b
#   248     19828      23114.0      1.2      0.0                  break
#   249                                           
#   250     19828      30479.0      1.5      0.0          if norm2Delta < 0.5:
#   251         1          4.0      4.0      0.0              print("Lösung gefunden?")
#   252         1      72704.0  72704.0      0.0              WW = PA(PB(W)[0])  # PA is overwriting, but PB is not
#   253         1    1013716.0 1013716.0      0.0              c2 = checkSolution(WW)
#   254         1          2.0      2.0      0.0              if c2:
#   255         1         10.0     10.0      0.0                  np.save("solution_"+str(n)+"_"+str(i)+"_"+str(time.time())+"_"+"V15",
#   256         1       8594.0   8594.0      0.0                          [WW[0], WW[1], WW[2], jumpFactor, diffs, jumps, heights, i, numOfCycles, numOfTries, bloomOn])
#   257         1         16.0     16.0      0.0                  print(".... Lösung korrekt")
#   258         1  110172115.0 110172115.0      2.9                  W = roundInit(n, p)
#   259         1        120.0    120.0      0.0                  BFs = [bf.bloomFilter(2*nn*p, 0.00001) for b in range(20)]
#   260         1          2.0      2.0      0.0                  numOfCycles = 0
#   261         1          1.0      1.0      0.0                  numOfTries = 0
#   262         1        100.0    100.0      0.0                  diffs = []
#   263         1          8.0      8.0      0.0                  jumps = []
#   264         1          3.0      3.0      0.0                  heights = []
#   265         1          1.0      1.0      0.0                  i = 0
#   266         1          1.0      1.0      0.0                  return
#   267                                                       else:
#   268                                                           print(".... keine gültige Lösung")
#   269                                           
#   270                                                   #mutex.acquire()
#   271     19827      26810.0      1.4      0.0          if i % 100 == 0:
#   272       197        961.0      4.9      0.0              print("---------------------------")
#   273       197        538.0      2.7      0.0              print("Prozess:", id)
#   274       197        488.0      2.5      0.0              print("Iter.:  ", i)
#   275       197        939.0      4.8      0.0              print("Delta:  ", norm2Delta)
#   276     19827      24065.0      1.2      0.0          if cyclCnt > 0:
#   277                                                       #print("**** Zyklus entdeckt! *****")
#   278      7230      41691.0      5.8      0.0              print("**** cyclCnt: ", cyclCnt)
#   279     19827      22616.0      1.1      0.0          if i > 2000 and norm2Delta > 3.0:
#   280         8         17.0      2.1      0.0              print(i, " cycles -> Reset")
#   281         8         16.0      2.0      0.0              print("tries:", numOfTries)
#   282                                                   #mutex.release()
#   283                                           
#   284     19827      21648.0      1.1      0.0          if cyclCnt > 0 and bloomOn:
#   285      7230     189001.0     26.1      0.0              W[0] += (np.random.rand(p*nn).reshape([p, nn])*2.0-1.0)*cyclCnt*jumpFactor
#   286      7230      98143.0     13.6      0.0              W[1] += (np.random.rand(p*nn).reshape([p, nn])*2.0-1.0)*cyclCnt*jumpFactor
#   287      7230      85041.0     11.8      0.0              W[2] += (np.random.rand(p*nn).reshape([nn, p])*2.0-1.0)*cyclCnt*jumpFactor
#   288     19827      22369.0      1.1      0.0          if cyclCnt > 0:
#   289      7230      10290.0      1.4      0.0              jumps.append(i)
#   290      7230       8629.0      1.2      0.0              heights.append(cyclCnt)
#   291      7230       8688.0      1.2      0.0              numOfCycles += 1
#   292     19827      22417.0      1.1      0.0          if i > 2000 and norm2Delta > 3.0:
#   293         8        295.0     36.9      0.0              seed = int(time.time())+int(uuid.uuid4())+id
#   294         8         47.0      5.9      0.0              np.random.seed(seed % 135790)
#   295         8  876373076.0 109546634.5     23.3              W = roundInit(n, p)
#   296         8       1002.0    125.2      0.0              BFs = [bf.bloomFilter(2*nn*p, 0.00001) for b in range(20)]
#   297         8         12.0      1.5      0.0              numOfCycles = 0
#   298         8         12.0      1.5      0.0              numOfTries += 1
#   299         8        936.0    117.0      0.0              diffs = []
#   300         8         79.0      9.9      0.0              jumps = []
#   301         8         23.0      2.9      0.0              heights = []
#   302         8          8.0      1.0      0.0              i = 0
#   303     19827      23208.0      1.2      0.0          i += 1
#   304                                               return  # diffMap
#
#
##
