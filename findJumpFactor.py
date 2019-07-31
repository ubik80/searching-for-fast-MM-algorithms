import numpy as np
import matplotlib.pyplot as plt
from numba import jit
import os


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


def bootStrapStats(_vals):
    vals = np.array(_vals)
    n = len(vals)
    k = 10000
    means = np.zeros(k, dtype=float)
    for i in range(k):
        draws = np.random.randint(0, n, n)
        means[i] = np.mean(vals[draws])
    means = np.sort(means)
    CIlower = means[int(round(k*0.025))]
    CIupper = means[int(round(k*0.975))]
    return np.mean(means), CIlower, CIupper


os.chdir("/Users/tillspaeth/Google Drive/V16DiffMap")
fileNames = os.listdir()
TfileNames = []
for f in fileNames:
    if "_V16.npy" in f:
        TfileNames.append(f)
fileNames = TfileNames

itersWithFactor = dict()
numbWithFactor = dict()
itersWithFactorAll = dict()

for f in fileNames:
    # print(f)
    sol = np.load(f, allow_pickle=True)
    Wa = sol[0]
    Wb = sol[1]
    Wc = sol[2]
    # if not checkSolution([Wa, Wb, Wc]):
    #     print("Lösung nicht korrekt !!!!")
    #     exit()
    jumpFactor = round(sol[3], 3)
    diffs = sol[4]
    jumps = sol[5]
    heights = sol[6]
    numOfIters = sol[7]
    numOfCycles = sol[8]
    numOfTries = sol[9]
    bloomOn = sol[10]
    success = sol[11]

    if numOfCycles > 0 and bloomOn:
        if str(jumpFactor) in itersWithFactor:
            itersWithFactor[str(jumpFactor)] += numOfIters
            numbWithFactor[str(jumpFactor)] += 1
            itersWithFactorAll[str(jumpFactor)].append(numOfIters)
        else:  # new
            itersWithFactor[str(jumpFactor)] = numOfIters
            numbWithFactor[str(jumpFactor)] = 1
            itersWithFactorAll[str(jumpFactor)] = [numOfIters]

        # print("filename: ",f)
        # print("bloomFilter on: ",bloomOn)
        # print("factor: ",jumpFactor)
        # print("numOfIters: ",numOfIters)
        # print("tries: ",numOfTries)
        # print("numOfCycles: ",numOfCycles)
        # print("-------------------------")

        # maxY = np.max(diffs)-0.2
        # minY = np.min(diffs)-0.2
        # plt.rcParams.update({'font.size': 10})
        # plt.plot(diffs)
        # plt.xlabel("iteration")
        # plt.ylabel("| Δ |")
        # y = [minY, maxY]
        # for j in range(len(jumps)):
        #     x = [jumps[j], jumps[j]]
        #     plt.plot(x, y, '-r', alpha=0.5)
        #     plt.text(jumps[j]-20, maxY-0.2, heights[j])
        # if bloomOn:
        #     ttl = str(jumpFactor)
        # else:
        #     ttl = 'no cycl. det.'
        # plt.title(ttl)
        # picName = 'example_'+str(i)+'.png'
        # # plt.savefig(picName,dpi=300)
        # plt.close()

# print("# with BF: ", blmOnCnt)
# print("# w/o BF: ", blmOffCnt)
# print("# iters with BF: ", itersWithBF/blmOnCnt)
# print("# iters WO BF: ", itersWOBF/blmOffCnt)
# m, l, u = bootStrapStats(itersWOBFAll)
# print("    bootstrap: ", m, l, u)
# print("# tries with BF: ", triesWithBF/blmOnCnt)
# print("# tries WO BF: ", triesWOBF/blmOffCnt)

for n in numbWithFactor:
    print("factor ", round(float(n), 5), ", samples: ",
          numbWithFactor[n], ", avg. #iters: ", round(itersWithFactor[n]/numbWithFactor[n], 1))
    m, l, u = bootStrapStats(itersWithFactorAll[n])
    print("                                   bootstrap: ",
          round(m, 1), round(l, 1), round(u, 1))

x = []
l = []
u = []
m = []
for f in itersWithFactor:
    x.append(float(f))
    mm, ll, uu = bootStrapStats(itersWithFactorAll[f])
    l.append(ll)
    u.append(uu)
    m.append(mm)
s = np.argsort(np.array(x))
plt.plot(np.array(x)[s], np.array(m)[s], '-o')
plt.plot(np.array(x)[s], np.array(l)[s], '-o')
plt.plot(np.array(x)[s], np.array(u)[s], '-o')
plt.xticks(np.array(x)[s])
plt.xlabel("factor")
plt.ylabel("avg. num. iterations")

# os.chdir("/Users/tillspaeth/Desktop/Masterarbeit/searching-for-fast-MM-algorithms")
#plt.savefig('factorItersGraph.png', dpi=300)

# itersWithFactorAll.keys()
# iters = itersWithFactorAll["0.021875000000000002"]
#
# n, bins, patches = plt.hist(iters, 5, facecolor='blue', alpha=1.0)
