import numpy as np
import matplotlib.pyplot as plt
from numba import jit
import os
import checkSolution as cs


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


os.chdir("/Users/tillspaeth/Google Drive/V21DiffMap")
fileNames = os.listdir()
TfileNames = []
for f in fileNames:
    if "_V21.npy" in f:
        TfileNames.append(f)
fileNames = TfileNames

numbWithFactor = dict()
itersWithFactorAll = dict()
failsWithFactor = dict()
examplePloted = False

i = 0
for f in fileNames:
    i += 1
    sol = np.load(f, allow_pickle=True)
    Wa = sol[0]
    Wb = sol[1]
    Wc = sol[2]
    jumpFactor = round(sol[3], 3)
    diffs = sol[4]
    jumps = sol[5]
    heights = sol[6]
    numOfIters = sol[7]
    numOfCycles = sol[8]
    numOfTries = sol[9]

    if numOfIters < 5000 and not numOfTries < 0:
        if not cs.checkSolutionInt([Wa, Wb, Wc]):
            print("Lösung nicht korrekt !!!!")
            exit()
        # if not examplePloted:
        #     maxY = np.max(diffs)-0.2
        #     minY = np.min(diffs)-0.2
        #     plt.rcParams.update({'font.size': 10})
        #     plt.plot(diffs)
        #     plt.xlabel("iteration")
        #     plt.ylabel("| Δ |")
        #     y = [minY, maxY]
        #     for j in range(len(jumps)):
        #         x = [jumps[j], jumps[j]]
        #         plt.plot(x, y, '-r', alpha=0.5)
        #         #plt.text(jumps[j]-20, maxY-0.2, heights[j])
        #     ttl = "jump factor = "+str(jumpFactor)
        #     plt.title(ttl)
        #     #examplePloted = True
        #     # break
        #     picName = 'example_'+str(i)+'.png'
        #     plt.savefig(picName, dpi=300)
        #     plt.close()

    if not str(jumpFactor) in numbWithFactor:
        numbWithFactor[str(jumpFactor)] = 0
    if not str(jumpFactor) in itersWithFactorAll:
        itersWithFactorAll[str(jumpFactor)] = []
    if not str(jumpFactor) in failsWithFactor:
        failsWithFactor[str(jumpFactor)] = 0

    numbWithFactor[str(jumpFactor)] += 1
    if numOfTries > -1:
        itersWithFactorAll[str(jumpFactor)].append(numOfIters)
    else:
        failsWithFactor[str(jumpFactor)] += 1

x = []
l = []
u = []
m = []
for f in itersWithFactorAll:
    if len(itersWithFactorAll[f]) > 0:
        x.append(float(f))
        mm, ll, uu = bootStrapStats(itersWithFactorAll[f])
        l.append(ll)
        u.append(uu)
        m.append(mm)
s = np.argsort(np.array(x))

fig, ax1 = plt.subplots()
ax1.plot(np.array(x)[s], np.array(m)[s], '-')  # , color='red')
# ax1.plot(np.array(x)[s], np.array(l)[s], '-.')
# ax1.plot(np.array(x)[s], np.array(u)[s], '-.')
ax1.set_xticks([0.1*i for i in range(0, 11)])
ax1.set_xlabel("jump factor (blue)")
ax1.set_ylabel("avg. num. iterations")

ax2 = ax1.twinx()
x = []
y = []
for f in failsWithFactor:
    x.append(float(f))
    y.append(failsWithFactor[f]/numbWithFactor[f]*100.0)
s = np.argsort(np.array(x))
ax2.plot(np.array(x)[s], np.array(y)[s], '-', color='green')
ax2.set_ylabel("% failed runs (green)")
plt.title("num. of iterations, n=3")

os.chdir("/Users/tillspaeth/Desktop/Masterarbeit/Ausarbeitung")
plt.savefig('factorItersGraph.png', dpi=300)
plt.close()
