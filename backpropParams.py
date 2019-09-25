# coding: utf8
import numpy as np
import backprop as bp
import matplotlib.pyplot as plt
np.set_printoptions(precision=2, suppress=True)


def PA(W):  # überschreibend
    W[0] = np.minimum(np.maximum(np.round(W[0]), -1.0), 1.0)
    W[1] = np.minimum(np.maximum(np.round(W[1]), -1.0), 1.0)
    W[2] = np.minimum(np.maximum(np.round(W[2]), -1.0), 1.0)
    return W  # PA


p = 23
n = 3
tol = 0.01
numOfSmpls = 100

nn = n**2
success = -1
while success < 0:
    print("Initialisierung ...")
    Wa = np.random.rand(p*nn).reshape([p, nn])*2.0-1.0
    Wb = np.random.rand(p*nn).reshape([p, nn])*2.0-1.0
    Wc = np.random.rand(nn*p).reshape([nn, p])*2.0-1.0
    success = bp.backpropNue(Wa, Wb, Wc, 6000000, 0.01, 0.05, 0.1)
print(" ... erfolgreich")

etas = [0.01*i for i in range(1, 26)]
avgDists = np.zeros(len(etas))
dists = np.zeros(numOfSmpls)

for ii in range(len(avgDists)):
    eta = etas[ii]
    print("eta =", eta, end='', flush=True)
    for i in range(len(dists)):
        success = -1
        while success < 0:
            PAWa, PAWb, PAWc = PA([Wa.copy(), Wb.copy(), Wc.copy()])
            success = bp.backpropNueABC(PAWa, PAWb, PAWc, 3000000, tol, eta, eta, eta)
        dist = np.linalg.norm(Wa-PAWa, 2)**2+np.linalg.norm(Wb-PAWb, 2)**2 + \
            np.linalg.norm(Wc-PAWc, 2)**2
        dist = np.sqrt(dist)
        dists[i] = dist
    avgDists[ii] = np.mean(dists)
    print("    avg =", avgDists[ii])

plt.rcParams.update({'font.size': 14})
plt.plot(etas, avgDists)
plt.xlabel("η c, η c* ")
plt.ylabel("proj. distance")
plt.title("n="+str(n))
plt.savefig('/Users/tillspaeth/Desktop/Masterarbeit/Ausarbeitung/etaAndDists_'+str(n)+'.png', dpi=300)

#
