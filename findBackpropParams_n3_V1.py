import numpy as np
import backprop as bp
import matplotlib.pyplot as plt
import os

n = 3
p = 23
nn = n**2

maxIters = 20000000
repetitions = 1000
minNueAB = 0.01
minNueC = 0.01
maxNueAB_alt = 0.5
maxNueC_alt = 0.5
maxNueAB = maxNueAB_alt+(maxNueAB_alt-minNueAB)
maxNueC = maxNueC_alt+(maxNueC_alt-minNueC)
steps_alt = 50
steps = steps_alt*2

nueAB = [minNueAB+(maxNueAB-minNueAB)/steps*i for i in range(0, steps + 1)]
nueC = [minNueC+(maxNueC-minNueC)/steps*i for i in range(0, steps + 1)]

costs, fails, smpls = np.load("backprop_n_3_params.npy", allow_pickle=True)

# workDone = True
# while workDone:
#     workDone = False
#     for i in range(len(nueAB)):
#         for j in range(len(nueC)):
#             if smpls[i, j] < repetitions:
#                 Wa = np.random.rand(nn*p).reshape(p, nn)*2.0-1.0
#                 Wb = np.random.rand(nn*p).reshape(p, nn)*2.0-1.0
#                 Wc = np.random.rand(nn*p).reshape(nn, p)*2.0-1.0
#                 print("x")
#                 iters = bp.backpropNue(Wa, Wb, Wc, maxIters, 0.0001, nueAB[i], nueC[j])
#                 if iters < 0:
#                     fails[i, j] += 1
#                     iters = maxIters
#                 costs[i, j] += iters
#                 smpls[i, j] += 1
#                 workDone = True
#                 np.save("backprop_n_3_params", [costs, fails, smpls])

opt = np.argmin(costs)
i = int(np.floor(opt/costs.shape[0]))
j = opt-i*costs.shape[1]
optNueAB = nueAB[i]
optNueC = nueC[j]
print(optNueAB, optNueC, " -> ", costs[i, j]/smpls[i, j])
print("finished")

nueC = nueC[0:101]
nueAB = nueAB[0:60]
costs = costs[0:60, 0:101]
smpls = smpls[0:60, 0:101]
plt.contourf(nueC, nueAB, costs/np.maximum(smpls, 1))
#plt.contourf(nueC, nueAB, costs/smpls, 1000, vmin=10000, vmax=maxIters)
plt.plot(optNueC, optNueAB, color='red', marker='o', markersize=10)
#plt.colorbar(ticks=[2000*i for i in range(5, 13)])
plt.colorbar()
plt.xlabel("η c")
plt.ylabel("η c*")
plt.title("num. of iterations")
# os.chdir("/Users/tillspaeth/Desktop/Masterarbeit/")
#plt.savefig('backpropParams_n3.png', dpi=300)
