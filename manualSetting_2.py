import numpy as np
import backprop as biM
import checkSolution as cs
import summationOps as so
np.set_printoptions(precision=2, suppress=True)

n = 2
nn = 4
p = 7

Wa = np.matrix([[1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0]], dtype=float)
Wb = np.matrix([[1, 1, 0, 0],
                [1, 1, 0, -1],
                [0, 0, 0, 1],
                [1, 0, 0, 0],
                [0, -1, 0, 1],
                [1, 1, -1, -1],
                [0, 0, 1, 0]], dtype=float)
Wc = np.matrix([[1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1]], dtype=float)
Ma = np.matrix([[0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0]], dtype=float)
Mb = np.matrix([[1, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1]], dtype=float)
Mc = np.matrix([[1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1]], dtype=float)

Wa = np.array(Wa)
Wb = np.array(Wb)
Wc = np.array(Wc)
Ma = np.array(Ma)
Mb = np.array(Mb)
Mc = np.array(Mc)

# signature = np.zeros(p)
# for i in range(p):
#     for j in range(nn):
#         #signature[i] += np.abs(Wa[i, j])*j*1000
#         signature[i] += Wa[i, j]
# sorted = np.argsort(signature)
# Wa = Wa[sorted, :]
# Wb = Wb[sorted, :]

while True:
    #Wa = np.random.rand(nn*p).reshape(p, nn)*2.0-1.0
    Wb = np.random.rand(nn*p).reshape(p, nn)*2.0-1.0
    Wc = np.random.rand(nn*p).reshape(nn, p)*2.0-1.0
    #Ma = np.ones(Wa.shape)
    Mb = np.ones(Wb.shape)
    Mc = np.ones(Wc.shape)

    print('.', end='', flush=True)
    success = biM.backpropM(Wa, Wb, Wc, Ma, Mb, Mc, 50000000, 0.3, 0.0001)

    if success > 0:
        break

if success < 0:
    print("failed")
else:
    print("solution found ", success)
print(Wa)
print(Wb)
print(Wc)

a = np.random.randint(0, 3, nn)
A = a.reshape([n, n])
b = np.random.randint(0, 3, nn)
B = b.reshape([n, n])
C = A.dot(B)
c = np.array(C.flatten())
aStar = Wa.dot(a)
bStar = Wb.dot(b)
cStar = aStar*bStar
cWave = Wc.dot(cStar)
print(np.linalg.norm(c-cWave, 2) < 0.001)
print(cs.checkSolutionReal([Wa, Wb, Wc]))

so.numOfOps(Wa)
so.numOfOps(Wb)
so.numOfOps(Wc)

#np.save("manualSolution_n2_p7_2", [Wa, Wb, Wc])
#W = np.load("manualSolution_n2_p7_1.npy", allow_pickle=True)
