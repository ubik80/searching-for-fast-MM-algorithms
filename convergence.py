import numpy as np
import matplotlib.pyplot as plt
from numba import jit
np.set_printoptions(precision=2, suppress=True)

def checkSolution(W):
    p=W[0].shape[0]
    nn=W[0].shape[1]
    n=int(np.sqrt(nn))
    BIdx=np.array([k*n for k in range(n)])
    c=np.zeros(nn,dtype=float)
    Wa=np.round(W[0])
    Wb=np.round(W[1])
    Wc=np.round(W[2])
    Wa=np.maximum(np.minimum(Wa,1.0),-1.0)
    Wb=np.maximum(np.minimum(Wb,1.0),-1.0)
    Wc=np.maximum(np.minimum(Wc,1.0),-1.0)

    @jit(nopython=True,nogil=True,cache=True)
    def fastLoop(n,nn,p,BIdx,c,Wa,Wb,Wc):
        for i in range(100):
            a=np.random.rand(nn)*2.0-1.0
            b=np.random.rand(nn)*2.0-1.0
            nA=np.linalg.norm(a,2)
            nB=np.linalg.norm(b,2)

            if np.abs(nA)>0.1 and np.abs (nB)>0.1:
                a/=nA
                b/=nB

                for ii in range(n): #Matrixmultiplikation für abgerollte Matrizen
                    AA=a[ii*n:ii*n+n]
                    for jj in range(n):
                        BB=b[BIdx+jj]
                        c[ii*n+jj]=AA.dot(BB)

                aWaveStar=Wa.dot(a)
                bWaveStar=Wb.dot(b)
                cWaveStar=aWaveStar*bWaveStar
                cWave=Wc.dot(cWaveStar)
                errC=cWave-c
                err2Norm=np.linalg.norm(errC,2)
                if err2Norm>0.001: return False
            else: i-=1
        return True #fastLoop
    ret=fastLoop(n,nn,p,BIdx,c,Wa,Wb,Wc)
    return ret # checkSolution

sol=np.load("solution_3_1020_15_V13_2.npy",allow_pickle=True)

Wa=sol[0]
Wb=sol[1]
Wc=sol[2]
if not checkSolution([Wa,Wb,Wc]):
    print("Lösung nicht korrekt !!!!")
    exit()
else: print("Lösung korrekt")
nn=Wc.shape[0]
p=Wc.shape[1]
n=int(np.round(np.sqrt(nn)))

diffs=sol[3]
numOfIters=sol[4]
numOfCycles=sol[5]
numOfTries=sol[6]

numOfCycles

plt.rcParams.update({'font.size': 10})
plt.plot(diffs)
plt.xlabel("iteration")
plt.ylabel("| Δ |")

plt.savefig('diffMap17.png',dpi=300)


Wc
