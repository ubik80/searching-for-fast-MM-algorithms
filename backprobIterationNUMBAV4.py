import numpy as np
np.set_printoptions(precision=4, suppress=True)
import matplotlib.pyplot as plt
from numba import jit
import time
import sys
import checkSolution as cs

def findCalcRule(n,p,numIters,Wa,Wb,Wc,MA,MB,MC,limit=0.01,nue=0.1):
    nn=n**2
    nueC=nue
    nueAB=nue
    Wa=np.maximum(np.minimum(np.asarray(Wa),99.9),-99.9)
    Wb=np.maximum(np.minimum(np.asarray(Wb),99.9),-99.9)
    Wc=np.maximum(np.minimum(np.asarray(Wc),99.9),-99.9)
    MA=np.asarray(MA)
    MB=np.asarray(MB)
    MC=np.asarray(MC)
    errHist=np.zeros(numIters,dtype=float)
    BIdx=np.array([k*n for k in range(n)])
    c=np.zeros(nn,dtype=float)

    @jit(nopython=True,nogil=True,cache=True)
    def iteration(n,nn,p,numIters,Wa,Wb,Wc,MA,MB,MC,limit,nueAB,nueC,BIdx,errHist,c):
        for i in range(numIters):
            a=np.random.rand(nn)*2.0-1.0
            b=np.random.rand(nn)*2.0-1.0
            nA=np.linalg.norm(a,2)
            nB=np.linalg.norm(b,2)

            if np.abs(nA)>0.1 and np.abs (nB)>0.1:
                a/=nA
                b/=nB

                for ii in range(n): #Matrixmultiplikation für aufgerollte Matrizen
                    AA=a[ii*n:ii*n+n]
                    for jj in range(n):
                        BB=b[BIdx+jj]
                        c[ii*n+jj]=AA.dot(BB)

                aWaveStar=Wa.dot(a)
                bWaveStar=Wb.dot(b)
                cWaveStar=aWaveStar*bWaveStar
                cWave=Wc.dot(cWaveStar)
                errC=cWave-c
                errHist[i]=np.linalg.norm(errC,2)
                errCStar=Wc.T.dot(errC)
                deltaWc=-nueC*np.outer(errC,cWaveStar)
                deltaWa=-nueAB*np.outer(errCStar*bWaveStar,a)
                deltaWb=-nueAB*np.outer(errCStar*aWaveStar,b)
                Wa+=deltaWa*MA
                Wb+=deltaWb*MB
                Wc+=deltaWc*MC
            else:
                i-=1

            if i>500 and np.max(errHist[i-500:i])<limit:
                return True

            #if i%100000==0:print(np.linalg.norm(errC,2))

        return False #iteration

    success=iteration(n,nn,p,numIters,Wa,Wb,Wc,MA,MB,MC,limit,nueAB,nueC,BIdx,errHist,c)
    hasNANs=np.sum(np.isnan(Wa))+np.sum(np.isnan(Wb))+np.sum(np.isnan(Wc))>0
    if hasNANs:success=False

    return Wa,Wb,Wc,errHist,success #findCalcRule

if __name__ == '__main__':
    #Strassen:
    # Wa=[[1,0,0,1],[0,0,1,1],[1,0,0,0],[0,0,0,1],[1,1,0,0],[-1,0,1,0],[0,1,0,-1]]
    # Wa=np.array(Wa)
    # Wb=[[1,0,0,1],[1,0,0,0],[0,1,0,-1],[-1,0,1,0],[0,0,0,1],[1,1,0,0],[0,0,1,1]]
    # Wb=np.array(Wb)
    # Wc=[[1,0,0,1,-1,0,1],[0,0,1,0,1,0,0],[0,1,0,1,0,0,0],[1,-1,1,0,0,1,0]]
    # Wc=np.array(Wc)
    # nn=Wc.shape[0]
    # n=int(np.sqrt(nn))
    # p=Wc.shape[1]

    n=3
    nn=9
    p=23

    Wa=np.random.rand(p,nn)
    Wb=np.random.rand(p,nn)
    Wc=np.random.rand(nn,p)
    M=np.ones(Wc.shape)

    start = time. time()
    Wa,Wb,Wc,errHist,success=findCalcRule(n,p,5000000,Wa,Wb,Wc,M,0.01,0.1)
    cs.checkResult(Wa,Wb,Wc,True,0.05)
    end = time. time()
    print("time: ",end - start)
    print(success)
    print(Wc)