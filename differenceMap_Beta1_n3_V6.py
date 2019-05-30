# coding: utf8
import os
import numpy as np
import backprobIteration_diffMap_NUMBAV4 as bi
import bloomFilter as bf
import multiprocessing as mp
import time
import uuid
from numba import jit
np.set_printoptions(precision=2, suppress=True)

def PA(W):
    W[0]=np.minimum(np.maximum(np.round(W[0]),-1.0),1.0)
    W[1]=np.minimum(np.maximum(np.round(W[1]),-1.0),1.0)
    W[2]=np.minimum(np.maximum(np.round(W[2]),-1.0),1.0)
    return W #PA

def PB(W,id):
    p=W[0].shape[0]
    nn=W[0].shape[1]
    n=int(np.sqrt(nn))
    oldDiff=99999.9
    solved=False
    for i in range(5):
        Wa=W[0].copy()
        Wb=W[1].copy()
        Wc=W[2].copy()
        Wa,Wb,Wc,eh,s=bi.findCalcRule(n,p,3000000,Wa,Wb,Wc,limit=0.01,nue=0.1)
        diff=np.linalg.norm(Wa-W[0],2)**2+np.linalg.norm(Wb-W[1],2)**2+np.linalg.norm(Wc-W[2],2)**2
        if s and diff<oldDiff:
            oldDiff=diff
            WaRet=Wa.copy()
            WbRet=Wb.copy()
            WcRet=Wc.copy()
            solved=True
    if solved: W=[WaRet,WbRet,WcRet]
    else:
        s=False
        while not s:
            print("   Prz: ",id," BP failed -> reset")
            seed=int(time.time())+int(uuid.uuid4())+id
            np.random.seed(seed%135745)
            W=[np.random.rand(p*nn).reshape([p,nn])*2.0-1.0,
            np.random.rand(p*nn).reshape([p,nn])*2.0-1.0,
            np.random.rand(nn*p).reshape([nn,p])*2.0-1.0]
            W[0],W[1],W[2],eh,s=bi.findCalcRule(n,p,3000000,W[0],W[1],W[2],limit=0.01,nue=0.1)
    return W,not solved #PB

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

def diffMap(id,mutex,success):
    p=23
    n=3
    nn=int(n**2)

    seed=int(time.time())+int(uuid.uuid4())+id
    np.random.seed(seed%135790)

    W=[np.random.rand(p*nn).reshape([p,nn])*2.0-1.0,
    np.random.rand(p*nn).reshape([p,nn])*2.0-1.0,
    np.random.rand(nn*p).reshape([nn,p])*2.0-1.0]

    BF=bf.bloomFilter(2*nn*p,0.00001)

    i=0
    while success.value==0:
        PBx,failed=PB(W.copy(),id)
        PAy=PA([2.0*PBx[0]-W[0],2.0*PBx[1]-W[1],2.0*PBx[2]-W[2]])
        delta=[PAy[0]-PBx[0],PAy[1]-PBx[1],PAy[2]-PBx[2]]
        norm2Delta=np.linalg.norm(delta[0],2)**2+np.linalg.norm(delta[1],2)**2+np.linalg.norm(delta[2],2)**2
        norm2Delta=np.sqrt(norm2Delta)
        W=[W[0]+delta[0],W[1]+delta[1],W[2]+delta[2]]

        cycle=BF.store(PAy)
        if cycle:
            W[0]+=np.random.rand(p*nn).reshape([p,nn])-0.5
            W[1]+=np.random.rand(p*nn).reshape([p,nn])-0.5
            W[2]+=np.random.rand(p*nn).reshape([nn,p])-0.5

        mutex.acquire()
        if cycle:
            print("**** Zyklus entdeckt! *****")
        if i%1==0:
            print("---------------------------")
            print("Prozess:",id)
            print("Iter.:  ",i)
            print("Delta:  ",norm2Delta)
        if norm2Delta<0.5:
            print("Lösung gefunden?")
            if checkSolution(W):
                W=PA(W)
                np.save("solution_"+str(n)+"_"+str(id)+str(time.time()),[W[0],W[1],W[2]])
                print(".... Lösung korrekt")
                W=[np.random.rand(p*nn).reshape([p,nn])*2.0-1.0,
                np.random.rand(p*nn).reshape([p,nn])*2.0-1.0,
                np.random.rand(nn*p).reshape([nn,p])*2.0-1.0]
                i=0
                #success.value=1
                #mutex.release()
                #return #diffMap
            else: print(".... keine gültige Lösung")
        if i==200:
            print(i," cycles -> Reset")

        mutex.release()

        i+=1
        if failed: i=0
        if i==200:
            W=[np.random.rand(p*nn).reshape([p,nn])*2.0-1.0,
            np.random.rand(p*nn).reshape([p,nn])*2.0-1.0,
            np.random.rand(nn*p).reshape([nn,p])*2.0-1.0]
            i=0
    return #diffMap

if __name__ == '__main__':
    start = time.time()
    numOfProc=int(mp.cpu_count()/2)
    print("Anzahl Prozessoren: ",numOfProc)

    mutex=mp.Lock()
    success=mp.Value('i',0)
    procs=[mp.Process(target=diffMap,args=(i,mutex,success)) for i in range(numOfProc)]

    for pp in procs: pp.start()
    for pp in procs: pp.join()

    end = time.time()
    print("Dauer:", end - start)


















#
