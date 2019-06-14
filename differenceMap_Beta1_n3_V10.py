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
    W[0],W[1],W[2],eh,success=bi.findCalcRule(n,p,3000000,W[0],W[1],W[2],limit=0.01,nue=0.1)
    return W,success #PB

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

def init(startVals):
    p=startVals[0][0].shape[0]
    nn=startVals[0][0].shape[1]
    n=int(np.sqrt(nn))
    candidates=[[np.random.rand(p*nn).reshape([p,nn])*2.0-1.0,
    np.random.rand(p*nn).reshape([p,nn])*2.0-1.0,
    np.random.rand(nn*p).reshape([nn,p])*2.0-1.0] for i in range(10)]
    ps=np.ones(len(candidates))

    if len(startVals)>0:
        for i in range(len(candidates)):
            c=candidates[i]
            distances=np.zeros(len(startVals))
            for ii in range(len(startVals)):
                s=startVals[ii]
                d=np.linalg.norm(c[0]-s[0],2)**2
                +np.linalg.norm(c[1]-s[1],2)**2
                +np.linalg.norm(c[2]-s[2],2)**2
                d=np.sqrt(d)
                distances[ii]=d
            ps[i]=np.sum(distances)
    ps/=np.sum(ps)
    print(ps)
    draw=np.random.multinomial(1,ps)
    draw=np.argwhere(draw==1)[0][0]
    startVals.append(candidates[draw])
    return candidates[draw].copy(),startVals #init

def diffMap(id,mutex,success):
    p=23
    n=3
    nn=int(n**2)

    seed=int(time.time())+int(uuid.uuid4())+id
    np.random.seed(seed%135790)
    W=[np.random.rand(p*nn).reshape([p,nn])*2.0-1.0,
    np.random.rand(p*nn).reshape([p,nn])*2.0-1.0,
    np.random.rand(nn*p).reshape([nn,p])*2.0-1.0]
    startVals=[]
    startVals.append(W)
    BF=bf.bloomFilter(2*nn*p,0.00001)
    BFs=[bf.bloomFilter(2*nn*p,0.00001) for b in range(20)]
    i=0
    cyclCnt=-999

    while success.value==0:
        s=False
        while not s:
            PBx,s=PB(W.copy(),id)
            if not s:
                print("   Prz: ",id," BP failed -> reset")
                seed=int(time.time())+int(uuid.uuid4())+id
                np.random.seed(seed%135745)
                W,startVals=init(startVals)
                i=0
        PAy=PA([2.0*PBx[0]-W[0],2.0*PBx[1]-W[1],2.0*PBx[2]-W[2]])
        delta=[PAy[0]-PBx[0],PAy[1]-PBx[1],PAy[2]-PBx[2]]
        W=[W[0]+delta[0],W[1]+delta[1],W[2]+delta[2]]
        norm2Delta=np.linalg.norm(delta[0],2)**2+np.linalg.norm(delta[1],2)**2+np.linalg.norm(delta[2],2)**2
        norm2Delta=np.sqrt(norm2Delta)

        cycle=BF.store(PAy)
        for b in range(len(BFs)):
            if not BFs[b].store(PAy):
                cyclCnt=b
                break

        mutex.acquire()
        if i%1==0:
            print("---------------------------")
            print("Prozess:",id)
            print("Iter.:  ",i)
            print("Delta:  ",norm2Delta)
        if norm2Delta<0.5:
            print("Lösung gefunden?")
            if checkSolution(W):
                W=PA(W)
                np.save("solution_"+str(n)+"_"+str(i)+"_"+str(time.time()),[W[0],W[1],W[2]])
                print(".... Lösung korrekt")
                W,startVals=init(startVals)
                i=0
            else: print(".... keine gültige Lösung")
        if cycle:
            print("**** Zyklus entdeckt! *****")
            print("**** cyclCnt: ",cyclCnt)
        if i>1500 and norm2Delta>3.0:
        #if i==2000:
            print(i," cycles -> Reset")
        mutex.release()

        if cycle:
            W[0]+=(np.random.rand(p*nn).reshape([p,nn])*2.0-1.0)*0.05*cyclCnt
            W[1]+=(np.random.rand(p*nn).reshape([p,nn])*2.0-1.0)*0.05*cyclCnt
            W[2]+=(np.random.rand(p*nn).reshape([nn,p])*2.0-1.0)*0.05*cyclCnt
        if i>1500 and norm2Delta>3.0:
        #if i==2000:
            W,startVals=init(startVals)
            i=0
            BF=bf.bloomFilter(2*nn*p,0.00001)
            BFs=[bf.bloomFilter(2*nn*p,0.00001) for b in range(20)]
        i+=1
    return #diffMap

if __name__ == '__main__':
    start = time.time()
    numOfProc=int(mp.cpu_count())*0+4
    print("Anzahl Prozessoren: ",numOfProc)

    mutex=mp.Lock()
    success=mp.Value('i',0)
    procs=[mp.Process(target=diffMap,args=(i,mutex,success)) for i in range(numOfProc)]

    for pp in procs: pp.start()
    for pp in procs: pp.join()

    end = time.time()
    print("Dauer:", end - start)

########

# for i in range(10):
#     c,startVals=init(startVals)
# startVals[5][0]











#
